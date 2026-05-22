"""
Slack integration.
Handles:
  /nexus <question>     — slash command
  @nexus <question>     — app mention in channels
  DM messages           — direct answers in DM

Nexus replies in the same thread with:
  - Streaming-style progressive updates ("Retrieving docs…", "Analyzing…")
  - Final answer with source list
  - Confidence indicator emoji
"""
import hmac, hashlib, time, asyncio
from typing import Optional

from slack_sdk.web.async_client import AsyncWebClient
from fastapi import Request, HTTPException

from app.core.config  import get_settings
from app.graph.runner import run as rag_run
import structlog

log      = structlog.get_logger()
settings = get_settings()


def get_slack_client() -> Optional[AsyncWebClient]:
    if not settings.slack_bot_token:
        return None
    return AsyncWebClient(token=settings.slack_bot_token)


def verify_slack_signature(request_body: bytes, timestamp: str, signature: str) -> bool:
    """Verify Slack request signature to prevent spoofing."""
    if not settings.slack_signing_secret:
        return True   # dev mode
    base    = f"v0:{timestamp}:{request_body.decode()}"
    expected = "v0=" + hmac.new(
        settings.slack_signing_secret.encode(),
        base.encode(),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


async def handle_slash_command(
    payload:   dict,
    tenant_id: str,
    user_id:   str,
) -> dict:
    """
    Handle /nexus slash command.
    Returns immediate acknowledgment, then sends answer asynchronously.
    Slack requires a response within 3 seconds.
    """
    question = payload.get("text", "").strip()
    channel  = payload.get("channel_id")
    slack_uid = payload.get("user_id")

    if not question:
        return {"text": "Usage: `/nexus <your engineering question>`"}

    client = get_slack_client()
    if not client:
        return {"text": "Slack integration not configured."}

    # Acknowledge immediately (Slack 3s limit)
    asyncio.create_task(
        _answer_async(client, channel, question, tenant_id, user_id, slack_uid)
    )

    return {
        "response_type": "in_channel",
        "text":          f":hourglass_flowing_sand: Searching docs for: *{question}*",
    }


async def handle_app_mention(
    event:     dict,
    tenant_id: str,
    user_id:   str,
) -> None:
    """Handle @nexus mention in a channel. Replies in thread."""
    client   = get_slack_client()
    if not client:
        return

    text     = event.get("text", "")
    channel  = event.get("channel")
    thread_ts = event.get("thread_ts") or event.get("ts")
    slack_uid = event.get("user")

    # Strip bot mention from text
    import re
    question = re.sub(r"<@[A-Z0-9]+>", "", text).strip()
    if not question:
        return

    await _answer_async(client, channel, question, tenant_id, user_id, slack_uid, thread_ts)


async def _answer_async(
    client:    AsyncWebClient,
    channel:   str,
    question:  str,
    tenant_id: str,
    user_id:   str,
    slack_uid: str,
    thread_ts: Optional[str] = None,
) -> None:
    """Run RAG pipeline and post result to Slack."""
    try:
        # Post "thinking" message
        thinking = await client.chat_postMessage(
            channel   = channel,
            thread_ts = thread_ts,
            text      = ":brain: Analyzing your question and searching docs…",
        )

        # Run full RAG
        response = await rag_run(tenant_id, user_id, question)

        # Confidence emoji
        conf  = response.confidence
        emoji = ":large_green_circle:" if conf > 0.8 else ":large_yellow_circle:" if conf > 0.6 else ":red_circle:"

        # Format Slack blocks
        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Q: {question}*"},
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": response.answer[:2900]},
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"{emoji} Confidence: {conf:.0%}  |  "
                            f"Sources: {', '.join(response.sources[:3])}  |  "
                            f"Latency: {response.latency_ms:.0f}ms"
                        ),
                    }
                ],
            },
        ]

        # Replace thinking message
        await client.chat_update(
            channel  = channel,
            ts       = thinking["ts"],
            text     = response.answer[:200],
            blocks   = blocks,
        )

    except Exception as exc:
        log.error("slack.answer_failed", error=str(exc))
        await client.chat_postMessage(
            channel   = channel,
            thread_ts = thread_ts,
            text      = f":x: Failed to retrieve answer: {exc}",
        )
