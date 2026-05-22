"""
Webhook engine.
- Fires events to registered tenant endpoints
- Signs payloads with HMAC-SHA256 (like Stripe/GitHub)
- Retries up to 3 times with exponential backoff
- Verifies delivery receipts
"""
import hashlib, hmac, json, time, uuid, asyncio
from datetime import datetime, timezone

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import WebhookEndpoint
from app.core.schemas  import WebhookEvent
import structlog

log = structlog.get_logger()


def _sign(secret: str, payload: bytes) -> str:
    """HMAC-SHA256 signature — same pattern as Stripe webhooks."""
    return "sha256=" + hmac.new(
        secret.encode(), payload, hashlib.sha256
    ).hexdigest()


async def fire_event(
    db:        AsyncSession,
    tenant_id: str,
    event:     WebhookEvent,
    data:      dict,
) -> None:
    """
    Dispatch an event to all active webhook endpoints for this tenant
    that are subscribed to this event type.
    """
    endpoints = (await db.execute(
        select(WebhookEndpoint).where(
            WebhookEndpoint.tenant_id == tenant_id,
            WebhookEndpoint.is_active == True,
        )
    )).scalars().all()

    for endpoint in endpoints:
        if event.value not in endpoint.events:
            continue
        asyncio.create_task(
            _deliver(endpoint.url, endpoint.secret, event.value, data)
        )


async def _deliver(
    url:    str,
    secret: str,
    event:  str,
    data:   dict,
    max_retries: int = 3,
) -> None:
    """Deliver with retry + exponential backoff."""
    payload = json.dumps({
        "id":         str(uuid.uuid4()),
        "event":      event,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data":       data,
    }).encode()

    signature = _sign(secret, payload)
    headers   = {
        "Content-Type":     "application/json",
        "X-Nexus-Signature": signature,
        "X-Nexus-Event":    event,
    }

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, content=payload, headers=headers)
                if resp.status_code < 300:
                    log.info("webhook.delivered", url=url, event=event)
                    return
                log.warning("webhook.failed", url=url, status=resp.status_code, attempt=attempt)
        except Exception as exc:
            log.error("webhook.error", url=url, error=str(exc), attempt=attempt)

        await asyncio.sleep(2 ** attempt)   # 1s, 2s, 4s

    log.error("webhook.exhausted", url=url, event=event)


def verify_signature(secret: str, payload: bytes, signature: str) -> bool:
    """For incoming webhooks — verify the HMAC signature."""
    expected = _sign(secret, payload)
    return hmac.compare_digest(expected, signature)
