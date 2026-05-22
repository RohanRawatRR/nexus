"""
Analytics service.
Stores query logs and computes aggregated metrics for the dashboard.
"""
import uuid
from datetime       import datetime, timezone, timedelta
from collections    import defaultdict

from sqlalchemy     import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import QueryLog
from app.core.schemas  import AnalyticsSummary, QueryResponse


async def log_query(
    db:        AsyncSession,
    tenant_id: str,
    user_id:   str | None,
    request_q: str,
    response:  QueryResponse,
) -> str:
    """Persist a query log entry. Returns log ID."""
    log_id = str(uuid.uuid4())
    entry  = QueryLog(
        id             = log_id,
        tenant_id      = tenant_id,
        user_id        = user_id,
        question       = request_q,
        query_type     = response.query_type,
        answer         = response.answer,
        sources        = response.sources,
        iterations     = response.iterations,
        latency_ms     = response.latency_ms,
        confidence     = response.confidence,
        faithfulness   = response.faithfulness,
        answer_relevancy = response.answer_relevancy,
    )
    db.add(entry)
    await db.commit()
    return log_id


async def get_analytics(
    db:        AsyncSession,
    tenant_id: str,
    days:      int = 30,
) -> AnalyticsSummary:
    """Aggregate query metrics for the analytics dashboard."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    logs = (await db.execute(
        select(QueryLog).where(
            QueryLog.tenant_id == tenant_id,
            QueryLog.created_at >= since,
        ).order_by(QueryLog.created_at)
    )).scalars().all()

    if not logs:
        return AnalyticsSummary(
            period              = f"last_{days}_days",
            total_queries       = 0,
            avg_latency_ms      = 0.0,
            avg_confidence      = 0.0,
            avg_faithfulness    = None,
            avg_answer_relevancy= None,
            top_query_types     = {},
            queries_per_day     = [],
        )

    total            = len(logs)
    avg_latency      = sum(l.latency_ms or 0 for l in logs) / total
    avg_confidence   = sum(l.confidence  or 0 for l in logs) / total

    faith_scores     = [l.faithfulness    for l in logs if l.faithfulness    is not None]
    rel_scores       = [l.answer_relevancy for l in logs if l.answer_relevancy is not None]
    avg_faith        = sum(faith_scores) / len(faith_scores) if faith_scores else None
    avg_rel          = sum(rel_scores)   / len(rel_scores)   if rel_scores   else None

    type_counts: dict[str, int] = defaultdict(int)
    daily_counts: dict[str, int] = defaultdict(int)
    for l in logs:
        type_counts[l.query_type or "general"] += 1
        day = l.created_at.strftime("%Y-%m-%d")
        daily_counts[day] += 1

    queries_per_day = [
        {"date": day, "count": count}
        for day, count in sorted(daily_counts.items())
    ]

    return AnalyticsSummary(
        period               = f"last_{days}_days",
        total_queries        = total,
        avg_latency_ms       = round(avg_latency, 1),
        avg_confidence       = round(avg_confidence, 3),
        avg_faithfulness     = round(avg_faith, 3) if avg_faith else None,
        avg_answer_relevancy = round(avg_rel,   3) if avg_rel   else None,
        top_query_types      = dict(type_counts),
        queries_per_day      = queries_per_day,
    )
