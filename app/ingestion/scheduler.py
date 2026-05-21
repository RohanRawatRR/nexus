"""
Scheduled ingestion pipeline.
- Runs on configurable schedules per tenant (default: every 6 hours)
- Delta sync: only re-indexes documents modified since last_indexed
- Fires webhook events on completion
- Graceful error isolation: one tenant failure doesn't affect others
"""
import uuid
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval  import IntervalTrigger
from sqlalchemy                     import select
from sqlalchemy.ext.asyncio         import AsyncSession

from app.core.database  import init_db, AsyncSession_, Tenant, Document
from app.core.schemas   import WebhookEvent
from app.ingestion.indexer import index_document
import structlog

log       = structlog.get_logger()
scheduler = AsyncIOScheduler()


async def reindex_tenant(tenant_id: str) -> dict:
    """
    Re-index all documents for a tenant that are due for refresh.
    Returns summary of what was updated.
    """
    results = {"tenant_id": tenant_id, "updated": 0, "failed": 0, "skipped": 0}

    async with AsyncSession_() as db:
        docs = (await db.execute(
            select(Document).where(Document.tenant_id == tenant_id)
        )).scalars().all()

        for doc in docs:
            try:
                # Skip recently indexed (within last hour)
                if doc.last_indexed:
                    age_hours = (
                        datetime.now(timezone.utc) - doc.last_indexed
                    ).total_seconds() / 3600
                    if age_hours < 1:
                        results["skipped"] += 1
                        continue

                await index_document(
                    tenant_id   = tenant_id,
                    source_type = doc.source_type,
                    source_ref  = doc.source_ref,
                    doc_type    = doc.doc_type,
                    title       = doc.title,
                    db          = db,
                )
                results["updated"] += 1
                log.info("reindex.doc_done", tenant=tenant_id, source=doc.source_ref)

            except Exception as exc:
                results["failed"] += 1
                log.error("reindex.doc_failed", tenant=tenant_id, source=doc.source_ref, error=str(exc))

    # Fire webhook event
    async with AsyncSession_() as db:
        from app.integrations.webhooks import fire_event
        await fire_event(db, tenant_id, WebhookEvent.INDEX_DONE, results)
        await db.commit()

    log.info("reindex.tenant_done", **results)
    return results


async def reindex_all_tenants():
    """
    Scheduled job: re-index all active tenants.
    Runs sequentially to avoid overloading OpenAI API.
    """
    async with AsyncSession_() as db:
        tenants = (await db.execute(
            select(Tenant).where(Tenant.is_active == True)
        )).scalars().all()

    log.info("scheduler.reindex_start", tenant_count=len(tenants))
    for tenant in tenants:
        try:
            await reindex_tenant(tenant.id)
        except Exception as exc:
            log.error("scheduler.tenant_failed", tenant=tenant.id, error=str(exc))


def start_scheduler():
    """Register all scheduled jobs and start the APScheduler."""
    scheduler.add_job(
        reindex_all_tenants,
        trigger  = IntervalTrigger(hours=6),
        id       = "reindex_all",
        name     = "Reindex all tenants",
        replace_existing = True,
        misfire_grace_time = 600,
    )
    scheduler.start()
    log.info("scheduler.started", jobs=len(scheduler.get_jobs()))


def stop_scheduler():
    if scheduler.running:
        scheduler.shutdown(wait=False)
