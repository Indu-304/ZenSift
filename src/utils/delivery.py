"""Delivery: send the assembled digest via Resend, at most once per day."""

from __future__ import annotations

import logging
import os
from datetime import UTC, date, datetime, timedelta

import resend
from resend.exceptions import ResendError

from src.graph.state import ZenSiftState

logger = logging.getLogger(__name__)

# Resend's shared testing sender; it can only deliver to your Resend account's own address.
FROM_ADDRESS = "ZenSift <onboarding@resend.dev>"

# Resend's error when a key is reused with a different payload: today's digest already went out.
DUPLICATE_ERROR_TYPE = "invalid_idempotent_request"

# Closer than this to delivery time (or past it, e.g. a late cron run), just send now.
MIN_SCHEDULE_LEAD = timedelta(minutes=2)


def idempotency_key(digest_date: date) -> str:
    return f"daily-digest/{digest_date.isoformat()}"


def schedule_time(deliver_at: datetime | None, now: datetime) -> datetime | None:
    if deliver_at is None or deliver_at - now < MIN_SCHEDULE_LEAD:
        return None
    return deliver_at


def send_email(
    *,
    api_key: str,
    to: str,
    subject: str,
    html: str,
    text: str,
    key: str,
    scheduled_at: datetime | None = None,
) -> str:
    resend.api_key = api_key
    params: resend.Emails.SendParams = {
        "from": FROM_ADDRESS,
        "to": [to],
        "subject": subject,
        "html": html,
        "text": text,
    }
    if scheduled_at is not None:
        params["scheduled_at"] = scheduled_at.isoformat()
    response = resend.Emails.send(params, {"idempotency_key": key})
    return response["id"]


def delivery_node(state: ZenSiftState) -> dict:
    api_key = os.getenv("RESEND_API_KEY")
    to = os.getenv("DIGEST_TO_EMAIL")
    if not api_key or not to:
        logger.error("RESEND_API_KEY and DIGEST_TO_EMAIL must be set in the environment.")
        return {"delivery_status": "failed", "errors": ["delivery: missing RESEND_API_KEY or DIGEST_TO_EMAIL"]}

    key = state.get("idempotency_key") or idempotency_key(
        state.get("digest_date") or datetime.now(UTC).date()
    )
    scheduled_at = schedule_time(state.get("deliver_at"), datetime.now(UTC))
    try:
        email_id = send_email(
            api_key=api_key,
            to=to,
            subject=state["email_subject"],
            html=state["email_html"],
            text=state["email_text"],
            key=key,
            scheduled_at=scheduled_at,
        )
    except ResendError as exc:
        if exc.error_type == DUPLICATE_ERROR_TYPE:
            logger.warning("Digest %s was already sent; skipping duplicate.", key)
            return {"delivery_status": "already_sent"}
        logger.error("Resend rejected the digest (%s): %s", exc.error_type, exc.message)
        return {"delivery_status": "failed", "errors": [f"delivery: {exc.error_type}: {exc.message}"]}

    if scheduled_at:
        logger.info(
            "Digest %s accepted by Resend; it will be delivered at %s UTC (Resend id %s).",
            key, f"{scheduled_at:%H:%M}", email_id,
        )
    else:
        logger.info("Digest %s accepted by Resend for immediate delivery (Resend id %s).", key, email_id)
    return {"delivery_status": "sent", "email_id": email_id}
