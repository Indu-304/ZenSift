from datetime import UTC, date, datetime, timedelta

import pytest
from resend.exceptions import ResendError

from src.utils import delivery
from src.utils.delivery import FROM_ADDRESS, delivery_node, idempotency_key, schedule_time

STATE = {
    "digest_date": date(2026, 10, 20),
    "email_subject": "ZenSift · 3 calm reads · Tue 20 Oct",
    "email_html": "<p>hi</p>",
    "email_text": "hi\n",
}


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.setenv("DIGEST_TO_EMAIL", "reader@example.com")


@pytest.fixture
def send(block_real_email):
    """The conftest guard, turned into a well-behaved fake Resend."""
    block_real_email.side_effect = None
    block_real_email.return_value = {"id": "email_123"}
    return block_real_email


def resend_error(error_type: str, code: int = 409) -> ResendError:
    return ResendError(code=code, error_type=error_type, message=f"{error_type} happened", suggested_action="")


def test_idempotency_key_format():
    assert idempotency_key(date(2026, 10, 20)) == "daily-digest/2026-10-20"


def test_send_passes_daily_idempotency_key(env, send):
    result = delivery_node(STATE)

    assert result == {"delivery_status": "sent", "email_id": "email_123"}
    params, options = send.call_args.args
    assert options == {"idempotency_key": "daily-digest/2026-10-20"}
    assert params == {
        "from": FROM_ADDRESS,
        "to": ["reader@example.com"],
        "subject": STATE["email_subject"],
        "html": STATE["email_html"],
        "text": STATE["email_text"],
    }


def test_explicit_key_in_state_wins(env, send):
    delivery_node({**STATE, "idempotency_key": "daily-digest/2026-10-20/manual-abc"})

    assert send.call_args.args[1] == {"idempotency_key": "daily-digest/2026-10-20/manual-abc"}


def test_same_day_reruns_reuse_the_same_key(env, send):
    delivery_node(STATE)
    delivery_node(STATE)

    keys = [c.args[1]["idempotency_key"] for c in send.call_args_list]
    assert keys == ["daily-digest/2026-10-20"] * 2


def test_from_address_is_resend_testing_sender():
    assert FROM_ADDRESS.endswith("<onboarding@resend.dev>")


def test_api_key_is_set_on_the_sdk(env, send, monkeypatch):
    monkeypatch.setattr(delivery.resend, "api_key", None)

    delivery_node(STATE)

    assert delivery.resend.api_key == "re_test_key"


def test_duplicate_rejection_means_already_sent_not_failure(env, send):
    send.side_effect = resend_error("invalid_idempotent_request")

    assert delivery_node(STATE) == {"delivery_status": "already_sent"}


@pytest.mark.parametrize(
    ("error_type", "code"),
    [("validation_error", 422), ("invalid_api_key", 403), ("concurrent_idempotent_requests", 409)],
)
def test_other_resend_errors_fail_without_raising(env, send, error_type, code):
    send.side_effect = resend_error(error_type, code)

    result = delivery_node(STATE)

    assert result["delivery_status"] == "failed"
    assert error_type in result["errors"][0]


@pytest.mark.parametrize("missing", ["RESEND_API_KEY", "DIGEST_TO_EMAIL"])
def test_missing_env_fails_before_calling_resend(env, send, monkeypatch, missing):
    monkeypatch.delenv(missing)

    result = delivery_node(STATE)

    assert result["delivery_status"] == "failed"
    send.assert_not_called()


# --- scheduling --------------------------------------------------------------

NOW = datetime(2026, 10, 20, 1, 35, tzinfo=UTC)
EIGHT_AM_IST = datetime(2026, 10, 20, 2, 30, tzinfo=UTC)


def test_future_delivery_time_is_scheduled():
    assert schedule_time(EIGHT_AM_IST, NOW) == EIGHT_AM_IST


@pytest.mark.parametrize(
    "now",
    [EIGHT_AM_IST - timedelta(minutes=1), EIGHT_AM_IST, EIGHT_AM_IST + timedelta(hours=2)],
)
def test_late_or_imminent_runs_send_immediately(now):
    assert schedule_time(EIGHT_AM_IST, now) is None


def test_no_delivery_time_sends_immediately():
    assert schedule_time(None, NOW) is None


def test_scheduled_send_passes_iso_time_to_resend(env, send, mocker):
    mocker.patch.object(delivery, "schedule_time", return_value=EIGHT_AM_IST)

    result = delivery_node({**STATE, "deliver_at": EIGHT_AM_IST})

    assert result["delivery_status"] == "sent"
    assert send.call_args.args[0]["scheduled_at"] == "2026-10-20T02:30:00+00:00"
