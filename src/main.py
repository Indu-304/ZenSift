"""Entry point: `poetry run python -m src.main [--force-send]`."""

from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

from src.config import load_config
from src.graph.workflow import run

logger = logging.getLogger("zensift")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and send today's ZenSift digest.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--force-send",
        action="store_true",
        help="Bypass today's duplicate-send protection (manual testing only).",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    result = run(load_config(args.config), force_send=args.force_send)

    for error in result.get("errors", []):
        logger.warning("Recovered error: %s", error)
    status = result.get("delivery_status", "failed")
    logger.info(
        "Done: status=%s, articles=%d, degraded=%s",
        status, len(result.get("digest_articles", [])), result.get("degraded", False),
    )
    # Non-zero exit turns the GitHub Actions run red so failures get noticed.
    return 0 if status in ("sent", "already_sent") else 1


if __name__ == "__main__":
    sys.exit(main())
