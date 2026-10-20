"""RAGAS evaluation of the Editor: are summaries faithful to, and representative of, the article?

    poetry run python -m scripts.eval_ragas                       # read today's articles, snapshot them, score
    poetry run python -m scripts.eval_ragas --snapshot PATH.jsonl # re-score the same articles after a prompt change

Metrics (0.0 to 1.0, higher is better):
    faithfulness   share of the summary's claims that the article text supports (catches hallucination)
    summarization  how much of the article's key information the summary covers, penalising length

Nothing is emailed and the sent-articles ledger is never touched.
"""

from __future__ import annotations

import os

# RAGAS sends usage analytics unless this is set before it is imported.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

import argparse
import csv
import logging
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

from src.agents.editor import editor_node
from src.agents.llm import DEFAULT_CHAT_MODEL, build_chat_model
from src.agents.planner import planner_node
from src.agents.sifter import sifter_node
from src.config import load_config
from src.graph.state import Article
from src.graph.workflow import initial_state
from src.utils.fetcher import fetcher_node
from src.utils.reader import reader_node
from src.utils.sanitizer import sanitizer_node

logger = logging.getLogger("zensift.eval")

EVAL_DIR = Path("data/eval")
TASK = "Summarize this news article calmly and objectively in at most three sentences."
METRICS = ("faithfulness", "summary_score")
README = Path("README.md")
RESULTS_START = "<!-- eval-results:start -->"
RESULTS_END = "<!-- eval-results:end -->"


def collect_articles(config_path: str) -> list[Article]:
    """Run the pipeline up to the reader, skipping dedup so the eval never depends on the ledger."""
    state = initial_state(load_config(config_path))
    state.update(planner_node(state))
    state.update(fetcher_node(state))
    state["fresh_articles"] = state.get("raw_articles", [])
    state.update(sifter_node(state))
    state.update(reader_node(state))
    return state["sifted_articles"]


def save_snapshot(articles: list[Article], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(a.model_dump_json() + "\n" for a in articles), encoding="utf-8")


def load_snapshot(path: Path) -> list[Article]:
    lines = path.read_text(encoding="utf-8").splitlines()
    # Drop summaries from older runs so the current prompt is what gets scored.
    return [Article.model_validate_json(line).model_copy(update={"summary": None}) for line in lines if line]


def summarize(articles: list[Article]) -> list[Article]:
    state = {"sifted_articles": articles}
    state.update(sanitizer_node(state))
    return editor_node(state)["digest_articles"]


def score(articles: list[Article]) -> list[dict]:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, SummarizationScore
    from ragas.run_config import RunConfig

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": TASK,
                "response": a.summary,
                # The editor only ever sees `body`, so judge the summary against exactly that.
                "retrieved_contexts": [a.body],
                "reference_contexts": [a.body],
            }
            for a in articles
        ]
    )
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), SummarizationScore()],
        # Gemini 3.x ignores temperature, so don't let RAGAS set it.
        llm=LangchainLLMWrapper(build_chat_model(), bypass_temperature=True),
        # Low concurrency keeps the judge under Gemini free-tier rate limits.
        run_config=RunConfig(max_workers=2, timeout=180),
    )
    return result.scores


def report(articles: list[Article], scores: list[dict], out_dir: Path) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"results-{datetime.now(UTC):%Y%m%d-%H%M%S}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", *METRICS, "tone_score", "summary", "url"])
        for article, row in zip(articles, scores, strict=True):
            writer.writerow(
                [article.source, *(row.get(m) for m in METRICS), article.tone_score, article.summary, article.url]
            )

    print(f"\n{'faith':>6} {'summ':>6}  source / summary")
    for article, row in zip(articles, scores, strict=True):
        cells = " ".join(f"{row.get(m, float('nan')):>6.2f}" for m in METRICS)
        print(f"{cells}  {article.source or '-'}: {(article.summary or '')[:90]}")

    # A metric is NaN when the judge failed on that row; leave those out of the average.
    valid = {m: [v for r in scores if (v := r.get(m)) is not None and v == v] for m in METRICS}
    means = {m: mean(v) if v else float("nan") for m, v in valid.items()}
    print("\nmean   " + "   ".join(f"{m}={v:.2f}" for m, v in means.items()))
    print(f"Saved {path}")
    return means


def results_markdown(scores: list[dict], means: dict[str, float], model: str, run_date: str) -> str:
    faithfulness = [v for r in scores if (v := r.get("faithfulness")) is not None and v == v]
    fully_faithful = sum(1 for v in faithfulness if v >= 0.999)
    return (
        f"Last run: {run_date} · {len(scores)} live articles · editor and judge: `{model}`\n\n"
        "| Metric | Result |\n"
        "|---|---|\n"
        f"| Mean faithfulness | **{means['faithfulness']:.2f}** |\n"
        f"| Fully faithful summaries | {fully_faithful} / {len(faithfulness)} |\n"
        f"| Mean summary score | **{means['summary_score']:.2f}** |"
    )


def update_readme(block: str, path: Path = README) -> None:
    text = path.read_text(encoding="utf-8")
    start, end = text.index(RESULTS_START) + len(RESULTS_START), text.index(RESULTS_END)
    path.write_text(f"{text[:start]}\n{block}\n{text[end:]}", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--snapshot", type=Path, help="Re-score articles saved by an earlier run.")
    parser.add_argument("--limit", type=int, default=8, help="Score at most this many articles.")
    parser.add_argument(
        "--min-faithfulness", type=float, default=0.8, help="Exit non-zero if mean faithfulness is below this."
    )
    parser.add_argument("--update-readme", action="store_true", help="Write the results into README.md.")
    args = parser.parse_args(argv)

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    for noisy in ("httpx", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.snapshot:
        articles = load_snapshot(args.snapshot)
    else:
        articles = collect_articles(args.config)
        snapshot = EVAL_DIR / f"snapshot-{datetime.now(UTC):%Y%m%d-%H%M%S}.jsonl"
        save_snapshot(articles, snapshot)
        logger.info("Saved %d articles to %s; pass it to --snapshot to re-score them.", len(articles), snapshot)

    articles = [a for a in articles if a.body][: args.limit]
    if not articles:
        logger.error("No readable articles to evaluate.")
        return 1

    summarized = summarize(articles)
    scores = score(summarized)
    means = report(summarized, scores, EVAL_DIR)
    if args.update_readme:
        model = os.getenv("GEMINI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
        update_readme(results_markdown(scores, means, model, f"{datetime.now(UTC):%Y-%m-%d}"))
        logger.info("Updated the results in %s.", README)
    if not means["faithfulness"] >= args.min_faithfulness:
        logger.error("Mean faithfulness %.2f is below %.2f.", means["faithfulness"], args.min_faithfulness)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
