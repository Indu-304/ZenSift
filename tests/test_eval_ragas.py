from scripts.eval_ragas import RESULTS_END, RESULTS_START, results_markdown, update_readme

SCORES = [
    {"faithfulness": 1.0, "summary_score": 0.6},
    {"faithfulness": 0.75, "summary_score": 0.4},
    {"faithfulness": float("nan"), "summary_score": 0.5},
]
MEANS = {"faithfulness": 0.875, "summary_score": 0.5}


def test_results_table_reports_means_and_fully_faithful_count():
    table = results_markdown(SCORES, MEANS, "gemini-x", "2026-10-20")

    assert "2026-10-20 · 3 live articles · editor and judge: `gemini-x`" in table
    assert "| Mean faithfulness | **0.88** |" in table
    assert "| Fully faithful summaries | 1 / 2 |" in table
    assert "| Mean summary score | **0.50** |" in table


def test_update_readme_replaces_only_the_marked_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n{RESULTS_START}\nold\n{RESULTS_END}\nfooter\n", encoding="utf-8")

    update_readme("new table", readme)
    update_readme("newer table", readme)

    assert readme.read_text(encoding="utf-8") == (
        f"# Title\n{RESULTS_START}\nnewer table\n{RESULTS_END}\nfooter\n"
    )
