import sqlite3

from src.utils.deduplicate import get_sent_urls, mark_sent


def test_empty_db_reports_nothing_sent(isolated_db):
    assert get_sent_urls(["https://a"], isolated_db) == set()


def test_mark_then_lookup_roundtrip(isolated_db):
    mark_sent(["https://a", "https://b"], isolated_db)

    assert get_sent_urls(["https://a", "https://c"], isolated_db) == {"https://a"}


def test_mark_sent_is_idempotent(isolated_db):
    mark_sent(["https://a", "https://a"], isolated_db)
    mark_sent(["https://a"], isolated_db)

    with sqlite3.connect(isolated_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sent_articles").fetchone()[0] == 1


def test_empty_inputs_are_noops(isolated_db):
    mark_sent([], isolated_db)
    assert get_sent_urls([], isolated_db) == set()


def test_url_with_sql_metacharacters_is_treated_as_data(isolated_db):
    evil = "https://x/'); DROP TABLE sent_articles; --"
    mark_sent([evil], isolated_db)

    assert get_sent_urls([evil], isolated_db) == {evil}


def test_creates_parent_directory(tmp_path):
    db_path = tmp_path / "nested" / "dir" / "z.db"
    mark_sent(["https://a"], db_path)

    assert db_path.exists()
