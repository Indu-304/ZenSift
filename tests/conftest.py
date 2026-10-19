import pytest


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "zensift.db"
    monkeypatch.setenv("ZENSIFT_DB_PATH", str(db_path))
    return db_path


@pytest.fixture(autouse=True)
def block_feed_network(mocker):
    return mocker.patch(
        "src.utils.fetcher.urlopen",
        side_effect=AssertionError("Real network call attempted in tests"),
    )
