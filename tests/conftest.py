import pytest


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "zensift.db"
    monkeypatch.setenv("ZENSIFT_DB_PATH", str(db_path))
    return db_path


@pytest.fixture(autouse=True)
def no_real_api_keys(monkeypatch):
    # Without keys, any accidentally un-mocked client fails instead of calling a real API.
    # The Google SDK also reads GOOGLE_API_KEY, so both names are cleared.
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "RESEND_API_KEY", "DIGEST_TO_EMAIL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def block_real_email(mocker):
    return mocker.patch(
        "resend.Emails.send", side_effect=AssertionError("Real email send attempted in tests")
    )


@pytest.fixture(autouse=True)
def block_feed_network(mocker):
    return mocker.patch(
        "src.utils.fetcher.urlopen",
        side_effect=AssertionError("Real network call attempted in tests"),
    )


@pytest.fixture(autouse=True)
def block_article_network(mocker):
    mocker.patch(
        "src.utils.reader.urlopen",
        side_effect=AssertionError("Real network call attempted in tests"),
    )
    mocker.patch(
        "src.utils.reader.gnewsdecoder",
        side_effect=AssertionError("Real network call attempted in tests"),
    )
