import pytest
from pydantic import ValidationError

from src.config import ZenSiftConfig, load_config


def test_repo_config_yaml_is_valid():
    config = load_config("config.yaml")

    assert len(config.interests) >= 1
    assert 0 < config.sifter.similarity_threshold < 1
    assert config.digest.hard_drop_tone < config.digest.avoid_drop_tone


def test_defaults_apply_when_sections_are_omitted(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("interests: [open source LLMs]\n")

    config = load_config(path)

    assert config.avoid == []
    assert config.sifter.max_articles == 8
    assert config.digest.hard_drop_tone == 0.15


def test_email_in_config_is_rejected_because_it_moved_to_env(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text('email: "me@example.com"\ninterests: [x]\n')

    with pytest.raises(ValidationError, match="email"):
        load_config(path)


def test_typos_are_rejected():
    with pytest.raises(ValidationError):
        ZenSiftConfig(interests=["x"], sifter={"similarity_treshold": 0.5})


@pytest.mark.parametrize("bad", [{"interests": []}, {}, {"interests": ["x"], "sifter": {"max_articles": 50}}])
def test_invalid_configs_are_rejected(bad):
    with pytest.raises(ValidationError):
        ZenSiftConfig.model_validate(bad)
