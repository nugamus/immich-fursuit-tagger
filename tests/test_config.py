import pytest

from tagger.config import Config, parse_keys


def test_parse_keys_accepts_commas_and_newlines():
    assert parse_keys("alice=aaa, bob=b=b\n carol = ccc ,") == {"alice": "aaa", "bob": "b=b", "carol": "ccc"}


def test_parse_keys_rejects_garbage():
    with pytest.raises(ValueError):
        parse_keys("just-a-key")


def test_users_requires_keys(monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.delenv("API_KEYS_FILE", raising=False)
    with pytest.raises(ValueError):
        Config().users()
    monkeypatch.setenv("API_KEYS", "alice=aaa")
    assert Config().users() == {"alice": "aaa"}
