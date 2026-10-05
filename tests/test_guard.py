from tagger.config import Config
from tagger.writer import REQUIRED_SCOPES, guard


class Fake:
    def __init__(self, version, perms=("all",), group="g1"):
        self._v, self._p, self._g = version, list(perms), group

    def version(self):
        return self._v

    def get(self, path, **_):
        return {"permissions": self._p}

    def me(self):
        return {"clusterGroupId": self._g}


def test_guard_version_range_and_scopes(monkeypatch):
    monkeypatch.delenv("ALLOW_UNTESTED_IMMICH", raising=False)
    cfg = Config()
    assert guard(cfg, {"a": Fake((3, 3, 0, 2))}) is None          # tested RC
    assert guard(cfg, {"a": Fake((3, 3, 0, None))}) is None       # the release sorts after its RCs
    assert guard(cfg, {"a": Fake((3, 4, 1, None))}) is None
    assert "outside" in guard(cfg, {"a": Fake((3, 3, 0, 1))})     # older RC
    assert "outside" in guard(cfg, {"a": Fake((3, 2, 4, None))})
    assert "outside" in guard(cfg, {"a": Fake((4, 0, 0, None))})
    assert "lacks" in guard(cfg, {"a": Fake((3, 3, 0, None), perms=sorted(REQUIRED_SCOPES - {"face.create"}))})
    assert "cluster" in guard(cfg, {"a": Fake((3, 3, 0, None)), "b": Fake((3, 3, 0, None), group="g2")})
    monkeypatch.setenv("ALLOW_UNTESTED_IMMICH", "true")
    assert guard(Config(), {"a": Fake((4, 0, 0, None))}) is None
