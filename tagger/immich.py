"""Minimal Immich API client, one instance per user API key. Endpoints verified in NOTES.md (M0)."""

import io
import logging
from collections.abc import Iterator

import requests
from PIL import Image

log = logging.getLogger("immich")


class ImmichError(RuntimeError):
    pass


class Immich:
    def __init__(self, url: str, api_key: str, timeout: float = 60):
        self.base = url.rstrip("/") + "/api"
        self.session = requests.Session()
        self.session.headers.update({"x-api-key": api_key, "Accept": "application/json"})
        self.timeout = timeout

    def _request(self, method: str, path: str, **kw) -> requests.Response:
        r = self.session.request(method, self.base + path, timeout=self.timeout, **kw)
        if r.status_code >= 400:
            raise ImmichError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
        return r

    def get(self, path: str, **params):
        return self._request("GET", path, params=params).json()

    def post(self, path: str, body: dict):
        r = self._request("POST", path, json=body)
        return r.json() if r.content else None

    def put(self, path: str, body: dict):
        r = self._request("PUT", path, json=body)
        return r.json() if r.content else {}

    # --- identity / server -------------------------------------------------------------------
    def version(self) -> tuple[int, int, int, int | None]:
        v = self.get("/server/version")
        return v["major"], v["minor"], v["patch"], v.get("prerelease")

    def me(self) -> dict:
        return self.get("/users/me")

    def cluster_members(self) -> list[str] | None:
        """User ids in the key owner's cluster group, or None if the key lacks clusterGroup.read."""
        try:
            return [u["id"] for u in self.get(f"/cluster-groups/{self.me()['clusterGroupId']}/users")]
        except ImmichError:
            return None

    def uploaded_since(self, since_iso: str, limit: int = 50) -> list[str]:
        """IDs of images uploaded (createdAt, not photo date: old shoots get uploaded late) at/after since_iso."""
        r = self.post("/search/metadata", {"size": limit, "filter": {
            "type": {"eq": "IMAGE"}, "trashedAt": {"eq": None}, "createdAt": {"gte": since_iso}}})["assets"]
        return [a["id"] for a in r["items"]]

    # --- asset selection ---------------------------------------------------------------------
    def search_assets(self, extra_filter: dict | None = None, page_size: int = 500) -> Iterator[dict]:
        """All non-trashed images visible to the key's user via /search/metadata (Immich >= 3.2 filter format)."""
        body = {"size": page_size, "withExif": True,
                "filter": {"type": {"eq": "IMAGE"}, "trashedAt": {"eq": None}, **(extra_filter or {})},
                "orderBy": {"field": "fileCreatedAt", "direction": "asc"}}
        cursor = None
        while True:
            page = self.post("/search/metadata", {**body, "cursor": cursor} if cursor else body)["assets"]
            yield from page["items"]
            cursor = page.get("nextCursor")
            if not cursor:
                return

    def smart_search(self, query: str, limit: int) -> list[dict]:
        body = {"query": query, "size": min(limit, 1000), "filter": {"type": {"eq": "IMAGE"}, "trashedAt": {"eq": None}}}
        return self.post("/search/smart", body)["assets"]["items"][:limit]

    def album(self, album_id: str) -> dict:
        return self.get(f"/albums/{album_id}")

    def shared_albums(self) -> list[dict]:
        return self.get("/albums", isShared="true")

    def partners_sharing_with_me(self) -> list[dict]:
        return self.get("/partners", direction="shared-with")

    # --- media -------------------------------------------------------------------------------
    def edits(self, asset_id: str) -> list:
        return self.get(f"/assets/{asset_id}/edits").get("edits", [])

    def preview(self, asset_id: str, edited: bool) -> Image.Image:
        """The preview Immich gives its own ML (1920 px long side). Edited assets need edited=true (see NOTES.md)."""
        r = self._request("GET", f"/assets/{asset_id}/thumbnail",
                          params={"size": "preview", "edited": "true" if edited else "false"})
        img = Image.open(io.BytesIO(r.content))
        img.load()
        return img.convert("RGB")
