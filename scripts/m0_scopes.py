"""M0: run every call the tagger needs with a key limited to the proposed scope list."""

import json
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
S = json.loads((ROOT / ".env.test").read_text())
URL = os.environ["IMMICH_URL"].rstrip("/") + "/api"
A, B = S["a"], S["b"]

SCOPES = ["user.read", "asset.read", "asset.view", "album.read", "partner.read",
          "person.read", "person.create", "person.update", "person.delete", "person.merge", "person.reassign",
          "face.read", "face.create", "face.update", "face.delete",
          "tag.read", "tag.create", "tag.asset", "tag.delete"]


def login(user):
    return requests.post(f"{URL}/auth/login", json={"email": user["email"], "password": user["password"]}).json()["accessToken"]


def main():
    token = login(A)
    key = requests.post(f"{URL}/api-keys", headers={"Authorization": f"Bearer {token}"},
                        json={"name": "m0-scoped", "permissions": SCOPES}).json()
    h = {"x-api-key": key["secret"]}
    a0, a1, _ = A["assets"]

    def check(title, method, path, **kw):
        r = requests.request(method, URL + path, headers=h, timeout=60, **kw)
        print(f"{r.status_code} {title} {r.text[:200] if r.status_code >= 300 or 'search' in path or 'tags' in path else ''}")
        return r

    check("users/me", "GET", "/users/me")
    check("api-keys/me", "GET", "/api-keys/me")
    check("search/metadata filter", "POST", "/search/metadata",
          json={"size": 2, "filter": {"type": {"eq": "IMAGE"}, "trashedAt": {"eq": None}},
                "orderBy": {"field": "fileCreatedAt", "direction": "asc"}})
    check("search/smart filter", "POST", "/search/smart", json={"query": "yellow rectangle", "size": 2})
    check("albums shared", "GET", "/albums", params={"isShared": "true"})
    check("partners shared-with", "GET", "/partners", params={"direction": "shared-with"})
    check("thumbnail preview", "GET", f"/assets/{a1}/thumbnail", params={"size": "preview", "edited": "true"})
    p = check("create person", "POST", "/people", json={"name": "Scope Probe"}).json()["id"]
    check("create face", "POST", "/faces", json={"assetId": a1, "personId": p, "imageWidth": 1440, "imageHeight": 1920,
                                                   "x": 100, "y": 100, "width": 400, "height": 400})
    faces = check("get faces", "GET", "/faces", params={"id": a1}).json()
    fid = next(f["id"] for f in faces if (f.get("person") or {}).get("id") == p)
    check("share person", "PUT", "/people/users", json={"personIds": [p], "sharedWithIds": [B["id"]], "role": "write"})
    check("people/users", "GET", "/people/users", params={"personId": p})
    check("update person", "PUT", f"/people/{p}", json={"name": "Scope Probe 2", "featureFaceAssetId": a1})
    tags = check("upsert tag", "PUT", "/tags", json={"tags": ["fursuit-m0"]}).json()
    tag_id = tags[0]["id"]
    check("tag asset", "PUT", f"/tags/{tag_id}/assets", json={"ids": [a1]})
    check("tag asset again", "PUT", f"/tags/{tag_id}/assets", json={"ids": [a1]})
    check("untag asset", "DELETE", f"/tags/{tag_id}/assets", json={"ids": [a1]})
    check("soft delete face", "DELETE", f"/faces/{fid}", json={"force": False})
    print("faces after soft delete:", [f["id"] for f in requests.get(f"{URL}/faces", headers=h, params={"id": a1}).json()])
    check("delete person", "DELETE", f"/people/{p}")
    check("delete tag", "DELETE", f"/tags/{tag_id}")
    requests.delete(f"{URL}/api-keys/{key['apiKey']['id']}", headers={"Authorization": f"Bearer {token}"})


if __name__ == "__main__":
    main()
