"""M0 live probes against a real Immich (>= 3.3) using the throwaway users from m0_setup.py.

Each probe prints what it observed. Results feed NOTES.md; nothing here is product code.
"""

import io
import json
import os
import time
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
S = json.loads((ROOT / ".env.test").read_text())
URL = os.environ["IMMICH_URL"].rstrip("/") + "/api"
A, B = S["a"], S["b"]


def call(user, method, path, **kw):
    r = requests.request(method, URL + path, headers={"x-api-key": user["key"]}, timeout=60, **kw)
    try:
        body = r.json()
    except ValueError:
        body = r.content if r.headers.get("content-type", "").startswith("image") else r.text
    return r.status_code, body


def show(title, status, body):
    text = body if isinstance(body, (str, bytes)) else json.dumps(body)
    print(f"--- {title}: {status} {text[:400] if not isinstance(text, bytes) else f'<{len(text)} bytes>'}")


def preview_size(user, asset_id, **params):
    status, body = call(user, "GET", f"/assets/{asset_id}/thumbnail", params={"size": "preview", **params})
    return (status, Image.open(io.BytesIO(body)).size) if status == 200 else (status, body)


def faces(user, asset_id):
    return call(user, "GET", "/faces", params={"id": asset_id})


def person(user, name):
    status, body = call(user, "POST", "/people", json={"name": name})
    assert status in (200, 201), body
    return body["id"]


def face_payload(asset_id, person_id, img_w, img_h, box=(0.3, 0.2, 0.6, 0.5)):
    x1, y1, x2, y2 = box
    return {"assetId": asset_id, "personId": person_id, "imageWidth": img_w, "imageHeight": img_h,
            "x": round(x1 * img_w), "y": round(y1 * img_h),
            "width": round((x2 - x1) * img_w), "height": round((y2 - y1) * img_h)}


def main():
    a0, a1, a2 = A["assets"]
    b0, b1, b2 = B["assets"]

    # 1. Preview dimensions (originals: 4000x3000, 3000x4000).
    for aid in (a0, a1):
        print("preview", aid, preview_size(A, aid))

    # 2. Owner creates a person and a MANUAL face on own asset.
    p_a = person(A, "M0 Character")
    status, size = preview_size(A, a0)
    st, body = call(A, "POST", "/faces", json=face_payload(a0, p_a, *size))
    show("A createFace on own asset", st, body)
    show("A faces on a0", *faces(A, a0))

    # 3. Shared album + partner sharing: can B create a face on A's asset?
    st, album = call(A, "POST", "/albums", json={"albumName": "m0-shared", "assetIds": [a0, a1],
                                                 "albumUsers": [{"userId": B["id"], "role": "editor"}]})
    show("A creates shared album", st, album.get("id") if isinstance(album, dict) else album)
    show("A partner-shares with B", *call(A, "POST", "/partners", json={"sharedWithId": B["id"]}))
    p_b_own = person(B, "B own person")
    show("B createFace on A's album asset (own person)", *call(B, "POST", "/faces", json=face_payload(a1, p_b_own, *size)))
    show("B createFace on A's asset with A's person id", *call(B, "POST", "/faces", json=face_payload(a1, p_a, *size)))
    show("B sees faces on A's a0 (before cluster/share)", *faces(B, a0))

    # 4. Cluster group: A invites B, B accepts.
    _, me_a = call(A, "GET", "/users/me")
    show("A /users/me clusterGroupId", 200, me_a.get("clusterGroupId"))
    st, req = call(A, "PUT", f"/cluster-groups/{me_a['clusterGroupId']}/requests", json={"userId": B["id"]})
    show("A invites B to cluster group", st, req)
    _, reqs = call(B, "GET", "/cluster-groups/requests")
    show("B pending requests", 200, reqs)
    for r in reqs if isinstance(reqs, list) else []:
        show("B accepts", *call(B, "POST", f"/cluster-groups/requests/{r['id']}/accept"))
    _, me_b = call(B, "GET", "/users/me")
    show("B clusterGroupId now", 200, me_b.get("clusterGroupId"))

    # 5. Share A's person with B (write role).
    show("A shares person with B", *call(A, "PUT", "/people/users",
                                         json={"personIds": [p_a], "sharedWithIds": [B["id"]], "role": "write"}))
    show("B GET /people", *call(B, "GET", "/people", params={"withHidden": "true"}))
    show("B GET /people/{p_a}", *call(B, "GET", f"/people/{p_a}"))
    show("B faces on A's a0 (after share)", *faces(B, a0))

    # 6. B tags own asset with the shared person group.
    st, bsize = preview_size(B, b0)
    show("B createFace on own b0 with shared person", *call(B, "POST", "/faces", json=face_payload(b0, p_a, *bsize)))
    show("B faces on b0", *faces(B, b0))
    show("A sees person stats", *call(A, "GET", f"/people/{p_a}/statistics"))
    show("B sees person stats", *call(B, "GET", f"/people/{p_a}/statistics"))

    # 7. Rename by B propagates?
    show("B renames shared person", *call(B, "PUT", f"/people/{p_a}", json={"name": "M0 Renamed By B"}))
    show("A reads person after B rename", *call(A, "GET", f"/people/{p_a}"))
    show("B hides person (personal property)", *call(B, "PUT", f"/people/{p_a}", json={"isHidden": True}))
    show("A isHidden after B hides", *call(A, "GET", f"/people/{p_a}"))
    call(B, "PUT", f"/people/{p_a}", json={"isHidden": False})

    # 8. Feature face from a MANUAL face (A picks a0).
    time.sleep(5)
    _, before = call(A, "GET", f"/people/{p_a}")
    show("A set featureFaceAssetId=a0", *call(A, "PUT", f"/people/{p_a}", json={"featureFaceAssetId": a0}))
    time.sleep(5)
    _, after = call(A, "GET", f"/people/{p_a}")
    print("thumbnailPath before/after:", before.get("thumbnailPath"), after.get("thumbnailPath"),
          "updatedAt", before.get("updatedAt"), after.get("updatedAt"))
    st, thumb = call(A, "GET", f"/people/{p_a}/thumbnail")
    print("person thumbnail:", st, len(thumb) if isinstance(thumb, bytes) else thumb)

    # 9. Edited asset: crop a2, compare previews and face coordinate handling.
    show("A crops a2", *call(A, "PUT", f"/assets/{a2}/edits",
                             json={"edits": [{"action": "crop", "parameters": {"x": 1000, "y": 500, "width": 2000, "height": 2000}}]}))
    time.sleep(15)
    print("a2 preview default:", preview_size(A, a2))
    print("a2 preview edited=true:", preview_size(A, a2, edited="true"))
    print("a2 preview edited=false:", preview_size(A, a2, edited="false"))
    st, esize = preview_size(A, a2, edited="true")
    if st == 200:
        show("A createFace on edited a2 (edited-preview coords)", *call(A, "POST", "/faces", json=face_payload(a2, p_a, *esize, box=(0.25, 0.25, 0.75, 0.75))))
        show("A faces on a2", *faces(A, a2))

    # 10. Delete face by B on A's asset (expect forbidden) and soft vs force delete by owner.
    _, fa = faces(A, a0)
    fid = fa[0]["id"] if isinstance(fa, list) and fa else None
    if fid:
        show("B deletes A's face", *call(B, "DELETE", f"/faces/{fid}", json={"force": True}))

    (ROOT / ".env.test").write_text(json.dumps({**S, "probe": {"p_a": p_a, "p_b_own": p_b_own,
                                                                "album": album.get("id") if isinstance(album, dict) else None}}, indent=2))


if __name__ == "__main__":
    main()
