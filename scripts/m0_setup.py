"""M0 live probe setup: two throwaway users with API keys and a few synthetic photos.

Writes credentials and IDs to .env.test (never commit). Re-running is safe: existing users are reused.
"""

import io
import json
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / ".env.test"
URL = os.environ["IMMICH_URL"].rstrip("/") + "/api"
ADMIN = {"x-api-key": os.environ["ADMIN_KEY"]}
USERS = {"a": "tagger-test-a@example.invalid", "b": "tagger-test-b@example.invalid"}


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def ensure_user(label: str, email: str, state: dict) -> None:
    if label in state:
        return
    password = secrets.token_urlsafe(18)
    r = requests.post(f"{URL}/admin/users", headers=ADMIN, json={
        "email": email, "password": password, "name": f"Tagger Test {label.upper()}", "shouldChangePassword": False,
    })
    r.raise_for_status()
    user_id = r.json()["id"]
    token = requests.post(f"{URL}/auth/login", json={"email": email, "password": password}).json()["accessToken"]
    key = requests.post(f"{URL}/api-keys", headers={"Authorization": f"Bearer {token}"},
                        json={"name": "m0-probe", "permissions": ["all"]}).json()["secret"]
    state[label] = {"id": user_id, "email": email, "password": password, "key": key, "assets": []}


def make_jpeg(w: int, h: int, seed: int) -> bytes:
    img = Image.new("RGB", (w, h), ((seed * 70) % 255, 120, 200))
    d = ImageDraw.Draw(img)
    d.rectangle([w * 0.3, h * 0.2, w * 0.6, h * 0.5], fill=(240, 200, 40), outline=(0, 0, 0), width=12)
    d.text((40, 40), f"m0 probe {seed}", fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def upload(label: str, state: dict) -> None:
    user = state[label]
    if user["assets"]:
        return
    now = datetime.now(timezone.utc).isoformat()
    for i, (w, h) in enumerate([(4000, 3000), (3000, 4000), (4000, 3000)]):
        seed = (0 if label == "a" else 10) + i
        r = requests.post(f"{URL}/assets", headers={"x-api-key": user["key"]},
                          files={"assetData": (f"m0-{label}-{i}.jpg", make_jpeg(w, h, seed), "image/jpeg")},
                          data={"fileCreatedAt": now, "fileModifiedAt": now})
        r.raise_for_status()
        user["assets"].append(r.json()["id"])


def main() -> None:
    state = load_state()
    for label, email in USERS.items():
        ensure_user(label, email, state)
        upload(label, state)
    STATE.write_text(json.dumps(state, indent=2))
    print(json.dumps({k: {"id": v["id"], "assets": v["assets"]} for k, v in state.items()}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
