# immich-fursuit-tagger — Project Plan (v2)

A sidecar container that finds fursuit heads in an Immich library, recognizes which character they are, and writes them into Immich's **People** as normal faces via the public API.

Stock Immich server and stock `immich-machine-learning` stay **completely untouched**: buffalo_l keeps handling human faces, and both images update normally.

---

## 1. Goals

- Detect fursuit heads in photos and cluster them by character.
- Write results into Immich as faces with `sourceType: MANUAL`, attached to People. These survive "Refresh faces" jobs.
- **Multi-user, same results for everyone:** one shared character registry. Every configured user who can see a photo gets that character in their own People list (section 7).
- Run acceptably on an **Intel N100** (4 E-cores, small Intel iGPU), and use a discrete GPU automatically if one is added later.
- Low maintenance: prebuilt images, automatic dependency updates, incremental and idempotent processing, and safe to restart at any time.
- Never fight the user. Edits made in the Immich UI are the ground truth.
- Fully reversible: one command removes everything the tagger created.

## 2. Non-goals (v1)

- Video frames: images only.
- A custom web UI or an in-app review queue. Logs, a CLI and an HTML dry-run report are enough.
- Notifications.
- Training or fine-tuning models: use the published checkpoints as-is.
- Modifying the Immich server, database or ML container in any way. **No direct Postgres access.** API only.

## 3. Reference material (read before writing code)

| What | Where | Why |
|---|---|---|
| Furry ML fork (reference implementation) | https://github.com/aibyou0830/immich-ml-furry | Exact pre/post-processing for the detector and embedder. **Port this faithfully; don't guess.** |
| Fursuit head detector (RF-DETR Medium fine-tune) | https://huggingface.co/aibyou0830/rf-detr-for-fur | Detection weights |
| Fursuit embedder head (SupCon MLP) | https://huggingface.co/aibyou0830/supcon-multiview-for-fur | 384→512 projection head; recommended thresholds |
| Embedder backbone (gated) | https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m | Needs `HF_TOKEN` with the DINOv3 license accepted |
| Prior art for the API-sidecar pattern | https://github.com/tedornitier/immich-pet-tagger | How it creates people and faces, and which API-key scopes it needs |
| Immich API | https://api.immich.app and the user's own instance | Verify every endpoint against the **user's running version** |
| Immich multi-user context | Discussions #7362, #19180 and #24290 on immich-app/immich | Confirms that person and face data is per-owner in Immich today |

Recommended settings from the model card, used as defaults: min detection score **0.5**, max cosine distance **0.15**, min faces before creating a character **3**.

## 4. Architecture

```
┌──────────────────────────── immich-fursuit-tagger ────────────────────────────┐
│                                                                               │
│  version guard ─► scheduler ─► access scanner (per user) ─► asset queue        │
│                                                │                              │
│        fetch preview ─► detector ─► quality score ─► embedder                  │
│          (Immich API)    (ONNX)                       (ONNX)                   │
│                                                         │                     │
│     shared character registry ◄── matcher / clusterer ◄─┘                     │
│        (state.sqlite)              (+ burst context)                          │
│                │                                                              │
│        reconciler (all users) ─► mirror writer ─► Immich API, per user key     │
│                                  (people, faces, tags, thumbnails)            │
└───────────────────────────────────────────────────────────────────────────────┘
          model exporter (one-shot, PyTorch) ─► /models/*.onnx
```

### 4.1 Inference backend: ONNX Runtime with pluggable execution providers

PyTorch is used **only** in the one-shot exporter. The long-running service uses ONNX Runtime.

| `INFERENCE_DEVICE` | ORT package | Hardware |
|---|---|---|
| `openvino` (default on N100) | `onnxruntime-openvino` | Intel iGPU (`GPU`) or CPU; needs `/dev/dri` passed through |
| `cuda` | `onnxruntime-gpu` | NVIDIA |
| `rocm` | ROCm or MIGraphX EP, whichever current ORT supports (**verify**) | AMD |
| `cpu` | `onnxruntime` | Fallback |
| `auto` | — | Pick the best provider available; log the choice |

- The ORT wheels conflict with each other, so build separate image tags: `:openvino`, `:cuda`, `:cpu` (and `:rocm` if practical).
- Fuse **DINOv3 backbone + SupCon head + L2 norm** into one embedder ONNX graph.
- Offer FP16 variants for OpenVINO GPU. INT8 is a later experiment.
- Thread count is configurable; default to `cores - 1`.
- **Idle unload:** release the ORT sessions after `MODEL_TTL_MIN` minutes without work, and reload lazily on the next job, because the N100's RAM is shared with Immich.

### 4.2 Model exporter and versioning

- A separate image target, `:exporter`, with CPU PyTorch, `transformers` and RF-DETR. It runs as a compose init service (`service_completed_successfully`) and is a no-op when the manifest already matches.
- It writes `models/manifest.json` with **separate `detector_version` and `embedder_version`** values.
  - Detector changed: re-detect everything.
  - Only the embedder changed: keep the stored boxes, re-fetch previews, crop and **re-embed only**. This is much cheaper on the N100.
- **Parity test:** ONNX outputs must match PyTorch on sample images. Embeddings need cosine similarity ≥ 0.999; boxes need IoU ≥ 0.98.

## 5. Processing pipeline

1. **Select assets per user** (`SCAN_MODE`: `all` | `albums` | `smart`):
   - `all`: the user's own assets, plus assets in albums shared with them, plus partner-shared assets if `INCLUDE_PARTNER=true`.
   - `albums`: only `ALBUM_IDS`. This is the recommended start on the N100.
   - `smart`: use Immich's CLIP smart search (`SMART_QUERY`, top `SMART_LIMIT`) as a cheap pre-filter.
   - Record **which users can see each asset** (`asset_access` table). ML runs **once per asset**, no matter how many users can see it.
2. **Skip** assets already processed with the current model versions and an unchanged asset.
3. **Fetch the preview** and record its pixel dimensions, because face boxes are submitted in that coordinate space (**verify**).
4. **Detect** heads; drop anything below `MIN_SCORE` or with a shorter side under 64 px.
5. **Quality score** for each crop, combining detector confidence, crop size and sharpness (variance of the Laplacian). Crops at or above `REF_QUALITY_MIN` can become **reference** embeddings. Lower-quality crops can still be assigned to a character but never become references, so one bad match can't drag a character's cluster off course.
6. **Embed** each crop, giving a 512-d L2-normalized vector.
7. **Match** (section 6), then **write mirrors** (section 7).

## 6. Matching and clustering

- **Gallery:** only `is_reference = true` detections that are assigned to a character.
- **Assignment:** k-NN with cosine distance. Assign if the nearest neighbor is ≤ `MAX_DISTANCE` and the top-k majority agrees.
- **Burst context** (tiebreaker only): if a detection is borderline (≤ `MAX_DISTANCE + BURST_MARGIN`) and the same character was confidently matched in another asset taken within `BURST_WINDOW_MIN` minutes (EXIF capture time, any owner), assign it but mark it `via_burst` and **never** make it a reference.
- **Pending pool:** periodically run DBSCAN-style clustering (cosine, `eps = MAX_DISTANCE`, `min_samples = MIN_FACES`) over reference-quality pending crops. A cluster that reaches `MIN_FACES` becomes a **new character**.
- Use brute-force numpy search behind an interface, so `sqlite-vec` can be swapped in later.

## 7. Multi-user design: shared characters, mirrored people

**Decided in M0 (Option A, verified live on 3.3.0-rc.2; see NOTES.md):** Immich ≥ 3.3 has cluster groups plus person sharing. A face points to a shared *person group*; each user has their own person row (name, hidden, favorite) in that group. `PUT /people/users` shares a group with users in the same cluster group, and renames propagate natively. Faces can only be created on assets **the caller owns**.

**The approach:** the tagger owns a single **character registry**. Each character is **one Immich person group**.

- **Users opt in** by providing a scoped API key (`users.yaml`: user label → key) and joining **one shared cluster group** in the Immich UI. This is a one-time step and also merges their human-face clustering; the README must say so. At startup the tagger checks that all keys report the same `clusterGroupId`; if not, it goes read-only for writes involving those users.
- **Creating a character:** the first user who owns an asset with a confirmed detection creates the person (`POST /people`) and immediately shares it with all other configured users (`PUT /people/users`, role `write`).
- **Faces:** for each detection, the **asset owner's** key creates one face pointing at the character's group ID. Everyone who can see the asset (album or partner) sees it labeled automatically.
- **Coverage:** complete for owners. Shared-album and partner viewers see the label on the owner's photo. No duplicate boxes, because there is only one face per detection.

### 7.1 Sync rules (edits in any account apply to everyone)

| User action in Immich | Tagger response |
|---|---|
| Renames the person | Native: Immich propagates the name to all writable owners. The tagger only reads the name back into the registry. |
| Merges two people | Merge the characters globally and remap the group ID (merge semantics across shared people still to be verified in M5). |
| Reassigns one of the tagger's faces | Move the detection to that character for everyone, and update the shared gallery. |
| Deletes one of the tagger's faces | Mark the detection `rejected` globally and **never recreate it**. |
| Hides a person, or changes favorites or birthday | Per-user preference; **not** synced. |
| Manually tags a face the tagger doesn't own | Leave it alone. |

### 7.2 Writer rules

- **Idempotent.** Store every face, person and tag ID the tagger creates, per user.
- **Overlap** (IoU > `OVERLAP_IOU`) with an existing face: skip if it's one of the tagger's faces or a user's MANUAL face. For buffalo_l faces, follow `OVERLAP_POLICY=skip|replace` (default `skip`).
- **Auto-tag:** add the tag `TAG_NAME` (default `fursuit`) to every asset with a confirmed detection, in each user's account, so photos are searchable even before the character is named.
- **Best thumbnail:** for each user, set their person row's featured face (`featureFaceAssetId`) to that character's highest-quality reference crop the user can see. Update it when a better one appears, unless the user has manually changed the thumbnail (detect this via the person's `updatedAt` and stop touching it).
- **Edited assets:** if `GET /assets/{id}/edits` is non-empty, detect on `thumbnail?size=preview&edited=true` and submit face boxes in edited-preview coordinates.
- `DRY_RUN=true` writes nothing to Immich.

## 8. Safety

### 8.1 Immich version guard

- On startup and before every write pass: read the server version and probe each required endpoint and payload shape.
- If the version is outside `TESTED_IMMICH_RANGE`, or a probe fails, switch to **read-only mode**: keep detecting and storing, but write nothing to Immich. Log loudly and report the container as unhealthy.
- `ALLOW_UNTESTED_IMMICH=true` overrides this.

### 8.2 Undo

- `tagger undo [--since DATE] [--user LABEL]` deletes every face, person and tag the tagger created in the selected scope, using the stored IDs. It never touches anything it didn't create.
- `undo` sets a **paused** flag so the loop doesn't immediately recreate everything. `tagger resume` clears it.
- `undo --dry-run` lists what would be removed.

## 9. State (SQLite, on a volume, WAL mode, migrations on startup)

- `users(label PK, immich_user_id, enabled)`
- `assets(asset_id PK, owner_id, taken_at, updated_at, detector_version, embedder_version, processed_at, status)`
- `asset_access(asset_id, user_label, via[own|album|partner])`
- `characters(id PK, person_group_id NULL, name NULL, created_at, merged_into NULL)`
- `detections(id PK, asset_id, x, y, w, h, img_w, img_h, edited, score, quality, is_reference, via_burst, embedding BLOB, character_id NULL, status[pending|assigned|rejected], face_id NULL)`
- `shares(character_id, user_label, thumbnail_locked)`: which users the person group has been shared with
- `mirror_tags(asset_id, user_label, tag_id)`
- `meta(key PK, value)`: schema version, paused flag, scan cursors

## 10. Configuration

- **Connection:** `IMMICH_URL`, `USERS_FILE=/data/users.yaml`, `HF_TOKEN` (exporter only).
- **Hardware:** `INFERENCE_DEVICE`, `THREADS`, `MODEL_TTL_MIN=10`.
- **Scanning:** `SCAN_MODE`, `ALBUM_IDS`, `SMART_QUERY`, `SMART_LIMIT`, `INCLUDE_PARTNER=false`.
- **Recognition:** `MIN_SCORE=0.5`, `MAX_DISTANCE=0.15`, `MIN_FACES=3`, `REF_QUALITY_MIN`, `BURST_WINDOW_MIN=10`, `BURST_MARGIN=0.05`.
- **Writing:** `OVERLAP_IOU=0.5`, `OVERLAP_POLICY=skip`, `TAG_NAME=fursuit`.
- **Safety:** `TESTED_IMMICH_RANGE=>=3.3.0-rc.2 <4`, `ALLOW_UNTESTED_IMMICH=false`.
- **Schedule:** `SCAN_INTERVAL_MIN=30`, `ACTIVE_HOURS` (optional).
- **Other:** `DRY_RUN=false`, `LOG_LEVEL=info`.

## 11. CLI

`tagger run` · `tagger scan --once [--album ID] [--limit N]` · `tagger report --out report.html` · `tagger bench` · `tagger undo [--since] [--user] [--dry-run]` · `tagger resume` · `tagger reembed` · `tagger status`

## 12. Deployment and maintenance

- **Prebuilt images:** a GitHub Actions matrix builds `:openvino`, `:cuda`, `:cpu` and `:exporter` and pushes them to GHCR with version tags and `latest`. ZimaOS pulls the images; nothing is built on the N100. CI runs the test suite before pushing. **Model weights are never baked into images.**
- **Renovate** (or Dependabot) for Python dependencies, base images and GitHub Actions. Patch updates auto-merge only when CI passes.
- `docker-compose.yml`: an `exporter` init service, plus the `tagger` service on `:openvino` with `/dev/dri` and volumes for `/models` and `/data`. It should be importable through ZimaOS. Document how to reach Immich over its compose network and over the host IP.
- `docker-compose.gpu.yml` override for CUDA, plus a commented AMD example.
- Healthcheck: last successful loop timestamp and version-guard state.

## 13. Milestones (stop for review where marked ⏸)

1. **M0 Recon** ⏸: read the fork, the model cards, pet-tagger and the live API spec. Write `NOTES.md` covering:
   - the exact pre/post-processing
   - verified endpoints and payloads, and the box coordinate space
   - **the multi-user tests:** can user B create a face on user A's asset that B sees via a shared album or partner sharing; does it appear in B's People; what does A see?
   - the endpoints for tags and for featured faces

   **Stop and report.**
2. **M1 Export + parity:** exporter, versioned manifest, parity tests.
3. **M2 Runtime + bench:** provider selection, idle unload, `tagger bench` on CPU and OpenVINO.
4. **M3 Dry run** ⏸: per-user access scan, detection, quality scoring, clustering with burst context, `report.html`. No writes. **Stop for threshold review.**
5. **M4 Writer (single user):** mirrors, faces, tags, thumbnails, overlap policy, version guard, undo and resume.
6. **M5 Multi-user + reconciler:** mirrors across users and the full sync-rules table.
7. **M6 Packaging + CI:** GHCR matrix builds, Renovate, compose files, README with ZimaOS steps and troubleshooting.

## 14. Testing

- Unit tests: box math, quality scoring, matching, burst logic, clustering, and the reconciler/sync state machine (every row of 7.1).
- Mock the Immich API with recorded responses from the user's version, including **two users**.
- Parity tests for the ONNX export.
- Integration tests against a throwaway Immich instance with two users, a shared album and partner sharing.

## 15. Licensing

- The fork is **AGPL-3.0**. If code is ported from it, license this project AGPL-3.0.
- The DINOv3-derived weights and exported ONNX models are governed by the DINOv3 License. Keep them local, never commit them or bake them into images, and add "Built with DINOv3" to the README.
- RF-DETR is Apache-2.0; keep its notices.
