# M0 Recon Notes

Date: 2026-10-05. Sources read:

- `aibyou0830/immich-ml-furry` @ `275e957` (2026-07-30)
- `tedornitier/immich-pet-tagger` v1.8.0 @ `64ce9a3`
- `rfdetr` 1.8.3 wheel (the version pinned in the fork's `uv.lock`)
- Immich server source at tags `v3.2.4` (latest stable) and `v3.3.0-rc.2`, plus the OpenAPI spec on `main` (3.3.0-rc.0)

Not done yet: **live tests against the user's Immich instance.** Everything below comes from reading source code. The items marked **LIVE** still need to be confirmed on the running server.

---

## 0. Headline: Immich changed, and section 7 of the plan needs revisiting

The plan assumes "people and faces are per-owner; nothing is shared". That was true up to Immich 3.1. It is no longer true:

| Version | Change | PR |
|---|---|---|
| 3.2.0 | **Cluster groups.** Users can join a shared cluster group. Faces now point to a `person_group`, and `person` became a per-user row `(ownerId, personGroupId)` holding name, hidden, favorite, birthDate, color and thumbnail. Immich's own ML clustering searches embeddings across the whole cluster group. | #30739, #30965 |
| 3.3.0 (rc.2, released 2026-10-02; not yet stable) | **People management / person sharing.** `PUT /people/users` shares a person group with other users **in the same cluster group**, with role `read`, `write` or `admin`. A DB trigger (`person_user_after_insert`) creates a `person` row for each recipient in the **same** group and copies the name. Shared people always show up in the recipient's People list. A rename (name/birthDate are "shared properties") propagates to all owners with write access. | #31620 |

In API terms, a person "id" is now the `personGroupId`. `POST /people` always creates a **new** group.

**Consequence:** on Immich ≥ 3.3, the plan's "mirror person per user" design maps directly onto native Immich features. There is one person group per character, shared with every tagger user. Renames sync natively. See section 5.

---

## 1. Detector pre/post-processing (port exactly)

Source: `immich_ml/models/fursuit_recognition/detection.py` and `rfdetr/detr.py::predict`.

- Input: the Immich **preview** image (Immich sends the preview to ML; the tagger fetches the same preview). Decoded with PIL and converted to RGB.
- `F.to_tensor` gives RGB float values in [0,1], CHW layout.
- `torchvision.transforms.functional.resize(t, [576, 576])` on a **tensor**: a plain **stretch** with no letterbox and no aspect-ratio preservation. The tensor path uses bilinear interpolation with `antialias=True`. **The ONNX preprocessing must reproduce this in numpy/PIL.** The parity test in M1 will show whether PIL's bilinear filter is close enough.
- Normalize with mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- The model produces `pred_logits` (B, 300, C) and `pred_boxes` (B, 300, 4), with boxes as cxcywh normalized to [0,1].
- Postprocess (`models/postprocess.py`): `sigmoid(logits)`, flatten, take the top-k with `num_select=300`, convert boxes to xyxy, scale by the **original** (w, h), and clamp.
- Keep detections with `score > threshold` (strict `>`; default 0.5).
- Class filter is `{"fursuit"}`; the model has a single class.
- Then the fork's own **NMS with IoU threshold 0.1** (very aggressive: boxes overlapping more than 10% are suppressed), sorted by score in descending order.
- The fork's checkpoint config: `RFDETRMedium`, `resolution=576`, `patch_size=16`, `num_windows=2`, `num_queries=300`, `num_select=300`, and `num_classes=1`. Confirm in M1 whether the logits have 1 or 2 slots.

ONNX export: `RFDETRMedium(pretrain_weights=ckpt, num_classes=1).export(output_dir, opset_version=17, dynamic_batch=…)` is built in. Its outputs are `dets` (cxcywh, normalized) and `labels` (logits). We implement the postprocessing (sigmoid, top-k, xyxy, NMS) in numpy.

## 2. Embedder pre/post-processing (port exactly)

Source: `immich_ml/models/fursuit_recognition/recognition.py` and the SupCon model card.

- Crop from the same decoded preview: `floor(x1), floor(y1), ceil(x2), ceil(y2)`, clamped to the image. **Skip** the crop if its shorter side is under 64 px. No padding or margin is added around the box.
- Convert the crop to a PIL RGB image, then `transforms.Resize((512, 512))`: a stretch using PIL bilinear with antialias. Then `ToTensor` and ImageNet normalization (same mean and std as above).
- Backbone: `facebook/dinov3-vits16-pretrain-lvd1689m` (`DINOv3ViTModel`, gated). Features come from `pooler_output`, falling back to `last_hidden_state[:, 0]` (CLS) if that is missing. The features are 384-d.
- Head: `Linear(384,1024) → Tanh → Dropout(0.3) → Linear(1024,1024) → Tanh → Dropout(0.3) → Linear(1024,512)`. The state-dict keys are `head.N.*`; strip the `head.` prefix. Run in eval mode so Dropout is a no-op.
- L2-normalize the output, giving a 512-d vector. Compare vectors with cosine distance.
- Postprocess dedup: boxes are rounded to integers, and when two boxes are identical, keep the higher score.
- Fusing the backbone, head and L2 norm into one ONNX graph is straightforward. Input is (B, 3, 512, 512). Note: 512 px input to ViT-S/16 gives 1024 patches, which is heavy on the N100. Benchmark this in M2.

Defaults from the model cards: min score 0.5, max cosine distance 0.15, min faces 3.

Versions pinned by the fork: `rfdetr 1.8.3`, `torch 2.12.1`, `transformers 5.13.0`.

## 3. Verified endpoints and payloads (Immich 3.2.4 / 3.3.0-rc)

| Purpose | Endpoint | Permission | Notes |
|---|---|---|---|
| Version | `GET /server/version` | none | `{major, minor, patch, prerelease}` |
| Who am I | `GET /users/me` | `user.read` | includes `clusterGroupId` (admin DTO; **LIVE**: check that the non-admin DTO has it too) |
| Key scopes | `GET /api-keys/me` | — | `permissions[]`; use this to validate scopes at startup |
| List assets | `POST /search/metadata` | `asset.read` | ≥ 3.2.0 uses a structured `filter` with `{eq}`, `{gte}` and similar, plus cursor pagination via `assets.nextCursor`. The old flat fields are removed in v4. Add `trashedAt: {eq: null}` explicitly. |
| Smart pre-filter | `POST /search/smart` | `asset.read` | same filter format |
| Albums | `GET /albums?isShared=` and `GET /albums/{id}` | `album.read` | |
| Partners | `GET /partners?direction=shared-with` | `partner.read` | |
| Preview | `GET /assets/{id}/thumbnail?size=preview[&edited=]` | `asset.view` | **LIVE**: preview pixel size (usually 1440 on the long side) and the default for `edited`. Use the **edited** preview so it matches createFace (below). |
| Faces on asset | `GET /faces?id=<assetId>` | `face.read` | `[{id, boundingBoxX1..Y2, imageWidth, imageHeight, sourceType: machine-learning\|exif\|manual, person}]`, showing only faces visible to the caller |
| Create face | `POST /faces` | `face.create` | `{assetId, personId(=group id), x, y, width, height, imageWidth, imageHeight}`. Returns **201 with an empty body**; to get the new face's ID, `GET /faces?id=` and match on person ID and box. Stored as `sourceType: manual`. |
| Delete face | `DELETE /faces/{id}` with body `{force: bool}` | `face.delete` | `force=false` soft-deletes |
| Reassign face | `PUT /faces/{id}` with body `{id: personId}` | `face.update` | |
| People | `GET /people` (paged; `withHidden`), `POST /people`, `PUT /people/{id}`, `DELETE /people/{id}` | `person.*` | `PersonUpdateDto.featureFaceAssetId` is an **asset ID**; the server picks this person's face on that asset |
| Merge | `POST /people/merge {ids}` (3.3); legacy `POST /people/{id}/merge` | `person.merge` | |
| Share person (3.3) | `PUT /people/users {personIds, sharedWithIds, role}`, `GET /people/users`, `DELETE /people/users` | `person.update` / `person.read` | everyone must be in the **same cluster group** |
| Cluster groups | `/cluster-groups/requests`, `/{id}/requests`, `/requests/{id}/accept`, `/{id}/users`, `/{id}/leave` | `clusterGroup.*` | joining is invite plus accept; a user-side action |
| Tags | `PUT /tags {tags:[name]}` (upsert, returns list) and `PUT /tags/{id}/assets {ids}`; `DELETE /tags/{id}/assets {ids}` | `tag.create`, `tag.asset` | the response is a per-ID list; `error: "duplicate"` means the asset is already tagged |

### Box coordinate space

`createFace` stores `x, y, x+w, y+h` together with the given `imageWidth` and `imageHeight`, and clients scale by those values. **Any consistent coordinate space works**, so we send boxes in preview pixels plus the preview dimensions.

Exception: if the asset has **edits** (crop or rotate), the server treats the coordinates as relative to the **edited preview**. It rescales them by `asset.width / imageWidth` and inverts the edits. So the tagger must detect on the **edited** preview whenever edits exist. **LIVE**: confirm what `thumbnail?size=preview` returns by default for edited assets.

### Permissions (access.ts, both versions)

- `POST /faces` needs **`AssetUpdate` on the asset, which is owner-only** (shared-album and partner access are not enough), **and** a `person` row owned by the caller for that group.
- `DELETE /faces/{id}` and reassign need the caller to **own the asset** the face is on.
- `PersonRead`/`PersonUpdate` on 3.2.4: caller owns a person row in that group. On 3.3 this is role-based via `person_user`.

### Safety facts

- "Refresh faces" and "reset recognition" delete or unassign **only `machine-learning` faces** (person.service.ts lines 373, 419, 520). The tagger's MANUAL faces survive.
- Unnamed people with fewer than `minimumFaces` faces (a per-user preference, default 3) are **hidden** from the People list. A named person with at least one face, or any shared person, is shown.
- People-list face counts include only faces on assets the user **owns** (3.2.4 and 3.3 `getAllForUser`). In 3.3, #31435 adds partner assets to the person timeline, but **LIVE** testing is needed to see how that affects counts.

API key scopes needed per user: `user.read`, `asset.read`, `asset.view`, `asset.update` (implied by face create? **LIVE**), `album.read`, `partner.read`, `person.read`, `person.create`, `person.update`, `person.delete`, `person.merge`, `person.reassign`, `face.read`, `face.create`, `face.update`, `face.delete`, `tag.read`, `tag.create`, `tag.asset`, `tag.delete`.

## 4. Multi-user answers (from source; LIVE confirmation pending)

1. **Can user B create a face on A's asset that B sees through a shared album or partner sharing?** **No**, on both 3.2.4 and 3.3. `AssetUpdate` is owner-only.
2. **Does it appear in B's People?** Not applicable for B-created faces. For A-created faces on A's asset: B sees the character only if B holds a `person` row in the **same person group**. That is possible only through 3.3 person sharing, or through Immich's own ML clustering (which cannot happen for MANUAL faces, because they have no embedding).
3. **What does A see?** Only faces created on A's own assets. There are no "foreign" boxes, because nobody else can write to A's assets.

## 5. Proposed revision of section 7 (needs the user's decision)

**Option A, native sharing (Immich ≥ 3.3.0 stable):**

- Users who opt in join **one cluster group**. This is a one-time UI step per user and is documented in the README.
- Each character is **one person group**, created by a "primary" user key and shared with all other tagger users via `PUT /people/users` with role `write`.
- Each user's key creates faces **only on assets that user owns**, all pointing at the same group ID. Shared-album viewers see the named face on the owner's photo, because they hold a person row in the same group.
- Rename and merge sync natively. Hidden, favorite and color stay per-user, which matches table 7.1.
- The `mirrors` table collapses to `characters.person_group_id`. Much less reconciler code.
- Cost: requires 3.3 (rc today), and all users must be in the same cluster group. Joining a cluster group also merges Immich's buffalo_l human-face clustering across those users (a side effect to tell them about).

**Option B, plan as written (works on 3.2.x):** one mirror person per user per character, created by each user's own key. Faces only on owned assets. The sync rules in 7.1 are implemented by the tagger. Shared-album viewers do **not** see the character on photos they don't own. This is the plan's documented "if no" limitation.

Recommendation: **Option A.** It covers more, needs less code, and Immich's sync rules are already built. Set `TESTED_IMMICH_RANGE=>=3.3.0` and wait for 3.3.0 stable before M4. M1–M3 do not depend on this choice.

## 6. LIVE results (2026-10-05, maintainer's instance)

**Decision: Option A** (user, 2026-10-05).

Setup: the production instance was upgraded from 3.2.4 to **3.3.0-rc.2** through ZimaOS, after taking a DB backup. Only the server and ML image tags changed; the ML container was on v2.7.2 and now matches. The asset count was unchanged after the upgrade, and a copy of the backup was kept off-server.

Tests used two throwaway users, `tagger-test-a` and `tagger-test-b`, created with `scripts/m0_setup.py`. The probes are `scripts/m0_probe.py` and `scripts/m0_scopes.py`; credentials are in `.env.test` (gitignored).

| Test | Result |
|---|---|
| Preview size | **1920 px on the long side** (4000×3000 gives 1920×1440). The plan's guess of 1440 was wrong. |
| `POST /faces` on own asset | `201`, empty body. `GET /faces` returns the box exactly as sent (preview space) with `sourceType: manual`. |
| B creates a face on A's asset (shared album editor plus partner) | `400 "Not found or no asset.update access"`, with B's own person and with A's person ID alike. **Owner-only, confirmed.** |
| B deletes A's face | `400 "no face.delete access"` |
| B views A's asset before the person is shared | B sees the face box with `person: null` |
| Cluster group | A invites with `PUT /cluster-groups/{id}/requests {userId}`; B accepts with `POST /cluster-groups/requests/{id}/accept`, which returns 204. `/users/me.clusterGroupId` is available to non-admins. |
| Share person (`PUT /people/users`, role `write`) | B's `GET /people` immediately lists it, with `otherPeople`/`sharedBy` filled in. B now sees **A's face on A's asset labeled with the name**. |
| B tags B's own asset with the shared group ID | `201`. Statistics: A sees 1 asset; B sees 2 (B's own plus A's, which B can see through the album/partner). |
| B renames | **Propagates to A.** B hiding the person does **not** affect A, as expected. |
| `featureFaceAssetId` = asset with a MANUAL face | Works; `updatedAt` changes and the thumbnail regenerates. The person's own `updatedAt` can be used to detect a thumbnail change by the user. |
| Edited (cropped) asset | `thumbnail?size=preview` **defaults to the unedited image**; `edited=true` gives the cropped preview. createFace with edited-preview coordinates round-trips consistently: GET returns the same relative box in edited space. **Rule: if `GET /assets/{id}/edits` is non-empty, detect on `edited=true` and submit edited-space coordinates.** Visual confirmation is deferred to M4. |
| Tags | `PUT /tags` upsert returns `[{id,…}]`; tagging again returns `success:false, error:"duplicate"`; untag works. |
| Search | `POST /search/metadata` with `filter` + `orderBy` and `POST /search/smart` both work on 3.3-rc. |
| Soft face delete (`force:false`) | The face disappears from `GET /faces`. |
| **Minimum key scopes** | `user.read asset.read asset.view album.read partner.read person.read person.create person.update person.delete person.merge person.reassign face.read face.create face.update face.delete tag.read tag.create tag.asset tag.delete`. All calls passed. **`asset.update` is not needed.** Joining a cluster group is a user action in the UI, so the tagger needs no `clusterGroup.*` scopes. |

Production facts: the instance's two real users are in **different cluster groups**. For multi-user Option A they must join one group, which also merges their buffalo_l human-face clustering.

Still open: merge semantics across shared persons (`POST /people/merge`). This goes to M5.

## 7. Licensing confirmations

- The fork is AGPL-3.0. We port its pre/post-processing, so this project is **AGPL-3.0**.
- RF-DETR fine-tune: Apache-2.0; retain notices and state the modification.
- SupCon head and exported embedder ONNX: DINOv3 License. Ship "Built with DINOv3" and never redistribute the weights.
- pet-tagger license not yet checked; we only take ideas (API patterns), no code.

---

# M1 Export + parity (2026-10-05)

- `python -m exporter.export` writes `detector-<v>.onnx`, `embedder-<v>.onnx` and `manifest.json` under `MODELS_DIR`. Versions are hashes of the pinned source revisions plus `EXPORT_CODE_VERSION`. Re-running is a no-op. The exit code is 1 while the embedder is missing, so the compose init service fails visibly.
- Pinned sources: detector `aibyou0830/rf-detr-for-fur@ce47a3d`; head `aibyou0830/supcon-multiview-for-fur@8b3fd67`; backbone resolved to the current `main` sha of `facebook/dinov3-vits16-pretrain-lvd1689m`.
- Detector ONNX: input `input` (1, 3, 576, 576); outputs `dets` (1, 300, 4, cxcywh normalized) and `labels` (1, 300, **2**). Slot 0 is `fursuit`; slot 1 is never `fursuit` and is dropped.
- Preprocessing gotcha: resizing **uint8** with PIL gave a worst IoU of 0.957 against torch. torchvision resizes the float tensor, so we resize each channel in PIL `"F"` mode. Max difference is now 2e-4.
- Embedder ONNX: `pixel_values` (batch, 3, 512, 512) maps to `embedding` (batch, 512), L2-normalized.
- Parity on 16 CC-licensed Wikimedia Commons fursuit photos (`work/samples`, not committed): detector worst IoU **0.9999** over 17 boxes, score diff 0.0000; embedder worst cosine **1.00000**.

# M2 Runtime + bench (2026-10-05)

- `tagger/runtime.py`: `INFERENCE_DEVICE=auto` picks cuda, then rocm (MIGraphX), then openvino, then cpu. OpenVINO uses GPU/FP16 when `/dev/dri` exists, with a compile cache in `OV_CACHE_DIR` (default `/data/ov_cache`). Sessions load lazily and are released by `release_if_idle()` after `MODEL_TTL_MIN`.
- CI pushes `ghcr.io/nugamus/immich-fursuit-tagger:{cpu,openvino,cuda,exporter}` (public) plus `<flavor>-<sha>` tags. The OpenVINO image installs Intel's OpenCL runtime debs (same versions as Immich's ML image), because Debian trixie no longer packages them.
- **OpenVINO GPU cannot compile the dynamic-batch embedder** (`rank().is_static()`), so the embedder is exported with static batch 1 (`EMBEDDER_EXPORT_VERSION=2`). Detector and embedder export versions are independent, so this triggered re-embedding only.
- **ZimaOS adds `cpus: "1.00"` to every container in a custom app.** The compose files must raise it for the tagger.

`tagger bench` with 10 synthetic 1920×1440 previews, on the N100 under ZimaOS:

| Device | Detect per photo | Embed per head | Session load | Peak RSS |
|---|---|---|---|---|
| CPU EP, 1-CPU limit (ZimaOS default) | 6.2 s | 4.9 s | 2.6 s + 6.1 s | 1.2 GB |
| CPU EP, 3 CPUs | 1.25 s | 1.15 s | 0.5 s + 0.4 s | 750 MB |
| OpenVINO iGPU FP16, cold | 0.43 s | 0.31 s | 10.1 s + 4.1 s (compile) | 960 MB |
| OpenVINO iGPU FP16, cached | **0.42 s** | **0.31 s** | **0.5 s + 0.3 s** | 610 MB |

That works out to about 1 s per photo with one head on the iGPU, or about 10 minutes for a 600-photo library. Still to check: that FP16 on the GPU doesn't shift embeddings (compare against CPU during the M3 dry run).

# M3 Dry run (2026-10-05) ⏸ threshold review

- `tagger scan --once [--album ID] [--limit N]` runs the per-user access scan, detection, quality scoring, embedding and recognition (gallery k-NN, then clustering of the pending pool, then burst context). `tagger recluster` redoes recognition from the stored embeddings with the current thresholds (dry run only). `tagger report --out report.html` writes a self-contained report. No Immich writes.
- Full library on the N100 iGPU: 588 photos in about 8 minutes, 0 errors. A CPU run on the desktop found 565 heads against the iGPU's 567, so FP16 on the iGPU is fine.

Problems found on the real library, and their fixes:
1. **Two suits that pose together were merged** (purple suit and grey/white husky; 33 photos had both in one "character"). DBSCAN's single linkage chained them through crops containing parts of both heads. Fix: a **same-photo cannot-link rule**. `split_conflicts()` splits any cluster holding two detections from one photo (2-means, with the conflicting pair pinned to opposite sides), and the match and burst passes never give a character to two heads in one photo.
2. **Geese became a character.** The detector fires on geese with scores of 0.51–0.6, against a median of 0.94 for real heads. Fix: `REF_SCORE_MIN=0.8`, so only confident heads can found a character or act as gallery references. Low-score heads can still join an existing character.
3. **Burst context was too loose.** About 10 of 15 burst assignments were right; the misses sat at distance 0.18–0.19. `BURST_MARGIN` was lowered from 0.05 to 0.03.
4. Founding with `CLUSTER_EPS=0.20` found two more real suits (an orange fox and a green/brown one) but also built a 27-crop junk cluster of back-of-head shots. The default stays at 0.15, and `CLUSTER_EPS` remains configurable.

Final defaults on the real library: 7 characters (husky 255, teal deer 114, purple 96, plus 7, 5, 4 and 4 crops). No photo has a character twice. 39–50 heads stay pending (other suits, geese, back-of-head shots). Reviewing the 36 lowest-quality crops of each character found about 2 wrong crops in the purple suit; the rest were correct or ambiguous back/side views. Mistakes cluster in low-quality crops, and deleting a face in Immich will mark it rejected for good (M4).

## M3 review round 2 (user feedback, 2026-10-05)

The user labelled specific crops. Detection IDs refer to the dry-run DB:
- 444 and 446 belong to the husky but were in the purple suit. 444 was a **bug**: the 2-means split can strand a point with no neighbour on its side (distance 0.272). Fix: re-run DBSCAN inside each split cluster (`split_conflicts(..., eps)`); stranded points become noise. Both now land in the husky.
- 181, 184 and 185 are husky side views at 0.24–0.28. They stayed pending because they were "low quality" (0.47–0.55) and therefore couldn't be references, even though their 6–8 burst siblings are 0.03 apart. Fix: `REF_QUALITY_MIN` 0.5 → 0.3 and `REF_SCORE_MIN` 0.8 → 0.7. All 8 crops that newly joined the husky are correct, and none left any character.
- Geese and ducks (score ≤ 0.64) and one human face (0.59) remain unable to found a character. The report now collapses pending heads further than 0.5 from every character into "Probably not a fursuit".
- Characters #2 and #7 are the same suit: back views against front views. No safe automatic merge signal was found; #6/#7 has the same profile (never in one photo, 0.23 apart) and those are different suits. The user merges in Immich, and M5 must follow merges.
- 240 and 244 (belonging to #6) are still pending: 0.24 from #6, but #6 has only 4 references.
- Tried and rejected:
  - **Flip test-time augmentation:** no measurable gain, at 2× the embedding cost.
  - **Relative (margin) assignment:** 12/15 correct, but it assigns unseen suits to the nearest character.
  - **Pending-neighbour open-set check:** known side views form tight pending groups just like unseen suits do.
- Design principle from this round: **over-split rather than over-merge.** An extra person costs one merge click; a wrong merge is painful to undo.

## Immich human people (same day)

On fursuit photos, buffalo_l found mostly tiny background faces (median width 2–7% of the frame). The result was 16 unnamed people, the same person split several times, and blurry thumbnails.
- System config change: `minScore` 0.7 → 0.8 and `maxDistance` 0.5 → 0.55, followed by forced face detection and recognition. That gave 6 people.
- `tagger people` (new, opt-in): sets each person's featured face to their largest face, unless the user changed it (tracked via the person's `updatedAt`). `--hide-below-pct 4` hides people whose largest face is under 4% of the photo width (1 person hidden). `--share` re-applies `PUT /people/users {type: everyone}`, because Immich's "everyone" share is a snapshot.
- Result: 4 visible people, all with sharp thumbnails, and shared with Nugamus.

## M3 review round 3 (2026-10-05)

- The user flagged six #6 crops at 0.19–0.29 that weren't grouped. Root cause: #242 is a #6 member with a detector score of 0.65 (just under `REF_SCORE_MIN`), so it is not a gallery reference, yet it sits 0.08–0.13 from 240, 243 and 244. Fix: **`_session_pass`**. A pending head within `MAX_DISTANCE` of any assigned head (reference or not) taken within `SESSION_MIN` (30) minutes joins that character, transitively, if it links to exactly one character and passes the same-photo rule. 240, 243 and 244 now belong to #6.
- 275, 278 and 284 (the same suit from the side) are about 0.2 from the *husky* and 0.25–0.37 from #6's other views; the model genuinely confuses that view. A looser "same session, relative" rule sent them to the husky, so it was rejected. `CLUSTER_EPS` 0.15 → 0.17 founds them as their own 3-crop character, which the user then merges. 0.17 forms no junk cluster (0.20 did).
- #2 (husky from behind) has no clean automatic merge signal. Time continuity gave 3 of 5 #2 shots within 5 s of a husky shot, but also 2 of 7 for two genuinely different suits. Left to the user's merge in Immich.
- Final: 8 characters, of which 2 are expected user merges (#2 into the husky, the 3-crop side group into #6). 0 same-photo duplicates. 45 heads pending.

# M4 Writer (2026-10-05)

`tagger run` is the loop: scan, version guard, write pass, optional people housekeeping, then sleep for `SCAN_INTERVAL_MIN`. `tagger write` runs one pass, `undo [--since] [--user] [--dry-run]` removes what the tagger created, `resume` clears the pause, `status` prints counts, and `health` backs the Docker HEALTHCHECK.

- **People:** one person group per character, created by the configured user who owns most of its photos and shared with all other configured users through `PUT /people/users` (role `write`). It is created unnamed, and the user names it in Immich.
- **Faces:** `POST /faces` with the asset owner's key, in preview coordinates. These are edited-preview coordinates for edited assets, which is what `createFace` expects. The new face ID is found by diffing `GET /faces` before and after. A face is skipped (`skip_reason = 'overlap'`) if it overlaps a face the tagger didn't create with IoU > `OVERLAP_IOU`.
- **Tags:** the `TAG_NAME` tag is upserted per user. The tagger records whether it created the tag, so `undo` deletes only tags it made.
- **Thumbnails:** per user, the best-quality written face that user can see. If the person's `updatedAt` changes without a name change, the user picked a thumbnail, so `thumbnail_locked` is set for that user only. Gotcha: PUT returns `updatedAt` as `...812Z`, while GET returns `...799867+00:00`, so the comparison must use a fresh GET. Without that, every person locked after our own update.
- **User deletes a tagger face:** the detection becomes `rejected` permanently. Rejected and written detections survive re-detection, and new boxes overlapping them (IoU > 0.5) are dropped.
- **Change detection uses `thumbhash|isEdited`, not `updatedAt`.** Our own faces and tags bump `updatedAt`, which caused a full re-detect after every write.
- **Version guard:** requires Immich ≥ 3.3.0-rc.2 and < 4, every key holding the required scopes, and all users in one cluster group. Otherwise the tagger stays read-only, logs `READ-ONLY`, and health turns unhealthy.

Sandbox test (users tagger-test-a and tagger-test-b, 16 Commons photos, an album shared from A to B):
- 1 person and 10 faces written. A sees 10 photos, B sees the 4 shared ones, and both see the labelled face on A's photo. 10 photos tagged.
- The second and third passes were no-ops.
- A deleted a face: it became a rejection and was not recreated. B renamed the person: A sees the new name, and neither user's thumbnail locked. A picked a thumbnail: only A locked.
- `undo` removed 9 faces, 1 person, the tag on 10 photos and the tag itself, then paused. `resume` followed by `run` recreated everything except the rejected face.
