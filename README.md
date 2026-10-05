# immich-fursuit-tagger

Face recognition for fursuits in [Immich](https://immich.app).

Immich's built-in face recognition is great at people and hopeless at fursuits. This runs next to Immich, finds
fursuit heads in your photos, works out which ones belong to the same character, and adds them to Immich's
**People** page like any other person. You name them there, merge them there, and search for them there.

Nothing in Immich is modified. It talks to the normal API with an API key, keeps its own small database, and
everything it adds can be removed again with one command.

## What you get

- A person per fursuit character, with faces boxed on every photo they appear in
- Characters shared with everyone in your Immich cluster group, so a shared album shows the same names to everyone
- A `fursuit` tag on every photo with a recognised suit
- Sharp thumbnails, picked from the clearest shot of each character
- New uploads picked up within a couple of minutes; the models unload again when there's nothing to do
- Optional: the same thumbnail and sharing treatment for Immich's regular (human) people

Corrections you make in Immich stick. Delete a face and it stays deleted. Merge two people and the tagger merges
its characters too, and from then on uses both sets of photos to recognise that suit. Move a face to someone
else and it follows.

## How it works

1. **Detect.** An [RF-DETR](https://github.com/roboflow/rf-detr) model fine-tuned on fursuit heads finds every head in
   the 1920px preview Immich already generates.
2. **Describe.** Each head is cropped and turned into a 512-number fingerprint by a DINOv3 backbone with a small
   head trained to tell suits apart.
3. **Group.** Heads with close fingerprints become a character. Two heads in the same photo are never the same
   character, and shots from the same session can vouch for each other, which catches side and back views.
4. **Write.** Characters become Immich people. Faces are added through the API by each photo's owner.

Both models come from [aibyou0830/immich-ml-furry](https://github.com/aibyou0830/immich-ml-furry) and run as ONNX,
so the container needs no PyTorch and can use an Intel iGPU, an NVIDIA GPU, or just the CPU.

## Requirements

- Immich **3.3** or newer (people sharing and cluster groups landed in 3.3)
- Docker with Compose
- A free [Hugging Face](https://huggingface.co) account that has accepted the
  [DINOv3 license](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m). Approval is manual and can take
  a few hours.
- About 1 GB of RAM while it works. On an Intel N100 a photo takes roughly a second on the iGPU.

## Setup

### 1. Hugging Face token

Once the DINOv3 request is approved, create a **Read** token under
[Settings → Access Tokens](https://huggingface.co/settings/tokens). It's only used once, to download and convert
the models.

### 2. Immich API keys

Every user who **owns** photos you want tagged needs an API key (Immich only lets a photo's owner add faces to it).
People who only look at shared albums don't need one.

In Immich, go to **Account Settings → API Keys → New API Key** and tick these permissions:

```
user.read  asset.read  asset.view  album.read  partner.read
person.read  person.create  person.update  person.delete  person.merge  person.reassign
face.read  face.create  face.update  face.delete
tag.read  tag.create  tag.asset  tag.delete
clusterGroup.read  (optional: lets it share with everyone in the group, not just key holders)
```

### 3. One cluster group

Immich only shares people between users in the same **cluster group**. Pick one user (whoever owns the most
photos) and invite the others from **Account Settings → People cluster group**. Each person accepts the invite
on that same page, in a browser; the mobile app doesn't show it yet.

> Joining a group merges everyone's face recognition, which is the point. Don't press **Reset facial
> recognition** on that page unless you mean it: it deletes every person for everyone in the group.

### 4. Start it

```sh
git clone https://github.com/nugamus/immich-fursuit-tagger
cd immich-fursuit-tagger
cp .env.example .env     # fill in HF_TOKEN and API_KEYS
docker compose up -d
```

The first start downloads and converts the models (a few minutes), then scans the whole library. Follow along
with `docker compose logs -f tagger`.

The compose file expects Immich's Docker network to be called `immich_default`. Check yours with
`docker network ls`, or remove the `networks:` sections and set `IMMICH_URL` to Immich's address instead.

### ZimaOS / CasaOS

ZimaOS doesn't read `.env` files and blanks out `${...}` variables, so paste the values straight in:

1. **App Store → Install Custom App → YAML**, and paste `docker-compose.yml`
2. Replace `${HF_TOKEN}` and `${API_KEYS}` with the real values, and drop the `${...:-default}` parts
3. Change the network name to `immich_immich`, which is what the ZimaOS Immich app uses
4. Change `./models` and `./data` to absolute paths, e.g. `/DATA/AppData/fursuit-tagger/models`
5. Raise the CPU limit to `3.00`. ZimaOS caps new apps at one core, which makes everything three times slower.

## Trying it without writing anything

Set `DRY_RUN=true` and the tagger only reads. To see what it would do:

```sh
docker compose exec tagger python -m tagger report --out /data/report.html
```

Then open `data/report.html`. It shows every character with all of its crops, everything left unassigned, and how far
each head was from its nearest match, which is handy when tuning the settings below. Click a crop to open the
photo in Immich.

## Fixing mistakes

Do it in Immich. The tagger checks for your edits on every pass.

| In Immich you... | The tagger... |
| --- | --- |
| rename a person | keeps the name (Immich passes it on to everyone sharing the person) |
| merge two people | merges the characters and learns from both |
| delete one of its faces | never adds that face again |
| move one of its faces to another fursuit person | moves it there |
| move one of its faces to someone it didn't create | leaves it alone for good |
| pick a thumbnail | stops changing that thumbnail |

The common case is one suit split into two people, for example front views and back views. Merge them once and
future photos of either kind land on the merged person.

## Undo

```sh
docker compose exec tagger python -m tagger undo --dry-run   # what would be removed
docker compose exec tagger python -m tagger undo             # remove it
docker compose exec tagger python -m tagger resume           # start tagging again
```

`undo` deletes the faces, people and tags the tagger created and nothing else, then pauses it so it doesn't put
everything back on the next pass. Use `--since 2026-10-01` or `--user alice` to narrow it down.

## Settings

All settings are environment variables. The defaults were tuned on a real 600-photo convention library.

| Variable | Default | |
| --- | --- | --- |
| `IMMICH_URL` | `http://immich-server:2283` | Where Immich is |
| `API_KEYS` | | `name=key` pairs, comma separated |
| `SCAN_MODE` | `all` | `all`, `albums` (with `ALBUM_IDS`), or `smart` (CLIP search for `SMART_QUERY`) |
| `INFERENCE_DEVICE` | `auto` | `auto`, `openvino`, `cuda`, `rocm` or `cpu` |
| `DRY_RUN` | `false` | Read only |
| `TAG_NAME` | `fursuit` | Tag added to tagged photos; empty disables it |
| `MIN_SCORE` | `0.5` | How sure the detector must be that something is a fursuit head |
| `MAX_DISTANCE` | `0.15` | How close a head must be to a character to join it. Lower is stricter |
| `CLUSTER_EPS` | `0.17` | How close unknown heads must be to each other to start a new character |
| `MIN_FACES` | `3` | Heads needed before a new character is created |
| `REF_SCORE_MIN` | `0.7` | Heads below this never start a character. Keeps birds and human faces out |
| `SESSION_MIN` | `30` | Photos this close in time can vouch for each other |
| `POLL_INTERVAL_SEC` | `120` | How often to look for new uploads |
| `SCAN_INTERVAL_MIN` | `60` | A full pass at least this often, to pick up edits made in Immich |
| `MODEL_TTL_MIN` | `10` | Unload the models after this long without work |
| `SHARE_PEOPLE` | `true` | Also share Immich's human people with everyone in the cluster group |
| `HUMAN_THUMBNAILS` | `true` | Give human people their sharpest face as thumbnail |
| `HIDE_BACKGROUND_PCT` | `0` | Hide human people whose largest face is narrower than this % of the photo (try `4`) |

If one character keeps getting split, raise `MAX_DISTANCE` a little. If different suits get mixed up, lower it.
Splits are cheap to fix (one merge), mix-ups are not, so the defaults lean towards splitting.

## Commands

Run them with `docker compose exec tagger python -m tagger <command>`.

| | |
| --- | --- |
| `status` | Counts: photos, heads, characters, faces written |
| `report --out FILE` | The HTML report described above |
| `scan` | One read-only pass: detect and group, write nothing |
| `write` | One write pass from what's already been scanned |
| `people` | The human-people housekeeping on its own |
| `undo` / `resume` | See [Undo](#undo) |
| `bench` | Time the models on this machine |

## Troubleshooting

**The logs say `READ-ONLY`.** The tagger checks Immich's version, the API key permissions and the cluster group
before every write. The message after `READ-ONLY` names the problem.

**The exporter fails with a 403 or "gated repo".** The DINOv3 license hasn't been approved for the account behind
`HF_TOKEN` yet.

**It's slow.** Check the first log line: it says which device is in use. For the Intel iGPU, `/dev/dri` has to be
passed through and the image has to be `:openvino`. The first run on a new GPU spends about 30 seconds compiling;
after that it's cached.

**Some people can't see the characters.** They need to be in the same cluster group as the photo owner.
Sharing is retried on every pass, so it starts working within the hour of them joining.

## Development

```sh
uv sync --extra exporter --group dev
uv run pytest
```

`MODELS_DIR=work/models uv run python -m exporter.export` builds the ONNX models locally. The parity tests compare
them against the original PyTorch models when they're present, and are skipped otherwise.

## License and credits

AGPL-3.0, because the detection and recognition pipeline is ported from
[aibyou0830/immich-ml-furry](https://github.com/aibyou0830/immich-ml-furry) (AGPL-3.0), which was in turn inspired
by [FurPhotos](https://fur.photos/).

**Built with DINOv3.** The recognition model is a derivative of Meta's DINOv3 and falls under the
[DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/). That's why the weights are
downloaded with your own token and never shipped in the images. The detector is a fine-tune of
[RF-DETR](https://github.com/roboflow/rf-detr) (Apache 2.0).
