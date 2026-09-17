# Backpack Pick/Place Event Detection

A local Python video pipeline with **YOLO11 Detect**, **BoT-SORT**, **YOLO11 Pose**,
a fixed placement ROI, and an explicit temporal state machine:

`PLACED -> PICKING -> PICKED -> PERSON_LEAVES -> PLACING -> PLACED`

The optional hand-object contact component implements the pretrained **CVPR 2020
Hand Object Detector**, *Understanding Human Hands in Contact at Internet Scale*
(Shan et al.). It supplies supporting evidence; geometry, tracking and the state
machine remain required.

## Run in VS Code / Windows

Select `.venv\Scripts\python.exe` as the Python interpreter. This workspace already
contains a tested environment and downloaded YOLO11/contact weights. From the
project directory:

```powershell
.venv\Scripts\python.exe main.py
```

Until `inputs/sample_video.mp4` exists, this runs an explicitly labeled synthetic
demonstration and creates:

- `inputs/demo_input.mp4`: generated input animation.
- `outputs/demo_processed.mp4`: annotated animation.
- `outputs/demo_processed.events.json`: detected transitions and evidence.
- `outputs/run.log`: readable transition log.

The demo uses scripted **observations**, then passes them through the same runtime,
geometry, histories, actor association, state machine, drawing, and MP4 writer.
It does not feed canned state labels to the machine. It is an engineering test,
**not a real recorded video or proof of model accuracy**.

For a clean installation (Python 3.10+; tested on Python 3.12, Windows, CPU):

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m scripts.setup_reid
.venv\Scripts\python.exe main.py --check-config
.venv\Scripts\python.exe main.py --demo
```

On Linux/macOS use `.venv/bin/python`. YOLO weights download automatically on the
first real-model run. The prepared local environment reuses the computer's
existing CPU PyTorch through `--system-site-packages`; a fresh install need not.

## Live processing and speed

The annotated video now opens in a live preview by default. It displays processed
frames as they become ready, along with processing FPS, progress and estimated
remaining time. This is processing-speed playback, not a promise of real-time CPU
inference. Press **Q/Esc** or close the preview to stop and save a valid partial MP4.
The JSON report records `stopped_by_user`, elapsed seconds and contact-inference count.

```powershell
.venv\Scripts\python.exe main.py --input inputs/sample_video.mp4
# Background processing without opening a window:
.venv\Scripts\python.exe main.py --input inputs/sample_video.mp4 --no-preview
```

`visualization.live_preview` controls the default. The full output is still written
at the original video FPS; preview processing speed does not change output duration.

The ResNet101 contact model is expensive on CPU. When enabled, it is checked only
with a fresh wrist close to a visible backpack during pickup or placement. One
confident learned contact associated with that actor/hand and backpack is remembered
for the interaction, and further contact inference stops. Negative, missing or
uncertain results allow at most **three attempts total**, separated by at least
**three video frames**, and only while the wrist is still close. Configure these
using `contact.max_attempts_per_interaction` and `contact.retry_interval_frames`.
Moving away pauses checks; it does not reset the retry budget. Switching actor/hand
does not transfer a confirmation, and returning to the same actor/hand does not
grant unlimited retries. Pickup and returning placement have separate budgets.

YOLO11, pose, tracking and the state machine still process **every** frame. The
overlay displays `CONFIRMED (frame N)` for remembered contact, not a claim of current
physical touch. JSON transitions distinguish `contact_detected` on the current
frame from `contact_confirmed` and its original `contact_confirmation_frame`.
Confirmation is cleared when pickup is complete and when placement is complete;
the returning placement gets its own check. No repeated contact is needed to prove
bag removal. Exhausting the attempts continues the geometry/state pipeline normally.

This trades repeated contact observations for speed; a brief touch missed by all
attempts remains unconfirmed. Set `enable_contact_detector: true` to use the model,
or `false` to omit it entirely. Real-time throughput is hardware dependent; this
machine's PyTorch installation is CPU-only.

## Process the final recording

First calibrate the ROI from the bag in your video's initial frame:

```powershell
.venv\Scripts\python.exe main.py --input inputs/sample_video.mp4 --setup-roi
```

Draw a rectangle around the **initially placed backpack with a small margin**,
then press Enter. Cancel with C without changing configuration. The preview is
scaled to fit the display; saved coordinates are mapped to original-video pixels.
Use `--roi-frame 30` to choose another zero-based reference frame before pickup.

Alternatively, detect the initial bag and add a 15% margin automatically:

```powershell
.venv\Scripts\python.exe main.py --input inputs/sample_video.mp4 --auto-roi
```

Use `--roi-padding 0.10` to adjust the margin. Multiple distinct bags or no bag
detection require manual selection. Both methods save only the four ROI coordinates
in the selected configuration, retain other settings/comments, save the previous
file as `config.yaml.roi-backup`, and write `outputs/roi_preview.jpg` for inspection.
The ROI stays **fixed at the initial placement location** throughout processing;
it does not follow the backpack. Setup exits after saving. Run processing again
to regenerate the video; existing output videos do not change on their own.
Backpacks outside the ROI are now drawn too, but cannot initialize an event there.

Set `input_video_path`, `output_video_path`, ROI and thresholds in
`config/config.yaml`, then run:

```powershell
.venv\Scripts\python.exe main.py --input inputs/sample_video.mp4 --output outputs/processed_video.mp4
```

Paths in YAML and CLI are relative to the project root unless absolute. An
explicitly supplied missing input is an error, never silently replaced by a demo.
`missing_video_behavior: error` also disables automatic demo fallback. Video
output preserves frame size and FPS; audio is not copied. Input/output collisions,
bad ROI dimensions and failed writers are checked.

Phases 1-4 can be inspected independently:

```powershell
.venv\Scripts\python.exe main.py --input inputs/sample_video.mp4 --perception-only
```

This shows person/backpack boxes, BoT-SORT IDs, wrists and ROI while keeping state
transitions disabled. `--max-frames 30` limits a diagnostic run. `--demo --output
outputs/my_demo.mp4` selects a custom demo output.

## Modules and frame flow

```text
main.py                    CLI, missing-video policy, config validation
pipeline.py                video/model orchestration
config/config.yaml         paths, ROI, models, thresholds, visualization
models/detector.py         YOLO11 person/backpack inference and decoding
tracking/tracker.py        Ultralytics BoT-SORT, no ReID, no camera motion model
models/reid.py             pretrained body-appearance encoder (OpenVINO)
tracking/recognition.py    conservative appearance gallery and identity matching
models/pose.py             full-frame wrists, one-to-one pose/person association
models/contact.py          generic contact API, actual 100DOH adapter, safe fallback
tracking/memory.py         tracked person visibility and bounded wrist histories
core/data.py               central Runtime and all observations
core/backpack.py           backpack selection, box/center history, ROI hysteresis
core/geometry.py           reusable geometric operations
core/temporal.py           fresh-observation counters and motion vectors
core/interaction.py        actor/hand selection, short spatial ID rebinding
core/state_machine.py      five ordered transitions
core/engine.py             per-frame updates and event logic
utils/video.py             checked MP4 reader/writer
utils/visualization.py     annotations, stale/uncertain evidence
utils/logging_utils.py     terminal/file logs and JSON transition report
utils/demo.py              deterministic synthetic input and observations
scripts/setup_contact.py   pinned contact source and checkpoint setup
scripts/verify_models.py   actual model inference smoke tests
tests/                     regression coverage, no weights needed
```

Each real frame runs YOLO11 detection plus BoT-SORT in **one** Ultralytics `.track`
call, retains only COCO person (0) and backpack (24), estimates full-frame pose,
matches pose boxes to tracked people, queries optional contact, updates geometry
and memory, evaluates temporal rules, draws, and writes the frame. Detector and
pose outputs use original-frame coordinates. There is no custom training, depth,
optical-flow event logic, face recognition, GCN, or database.

The central runtime holds current/previous event state, initialization status,
frame/time, ROI, backpack box history and trajectory, each tracked person's wrists,
per-hand contact, visibility, selected actor/hand, movement and counters.

## ROI and motion

`roi.x1/y1/x2/y2` define the initial placement rectangle in original pixels. The
program waits in `PLACED`/uncertain until a visible backpack is stable within that
ROI for `initialization_required_frames`. It never treats an empty ROI as a known
initial placement.

Overlap is **intersection area / backpack area**, not IoU. The bag is inside when
its center is in the ROI and overlap reaches `overlap_threshold`. It is confirmed
outside only when its center is outside and overlap is at or below
`exit_overlap_threshold`. Between those thresholds, the previous inside status is
retained, and the ambiguous band is not counted as confirmed removal. Both center
membership and overlap are stored.

Motion uses bounded center/wrist histories and average pixel displacement per
frame over `motion_window_frames`. Missing frames break the motion segment.
Pickup requires real bag and wrist displacement from their interaction-start
positions; disappearance alone never establishes displacement.

## State transition rules

| Transition | Required primary evidence |
| --- | --- |
| `PLACED -> PICKING` | Initialized bag inside ROI; visible tracked actor; observed wrist inside ROI and near bag for several frames. |
| `PICKING -> PICKED` | Same actor/hand; observed bag and wrist outside ROI; both displaced from pickup anchors; wrist still near bag; temporal confirmation. |
| `PICKED -> PERSON_LEAVES` | Locked interacting person's track absent for the configured consecutive missing-person threshold. Wrist loss alone cannot cause departure. |
| `PERSON_LEAVES -> PLACING` | Visible returning person, fresh wrist near bag and inside ROI, bag visibly entering/overlapping ROI; repeated observations. A new track ID is accepted. |
| `PLACING -> PLACED` | Fresh bag observations stable inside ROI, actual interacting wrist observed outside, no other observed/temporarily retained hand still engaged; repeated frames. |

All transition thresholds are at least two frames. One confident learned contact
for the same actor and hand is retained as supporting evidence. It can reduce the `PICKING` or `PLACING`
confirmation threshold by `supporting_confirmation_reduction_frames`, never below
two actual fresh frames. It cannot bypass any primary condition or influence
removal/departure/final-placement transitions. An absent contact result is unknown,
not evidence of release. No contact detector is required to finish the sequence.

## Missing observations and multiple people

The previous forced **Actor 1** return assumption has been removed. Body-appearance
recognition now runs independently of bag contact and the event state machine.
The pretrained [Open Model Zoo person-reidentification-retail-0288 model](https://docs.openvino.ai/2023.3/omz_models_model_person_reidentification_retail_0288.html)
encodes person crops into 256-dimensional descriptors compared by cosine similarity.
No face recognition, training, person names or persistent identity database is used.
BoT-SORT itself still has its internal ReID disabled; this separate module owns the
session-local appearance identities so they can survive tracker expiration.

`Person 1 (appearance) [track 11]` means a learned appearance association, not simply
a renamed tracker ID. Two consistent sampled observations are required to enroll
or match a new track. Similarity at or above 0.70 plus a margin over other
candidates permits a return match. Consistently low similarity (<=0.55) enrolls a
different appearance ID. Intermediate/ambiguous results remain `Person unknown`.
Simultaneous people cannot share one identity. Short continuous tracks tolerate
pose changes, retain diverse views at similarity >=0.65 and revoke identity after
two contradictory samples. A track returning after a >12-frame gap must revalidate.
Poor/small or overlapping crops do not advance recognition confirmation.

Settings are under `recognition` in `config/config.yaml`; sampling every five frames
limits CPU cost. The crop encoder is isolated in `models/reid.py` with the generic
`encode(BGR_crop) -> descriptor` interface. Install weights on another machine with
`python -m scripts.setup_reid`; the downloader verifies official SHA-384 checksums.
Missing dependencies/weights or inference failure log a warning and leave identity
unknown while the event pipeline continues. Disable with `recognition.enabled: false`.
The synthetic demo deliberately does not claim learned recognition.

JSON `actor_id` is the appearance ID (null when unknown), while `track_id` remains
the raw BoT-SORT ID. `actor_trace` records appearance links and cosine evidence;
`pickup_person_id` and `same_as_pickup_person` distinguish the original picker from
the returner. The latter is true/false/null based on appearance IDs, **not a verified
physical-identity claim**. A different person can return/place the bag and complete
the event without inheriting the original person's identity. Wrist histories,
contact, geometry and state transitions still use their own fresh observations.

Appearance recognition can fail with similar clothing, pose changes, occlusion,
lighting or a carried bag affecting the crop. Thresholds are conservative defaults,
not calibrated probabilities. Test with a second real person before relying on
same/different-person judgments. This does not recognize an individual backpack:
backpack continuity still assumes one bag and uses detection/tracking/spatial history.
The `VISIBLE` overlay reports recognized/unknown people immediately, independently
of `PLACING` or bag contact. Person labels have opaque backgrounds below the status
panel. The 0.70 matching threshold accepts weaker matches than the previous 0.80;
it can recognize returns earlier but increases false-match risk.

- Wrist and backpack history survive configured short gaps. Stale positions are
  marked on screen; they **pause**, rather than advance, a pending confirmation.
  After the grace period the pending counter resets; event state is preserved.
- A contradictory fresh observation resets its relevant counter immediately.
- The actor and hand lock at pickup and placement. Another person's wrist cannot
  silently take over; changing a tentative actor also resets confirmation.
- During a short in-view tracking gap, exactly one spatially overlapping new ID
  with a fresh wrist near the bag may resume the event. This is local continuity,
  not permanent identity or learned ReID. An ambiguous match is left uncertain.
- Departure tracks actor visibility independently of pose/backpack visibility.
  After departure, any geometrically plausible returning actor may continue.
- Pose/person matching and contact-hand/wrist matching are one-to-one. Untracked
  people can be drawn but cannot trigger an event using a fabricated ID.
- Backpack selection starts in the ROI and then favors its track/spatial history.
  This is intentionally a one-backpack event system, not a multi-object ownership
  system. The v1 recording should contain one main actor and one backpack.

## Actual pretrained contact detector

The isolated adapter in `models/contact.py` uses the original ResNet101/Faster-RCNN
100DOH architecture and pretrained `faster_rcnn_1_8_132028.pth` checkpoint. The
architecture comes from the [PyTorch port](https://github.com/larsrpe/hand_object_detector_pytorch)
at commit `ff6303e6e99acd710fcc36e9fba0da606cea3551`, which uses torchvision ROI/NMS
operations instead of compiling legacy CUDA extensions. The network is not trained
or replaced by a wrist-distance heuristic. Its local inference adapter also fixes
the port's convenience-wrapper issue that returns the hand array twice.

To install on another machine:

```powershell
.venv\Scripts\python.exe -m scripts.setup_contact
```

Then change only:

```yaml
contact:
  enable_contact_detector: true
  provider: cvpr2020
```

The main config already contains the source path, weights path, device and other
options. On this machine, source and weights are already downloaded. CPU inference
works but is substantially slower than YOLO; set `contact.device: cuda:0` only on a
compatible GPU installation.

The authors' original Google Drive links were unavailable during setup. The script
uses an [archived checkpoint mirror](https://huggingface.co/highbyte/frankmocap/tree/1f9f0d3c77e74ebf1b86ad7362cebda09d82ffb3)
and verifies SHA-256:

```text
3444e3b992417cb6adfcd71539ca8c92602c21a3e2fdfff4628dbdbdf8a2dd03
```

That hash verifies the pinned mirror's content, not an independently published
author checksum. Loading uses PyTorch `weights_only=True`. Source/checkpoint
provenance and upstream licensing are recorded in `THIRD_PARTY.md`.

The model predicts hand boxes, contact state, a hand-object offset and object
boxes. For backpacks, only the learned **portable object** state is positive.
Confidence is hand-detection confidence multiplied by portable-contact probability;
it is an evidence score, not a calibrated probability of touching this backpack.
Objects are matched with the published hand-offset rule, then associated to the
YOLO backpack using box IoU or close spatial proximity. When no object region is
available, a positive learned contact can be associated by wrist proximity. If a
different object region is available, wrist proximity cannot override its mismatch.

Disabled mode never imports the optional network. Import errors, missing weights,
dependency failures, inference exceptions and malformed provider outputs log a
warning and continue without contact evidence. A failed provider is disabled for
the rest of the run so it does not spam warnings or repeatedly crash.

### Swapping the detector

Implement `predict(frame) -> list[RawContact]` in your own provider. Each
`RawContact` contains `hand_region`, `detected`, `confidence`, optional
`object_region`, and optional contact-state text in original-frame coordinates.
Configure `provider: custom`, `custom_factory: package.module:Factory`, and optional
constructor arguments under `contact.options`. No inheritance is required.

The service associates each detected hand with one tracked wrist and returns
`ContactResult(person_id, hand, detected, confidence, object_region,
backpack_associated, association)`. The rest of the pipeline only consumes this
generic result. Neither a missing hand prediction nor a disabled detector creates
a false observed-release signal.

## Verification and deliverables

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m scripts.verify_models
```

Unit tests cover the full ordered cycle without contact, short/long losses,
one-frame intrusions, fresh/stale counters, ID changes, unrelated people, ROI
boundaries, contact failure/association, pose association, config and MP4 roundtrip.
`scripts.verify_models` separately runs actual YOLO11, BoT-SORT, pose and pretrained
CVPR 2020 inference and saves `outputs/model_verification.json`.

Generated artifacts currently include:

| Artifact | What it verifies |
| --- | --- |
| `inputs/demo_input.mp4` / `outputs/demo_processed.mp4` | Full event sequence on explicit synthetic observations; annotations and MP4 IO. |
| `outputs/demo_processed.events.json` | Exact five transitions with reasons, frame/time, actor and geometric/contact evidence. |
| `inputs/model_smoke_input.mp4` / `outputs/model_smoke_processed.mp4` | Actual YOLO11/BoT-SORT/Pose integration on a repeated public still image, not an event recording. |
| `outputs/contact_model_smoke.jpg` | Actual checkpoint inference on the upstream annotated reference image. |
| `outputs/model_verification.json` | Model loading, observed people/wrists, stable IDs, contact inference results. |

## Final recording and limitations

The supplied `inputs/sample_video.mp4` has completed all five intended transitions.
Synthetic/demo files remain explicitly labeled development substitutes. Recognition
has deterministic same/different/uncertain feature tests; a separate recording with
a different returning person is still needed to validate real-world separation.

Use a fixed camera, clear initial bag fully inside ROI, visible wrists, good light,
and one primary actor. Record pickup, actual exit, return with backpack, placement,
and wrist withdrawal. Check the output and JSON sequence for all five transitions.
Frames, rather than seconds, drive confirmation, so adjust thresholds for FPS.

General COCO backpack detection can miss small/occluded bags; wrist/contact models
can be uncertain. The system preserves the last valid event state when required
evidence is absent. An aborted pickup or an actor never leaving stays at the last
valid state instead of inventing a shortcut through the required sequence. Multiple
bags, crowded scenes, long ID swaps, camera movement and hands hidden throughout a
transition are outside the controlled v1 assumptions. No live camera, permanent
identity, model training or audio processing is included.
