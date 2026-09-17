"""Video-level orchestration. Core logic also accepts synthetic/test observations."""
import logging
from time import perf_counter
from core.data import Observation
from core.engine import Engine
from models.contact import create_contact_detector
from utils.config import resolve_path
from utils.logging_utils import write_report
from utils.video import VideoReader, VideoWriter
from utils.visualization import annotate
from utils.preview import LivePreview

LOG = logging.getLogger(__name__)


def run_pipeline(cfg, *, demo=False, perception_only=False, max_frames=None):
    tracker = None
    preview = LivePreview(cfg["visualization"].get("live_preview", True))
    stopped_early = False
    started = perf_counter()
    if demo:
        from utils.demo import generate_video
        source = resolve_path("inputs/demo_input.mp4")
        output = resolve_path(cfg["output_video_path"])
        if source == output:
            raise ValueError("Output must not overwrite synthetic input")
        generate_video(source, cfg)
        LOG.warning("SYNTHETIC DEMO: scripted perception exercises event logic/video IO; this is not model validation")
        # Respect contact enable/failure behavior, but learned contact on a drawing is not acceptance evidence.
    else:
        source, output = resolve_path(cfg["input_video_path"]), resolve_path(cfg["output_video_path"])
        if not source.is_file():
            raise FileNotFoundError(f"Input video does not exist: {source}")
    if source == output:
        raise ValueError("Output must not overwrite input")
    contact = create_contact_detector(cfg["contact"])
    from tracking.recognition import PersonRecognition
    recognition = None if demo else PersonRecognition(cfg["recognition"])
    engine = Engine(cfg, contact, perception_only, recognition)
    try:
        with VideoReader(source, cfg["video"]["fallback_fps"]) as reader:
            roi = engine.runtime.roi
            if roi[2] > reader.width or roi[3] > reader.height:
                raise ValueError(f"ROI {roi} lies outside {reader.width}x{reader.height} video; configure original-frame pixels")
            if not demo:
                from models.detector import Detector
                from models.pose import PoseEstimator
                from tracking.tracker import Tracker
                detector = Detector(cfg["models"])
                tracker = Tracker(detector, cfg["tracking"])
                pose = PoseEstimator(cfg["models"])
            with VideoWriter(output, reader.fps, (reader.width, reader.height), cfg["video"]["codec"]) as writer:
                processing_started = last_progress = perf_counter()
                for index, frame in enumerate(reader):
                    if max_frames is not None and index >= max_frames:
                        break
                    if demo:
                        from utils.demo import observation_at
                        observation = observation_at(index, cfg)
                    else:
                        detections = tracker.update(frame)
                        people = [d for d in detections if d.label == "person"]
                        observation = Observation(people, [d for d in detections if d.label == "backpack"], pose.estimate(frame, people))
                    runtime = engine.step(frame, observation, index + 1, index / reader.fps)
                    image = annotate(frame, runtime, cfg["visualization"], demo, perception_only) if cfg["visualization"]["enabled"] else frame
                    writer.write(image)
                    elapsed = perf_counter() - processing_started
                    fps = writer.count / max(elapsed, .001)
                    total = min(reader.frame_count, max_frames) if max_frames and reader.frame_count else reader.frame_count
                    if not preview.show(image, writer.count, total, fps):
                        stopped_early = True
                        LOG.info("Preview stopped by user; saving %d processed frames", writer.count)
                        break
                    if perf_counter() - last_progress >= 5:
                        LOG.info("Progress %d/%s frames | %.1f FPS | contact inferences %d", writer.count, total or "?", fps, engine.contact_inferences)
                        last_progress = perf_counter()
                count = writer.count
        if count == 0:
            raise OSError("Video contained no decodable frames")
        seconds = perf_counter() - started
        report = write_report(output.with_suffix(".events.json"), engine.runtime, source, output, count, demo, perception_only,
                              elapsed_seconds=round(seconds, 3), processing_fps=round(fps, 2),
                              contact_inferences=engine.contact_inferences, stopped_by_user=stopped_early,
                              reid_inferences=recognition.inferences if recognition else 0)
        LOG.info("Completed in %.1fs (processing %.1f FPS; %d contact inferences)", seconds, fps, engine.contact_inferences)
        LOG.info("Wrote %d frames: %s", count, output)
        LOG.info("Observed sequence: %s", " -> ".join(report["state_sequence"]))
        if not engine.runtime.initialized and not perception_only:
            LOG.warning("No stable backpack initialized inside ROI; state stayed PLACED/uncertain. Check ROI, visibility and thresholds.")
        return report
    finally:
        preview.close()
        if tracker is not None:
            tracker.close()
