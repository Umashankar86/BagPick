import json
import logging
from pathlib import Path


def configure_logging(debug=False, log_path=None):
    handlers = [logging.StreamHandler()]
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8", mode="w"))
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=handlers, force=True)


def write_report(path, runtime, source, output, frame_count, demo, perception_only, **metrics):
    data = dict(input_video=str(source), output_video=str(output), synthetic_demo=demo,
                perception_only=perception_only, frames_processed=frame_count,
                initialized=runtime.initialized, final_state=runtime.state.value,
                contact_status=runtime.contact_status,
                actor_id_semantics="Session-local learned appearance ID, null if unknown; not verified identity. track_id is raw BoT-SORT ID",
                recognition_status=runtime.recognition_status,
                actor_trace=runtime.actor_trace,
                state_sequence=["PLACED"] + [event["state"] for event in runtime.transitions],
                transitions=runtime.transitions)
    data.update(metrics)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data
