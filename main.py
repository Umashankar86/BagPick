"""Command-line entry point for the backpack event detector."""
import argparse
import logging
from utils.config import load_config, resolve_path, validate
from utils.logging_utils import configure_logging


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML config; defaults to config/config.yaml")
    parser.add_argument("--input", help="Recorded MP4 path (missing explicit paths are errors)")
    parser.add_argument("--output", help="Annotated MP4 path")
    parser.add_argument("--demo", action="store_true", help="Generate labeled synthetic input and run scripted observations through real event logic")
    parser.add_argument("--perception-only", action="store_true", help="Phases 1-4: detection, tracking, wrists and ROI; no event transitions")
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=None, help="Show frames while processing (default from config); --no-preview for background runs")
    roi_mode = parser.add_mutually_exclusive_group()
    roi_mode.add_argument("--setup-roi", action="store_true", help="Draw the fixed placement ROI around the initial bag and save config")
    roi_mode.add_argument("--auto-roi", action="store_true", help="Save ROI from the detected initial backpack plus padding")
    parser.add_argument("--roi-frame", type=int, default=0, help="Zero-based reference frame for ROI setup (before pickup)")
    parser.add_argument("--roi-padding", type=float, default=.15, help="Automatic ROI margin as a fraction of bag width/height per side")
    parser.add_argument("--max-frames", type=int, help="Limit frames for a model smoke test")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.input:
            cfg["input_video_path"] = args.input
        if args.output:
            cfg["output_video_path"] = args.output
        if args.preview is not None:
            cfg["visualization"]["live_preview"] = args.preview
        validate(cfg)
        if args.max_frames is not None and args.max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        if args.setup_roi or args.auto_roi:
            from utils.roi_setup import setup_roi
            setup_roi(cfg, args.config, automatic=args.auto_roi,
                      frame_index=args.roi_frame, padding=args.roi_padding)
            return 0
        if args.check_config:
            print("Configuration valid. No models loaded.")
            return 0
        configure_logging(cfg["debug"], resolve_path("outputs/run.log"))
        demo = args.demo
        if not demo and not resolve_path(cfg["input_video_path"]).is_file():
            if args.input or cfg["missing_video_behavior"] == "error":
                raise FileNotFoundError(f"Missing input video: {resolve_path(cfg['input_video_path'])}")
            logging.warning("No final video available. Running explicitly labeled synthetic demonstration.")
            demo = True
        if demo and not args.output:
            cfg["output_video_path"] = "outputs/demo_processed.mp4"
        from pipeline import run_pipeline
        run_pipeline(cfg, demo=demo, perception_only=args.perception_only, max_frames=args.max_frames)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
