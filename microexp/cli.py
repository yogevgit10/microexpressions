from __future__ import annotations

import argparse
from pathlib import Path

from .analyze import analyze_trials, annotations_ready, create_annotations_template, load_annotations
from .contact_sheets import build_contact_sheets
from .full_extract import extract_all
from .inventory import build_inventory, load_inventory
from .reporting import build_report


DEFAULT_VIDEO_DIR = Path(r"C:\Users\gitya\Downloads\Micro Exp Papers")
DEFAULT_OUT_DIR = Path("results")


def cmd_inventory(args) -> None:
    inventory = build_inventory(args.video_dir, args.out_dir, hash_files=not args.no_hash)
    print(f"Wrote {len(inventory)} inventory rows to {args.out_dir / 'inventory.csv'}")


def cmd_contact_sheets(args) -> None:
    inventory = load_inventory(args.out_dir / "inventory.csv")
    outputs = build_contact_sheets(inventory, args.out_dir, every_s=args.every_s, representative_only=not args.include_duplicates)
    print(f"Wrote {len(outputs)} contact sheets to {args.out_dir / 'contact_sheets'}")


def cmd_init_annotations(args) -> None:
    inventory = load_inventory(args.out_dir / "inventory.csv")
    out_path = args.out_dir / "annotations.csv"
    if out_path.exists() and not args.force:
        print(f"Annotations already exist: {out_path}")
        return
    annotations = create_annotations_template(inventory, out_path)
    print(f"Wrote {len(annotations)} annotation rows to {out_path}")


def cmd_analyze(args) -> None:
    outputs = analyze_trials(
        args.out_dir,
        annotations_path=args.annotations,
        include_deepface=args.deepface,
        deepface_every_s=args.deepface_every_s,
    )
    print(outputs["message"])
    for key, value in outputs.items():
        if key != "message" and value:
            print(f"{key}: {value}")


def cmd_report(args) -> None:
    report = build_report(args.out_dir)
    print(f"Wrote report: {report}")


def cmd_extract_all(args) -> None:
    outputs = extract_all(
        video_dir=args.video_dir,
        out_dir=args.out_dir,
        target_fps=args.target_fps,
        include_deepface=args.deepface,
        deepface_every_s=args.deepface_every_s,
        include_duplicates=args.include_duplicates,
        video_ids=args.video_id,
        limit_videos=args.limit_videos,
        max_width=args.max_width,
        force=args.force,
        max_frames=args.max_frames,
    )
    print("Full extraction complete.")
    for key, value in outputs.items():
        if value:
            print(f"{key}: {value}")


def cmd_run_all(args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory(args.video_dir, args.out_dir, hash_files=not args.no_hash)
    print(f"Wrote inventory rows: {len(inventory)}")
    sheets = build_contact_sheets(inventory, args.out_dir, every_s=args.every_s, representative_only=True)
    print(f"Wrote contact sheets: {len(sheets)}")

    annotations_path = args.out_dir / "annotations.csv"
    if not annotations_path.exists():
        annotations = create_annotations_template(inventory, annotations_path)
        print(f"Wrote annotation template rows: {len(annotations)}")
    else:
        annotations = load_annotations(annotations_path)
        print(f"Using existing annotations: {annotations_path}")

    if annotations_ready(annotations):
        outputs = analyze_trials(
            args.out_dir,
            annotations_path=annotations_path,
            include_deepface=args.deepface,
            deepface_every_s=args.deepface_every_s,
        )
        print(outputs["message"])
    else:
        print("No sniff_onset_s values found yet; trial-level analysis is pending annotation.")

    report = build_report(args.out_dir)
    print(f"Wrote report: {report}")


def cmd_smoke_test(args) -> None:
    from .inventory import iter_video_files, video_metadata
    from .contact_sheets import make_contact_sheet
    from .facial_metrics import FaceMetricExtractor, METRIC_COLUMNS
    from .inventory import require_cv2

    candidates = [p for p in iter_video_files(args.video_dir) if "2026-01-08 10-25-28" in p.name]
    if not candidates:
        candidates = list(iter_video_files(args.video_dir))[:1]
    if not candidates:
        raise SystemExit(f"No videos found in {args.video_dir}")
    video = candidates[0]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = video_metadata(video)
    sheet = make_contact_sheet(video, args.out_dir / "smoke_contact_sheet.jpg", every_s=2.0)
    cv2 = require_cv2()
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, 2000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read smoke-test frame from {video}")
    extractor = FaceMetricExtractor()
    try:
        metrics = extractor.process_bgr(cv2, frame, frame_idx=200, timestamp_s=2.0)
    finally:
        extractor.close()
    print(f"Smoke video: {video.name}")
    print(f"Metadata: {metadata}")
    print(f"Contact sheet: {sheet}")
    print(f"MediaPipe face_detected: {metrics['face_detected']}")
    preview = {key: metrics[key] for key in METRIC_COLUMNS[:5]}
    print(f"Metric preview: {preview}")
    if args.deepface:
        from .deepface_emotions import DeepFaceEmotionExtractor

        deepface = DeepFaceEmotionExtractor()
        emotion_preview = deepface.analyze_bgr(frame)
        print(f"DeepFace ok: {emotion_preview['deepface_ok']}")
        print(f"DeepFace dominant emotion: {emotion_preview['deepface_dominant_emotion']}")
        if not emotion_preview["deepface_ok"]:
            error = str(emotion_preview["deepface_error"]).encode("ascii", errors="backslashreplace").decode("ascii")
            print(f"DeepFace error: {error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Micro-expression video analysis pipeline")
    sub = parser.add_subparsers(required=True)

    def add_common(p):
        p.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
        p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)

    p = sub.add_parser("inventory")
    add_common(p)
    p.add_argument("--no-hash", action="store_true")
    p.set_defaults(func=cmd_inventory)

    p = sub.add_parser("contact-sheets")
    add_common(p)
    p.add_argument("--every-s", type=float, default=5.0)
    p.add_argument("--include-duplicates", action="store_true")
    p.set_defaults(func=cmd_contact_sheets)

    p = sub.add_parser("init-annotations")
    add_common(p)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init_annotations)

    p = sub.add_parser("analyze")
    add_common(p)
    p.add_argument("--annotations", type=Path, default=None)
    p.add_argument("--deepface", action="store_true", help="Also sample DeepFace emotion predictions per analysis window.")
    p.add_argument("--deepface-every-s", type=float, default=0.25, help="Seconds between DeepFace samples when --deepface is enabled.")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("report")
    add_common(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("extract-all")
    add_common(p)
    p.add_argument("--target-fps", type=float, default=100.0, help="Target MediaPipe processing FPS. 100 means every frame for the current videos.")
    p.add_argument("--deepface", action="store_true", help="Also sample DeepFace emotions at a slower interval.")
    p.add_argument("--deepface-every-s", type=float, default=1.0, help="Seconds between DeepFace samples.")
    p.add_argument("--include-duplicates", action="store_true", help="Process exact duplicate video files too.")
    p.add_argument("--video-id", action="append", default=None, help="Restrict extraction to one video_id. Can be repeated.")
    p.add_argument("--limit-videos", type=int, default=None, help="Process only the first N selected videos.")
    p.add_argument("--max-width", type=int, default=960, help="Resize frames to this width before landmark/emotion models; use 0 for original size.")
    p.add_argument("--force", action="store_true", help="Recompute per-video feature files even if they already exist.")
    p.add_argument("--max-frames", type=int, default=None, help="Debug/smoke-test limit on processed frames per video.")
    p.set_defaults(func=cmd_extract_all)

    p = sub.add_parser("run-all")
    add_common(p)
    p.add_argument("--every-s", type=float, default=5.0)
    p.add_argument("--no-hash", action="store_true")
    p.add_argument("--deepface", action="store_true", help="Run DeepFace emotion sampling if annotations are ready.")
    p.add_argument("--deepface-every-s", type=float, default=0.25)
    p.set_defaults(func=cmd_run_all)

    p = sub.add_parser("smoke-test")
    add_common(p)
    p.add_argument("--deepface", action="store_true", help="Include a one-frame DeepFace emotion smoke test.")
    p.set_defaults(func=cmd_smoke_test)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
