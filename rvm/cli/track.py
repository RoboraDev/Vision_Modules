# rvm/cli/track.py
import argparse
from rvm import api


def main():
    parser = argparse.ArgumentParser(description="RVM Tracking CLI")

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Path to video file or webcam index (e.g., '0' for default webcam)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="YOLO model weights (default: yolo11n.pt)."
    )
    parser.add_argument(
        "--tracker",
        type=str,
        choices=["iou", "ultralytics"],
        default="iou",
        help="Tracking backend: 'iou' (lightweight CPU) or 'ultralytics' (stronger)."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Directory to save outputs (default: results)."
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        default=True,
        help="Show real-time visualization during tracking."
    )

    args = parser.parse_args()

    print(f"[RVM] Running tracking on {args.source}...")
    results = api.track(
        source=args.source,
        model=args.model,
        tracker_type=args.tracker,
        out_dir=args.out_dir,
        realtime=args.realtime
    )

    print(f"[RVM] Tracking finished. {len(results)} objects processed.")
    print(f"[RVM] Results saved to {args.out_dir}/track_result.json")


if __name__ == "__main__":
    main()
