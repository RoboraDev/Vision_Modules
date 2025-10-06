# demos/demo_tracking.py
import argparse
from rvm.api import track


def main():
    parser = argparse.ArgumentParser(description="Vision Modules - Object Tracking Demo")
    parser.add_argument("--source", type=str, default="0", help="Path to video file or webcam index")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model weights path")
    parser.add_argument("--tracker", type=str, default="ultralytics", choices=["ultralytics", "iou"], help="Tracking backend")
    parser.add_argument("--out", type=str, default="results", help="Output directory to save results")
    parser.add_argument("--realtime", action="store_true", default=True, help="Show real-time tracking window (for webcam)")

    args = parser.parse_args()

    results = track(
        source=args.source,
        model=args.model,
        tracker_type=args.tracker,
        out_dir=args.out,
        realtime=args.realtime
    )

    print(f"Tracking completed! Saved results to: {args.out}")
    print(f"Number of tracked objects: {len(results)}")


if __name__ == "__main__":
    main()
