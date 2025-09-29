# rvm/cli/detect.py
import argparse
from rvm.api import detect

def main():
    parser = argparse.ArgumentParser(description="Run object detection")
    parser.add_argument("--source", required=True, help="Path to image, video or webcam index", default="0")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model weights (e.g., yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt")
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument("--realtime", action="store_true", help="Enable real-time display for webcam")
    args = parser.parse_args()

    # Auto-enable realtime if source is webcam
    if args.source.isdigit():
        realtime = True
    else:
        realtime = args.realtime

    detect(args.source, args.model, args.out, realtime)

if __name__ == "__main__":
    main()
