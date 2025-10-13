import argparse
from rvm.api import detect_marker_poses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo: ArUco marker pose estimation")
    parser.add_argument("--image", required=True)
    parser.add_argument("--calib", default=None)
    parser.add_argument("--marker_size", type=float, default=0.05)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    poses = detect_marker_poses(args.image, camera_calib=args.calib, marker_size=args.marker_size, out_dir=args.out)
    print("Detected poses:", poses)