import argparse
from rvm.api import detect_marker_poses


def main():
    parser = argparse.ArgumentParser(description="Estimate marker poses (rvec, tvec) using solvePnP")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--calib", default=None, help="Path to camera calibration file")
    parser.add_argument("--marker_size", type=float, default=0.05, help="Marker size in meters")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()

    poses = detect_marker_poses(
        image_path=args.image,
        camera_calib=args.calib,
        marker_size=args.marker_size,
        out=args.out
    )

    print(f"Saved results to {args.out}. Found {len(poses)} markers.")


if __name__ == "__main__":
    main()
