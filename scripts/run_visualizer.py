#!/usr/bin/env python3
"""Launch the interactive embedding visualizer.

Usage:
    python scripts/run_visualizer.py [--port PORT]

Requires:
    - Prepared data in outputs/visualizer/mayon_DAPI_GLYCORNA.parquet
    - Run prepare_visualizer_data.py first if data doesn't exist

The visualizer will be available at http://localhost:8050
"""

import argparse
from pathlib import Path

from scpepato.viz import launch_visualizer

# Configuration for mayon DAPI_GLYCORNA
DATA_PATH = Path("outputs/visualizer/mayon_DAPI_GLYCORNA.parquet")
IMAGES_DIR = Path(
    "/archive/cheeseman/ops_analysis/mayon-analysis/analysis/brieflow_output/phenotype/images"
)
CHANNELS = ["DAPI", "GLYCORNA"]
# Blue for DAPI, Green for GLYCORNA
CHANNEL_COLORS = [
    (0.0, 0.0, 1.0),  # Blue
    (0.0, 1.0, 0.0),  # Green
]


def main():
    parser = argparse.ArgumentParser(description="Launch interactive embedding visualizer")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help=f"Path to embedding parquet file (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--port", type=int, default=8050, help="Port to run server on (default: 8050)"
    )
    parser.add_argument(
        "--debug", action="store_true", default=True, help="Enable debug mode (default: True)"
    )
    parser.add_argument("--no-debug", action="store_false", dest="debug", help="Disable debug mode")
    args = parser.parse_args()

    data_path = args.data

    # Check if data exists
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Run prepare_visualizer_data.py first:")
        print("  python scripts/prepare_visualizer_data.py")
        return 1

    # Check if images directory exists
    if not IMAGES_DIR.exists():
        print(f"Warning: Images directory not found at {IMAGES_DIR}")
        print("Cell images will not display correctly")

    # Launch visualizer
    launch_visualizer(
        data_path=data_path,
        images_dir=IMAGES_DIR,
        channels=CHANNELS,
        channel_colors=CHANNEL_COLORS,
        port=args.port,
        debug=args.debug,
    )

    return 0


if __name__ == "__main__":
    exit(main())
