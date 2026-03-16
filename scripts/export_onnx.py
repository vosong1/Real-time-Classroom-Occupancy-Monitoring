import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--width", type=int, default=416)
    parser.add_argument("--height", type=int, default=416)
    args = parser.parse_args()

    cmd = [
        "python",
        "tools/export_model.py",
        "--cfg", args.cfg,
        "--weights", args.weights,
        "--output", args.output,
        "--batch-size", str(args.batch_size),
        "--width", str(args.width),
        "--height", str(args.height),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()