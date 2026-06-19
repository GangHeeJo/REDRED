"""
Generate data/custom.yaml from names.txt.

Usage (run on server):
    python make_yaml.py \
        --names  ~/Dataset/names.txt \
        --train  ~/yolov7/data/train.txt \
        --val    ~/yolov7/data/val.txt \
        --out    ~/yolov7/data/custom.yaml
"""

import argparse
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val",   required=True)
    parser.add_argument("--out",   default="custom.yaml")
    args = parser.parse_args()

    with open(args.names) as f:
        names = [line.strip() for line in f if line.strip()]

    config = {
        "train": args.train,
        "val":   args.val,
        "test":  args.val,
        "nc":    len(names),
        "names": names,
    }

    with open(args.out, "w") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    print(f"Wrote {args.out}  (nc={len(names)})")
    print("First 5 classes:", names[:5])


if __name__ == "__main__":
    main()
