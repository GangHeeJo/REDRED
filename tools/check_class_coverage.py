"""데이터셋에 60개 클래스가 전부 있는지, 어느 클래스가 valid에 0개인지 확인"""
import json
from collections import Counter
from pathlib import Path

dataset_dir = Path("data/coco_rfdetr")

for split in ["train", "valid"]:
    path = dataset_dir / split / "_annotations.coco.json"
    if not path.exists():
        print(f"{split}: {path} 없음")
        continue
    with open(path) as f:
        d = json.load(f)
    names = {cat["id"]: cat["name"] for cat in d["categories"]}
    counts = Counter(a["category_id"] for a in d["annotations"])
    zero = [names[cid] for cid in names if counts.get(cid, 0) == 0]
    print(f"=== {split} === categories={len(names)}, images={len(d['images'])}, annotations={len(d['annotations'])}")
    if zero:
        print(f"인스턴스 0개인 클래스 ({len(zero)}개): {zero}")
    else:
        print("모든 클래스에 인스턴스 있음")
