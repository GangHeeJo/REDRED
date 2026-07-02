"""
SAM2 도메인 데이터(instances_video_domain.json)의 클래스별 분포 확인.
과다발화 클래스가 이 데이터에서 유독 많이/적게 뽑혔는지 진단.

Usage:
    python tools/analyze_sam2_domain_classes.py \
        --sam2_json data/coco_rfdetr/annotations/instances_video_domain.json \
        --names data/names.txt
"""
import argparse
import json
from collections import Counter

OVERFIRE_CLASSES = {
    "pepperidge_farm_milano_cookies_double_chocolate",
    "nature_valley_crunchy_oats_n_honey",
    "nabisco_nilla_wafers",
    "pepperidge_farm_milk_chocolate_macadamia_cookies",
    "chewy_dips_peanut_butter",
    "crayola_24_crayons",
    "dove_white",
    "a1_steak_sauce",
    "aunt_jemima_original_syrup",
}
MISSED_10EP_CLASSES = {
    "bulls_eye_bbq_sauce_original",
    "campbells_chicken_noodle_soup",
    "frappuccino_coffee",
    "hunts_sauce",
    "ritz_crackers",
    "pepperidge_farm_milk_chocolate_macadamia_cookies",  # macadamia
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sam2_json", default="data/coco_rfdetr/annotations/instances_video_domain.json")
    p.add_argument("--names", default="data/names.txt")
    return p.parse_args()

def main():
    args = parse_args()
    with open(args.names, encoding="utf-8") as f:
        names = [l.strip() for l in f if l.strip()]

    with open(args.sam2_json, encoding="utf-8") as f:
        data = json.load(f)

    cat_count = Counter(ann["category_id"] for ann in data["annotations"])
    conf_by_cat = {}
    for ann in data["annotations"]:
        conf_by_cat.setdefault(ann["category_id"], []).append(ann.get("score", 0))

    print(f"총 {len(data['images'])}장, {len(data['annotations'])}개 어노테이션, {len(cat_count)}개 클래스 등장\n")
    print(f"{'클래스':<55}{'개수':>6}{'평균conf':>10}  플래그")
    print("-" * 90)
    for cid, cnt in cat_count.most_common():
        name = names[cid] if cid < len(names) else f"id={cid}"
        avg_conf = sum(conf_by_cat[cid]) / len(conf_by_cat[cid])
        flag = ""
        if name in OVERFIRE_CLASSES:
            flag += " <-- 과다발화 클래스(50ep)"
        if name in MISSED_10EP_CLASSES:
            flag += " <-- 10ep 미탐지 클래스"
        print(f"{name:<55}{cnt:>6}{avg_conf:>10.3f}{flag}")

    print("\n=== SAM2 도메인 데이터에 전혀 없는 클래스 (0장) ===")
    missing = [names[i] for i in range(len(names)) if i not in cat_count]
    for m in missing:
        flag = " <-- 10ep 미탐지 클래스" if m in MISSED_10EP_CLASSES else ""
        print(f"  {m}{flag}")

if __name__ == "__main__":
    main()
