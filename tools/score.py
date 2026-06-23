"""
Score a submission against ground truth and update leaderboard.

Usage:
  python tools/score.py --desc "UNKNOWN 제거 + history reset"
  python tools/score.py --sub output/submission.csv --desc "iou=0.3 테스트"

Output:
  - 터미널에 TP/FP/FN/Precision/Recall 출력
  - output/leaderboard.csv 누적 기록
  - output/leaderboard.html 갱신 (브라우저에서 바로 열기)
"""

import csv
import argparse
from datetime import datetime
from collections import Counter
import os
import json

GT_PATH          = "data/ground_truth.csv"
LEADERBOARD_PATH = "output/leaderboard.csv"
HTML_PATH        = "output/leaderboard.html"


def load_gt(path):
    events = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            events.append((row["class_name"].strip(), row["action"].strip()))
    return events


_ACTION_MAP = {"구매": "purchase", "반환": "return"}

def load_submission(path):
    events = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name   = row.get("품목명", "").strip()
            action = row.get("구매/반환 여부", "").strip()
            action = _ACTION_MAP.get(action, action)  # normalise to EN
            if name and action:
                events.append((name, action))
    return events


def score(gt_events, sub_events):
    gt_counter  = Counter(gt_events)
    sub_counter = Counter(sub_events)
    all_keys = set(gt_counter) | set(sub_counter)

    tp = fp = fn = 0
    details = []
    for key in sorted(all_keys):
        g = gt_counter[key]
        s = sub_counter[key]
        matched = min(g, s)
        tp += matched
        fp += max(0, s - g)
        fn += max(0, g - s)
        if g != s:
            details.append((key[0], key[1], g, s))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return tp, fp, fn, precision, recall, f1, details


def load_leaderboard(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def append_leaderboard(path, row_dict):
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row_dict)


def generate_html(lb_rows, html_path):
    data_json = json.dumps(lb_rows, ensure_ascii=False)
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>REDRED 리더보드</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; padding: 32px; }}
  h1 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: 4px; color: #fff; }}
  .subtitle {{ color: #888; font-size: 0.85rem; margin-bottom: 28px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  th {{ background: #1e2130; color: #aaa; padding: 10px 14px; text-align: left;
        font-weight: 600; border-bottom: 2px solid #2a2d3e; white-space: nowrap; }}
  tr {{ border-bottom: 1px solid #1e2130; transition: background 0.15s; }}
  tr:hover {{ background: #1a1d2e; }}
  td {{ padding: 10px 14px; vertical-align: middle; }}
  .rank {{ font-weight: 700; color: #888; width: 36px; text-align: center; }}
  .rank.gold   {{ color: #f5c518; }}
  .rank.silver {{ color: #b0b0b0; }}
  .rank.bronze {{ color: #cd7f32; }}
  .desc {{ font-weight: 500; color: #ddd; max-width: 280px; }}
  .ts {{ color: #666; font-size: 0.78rem; white-space: nowrap; }}
  .bar-wrap {{ display: flex; align-items: center; gap: 8px; }}
  .bar {{ height: 8px; border-radius: 4px; min-width: 2px; }}
  .val {{ font-weight: 600; width: 44px; text-align: right; }}
  .f1  {{ color: #7ee8a2; }}
  .pre {{ color: #79c8ff; }}
  .rec {{ color: #ffb347; }}
  .tp  {{ color: #7ee8a2; }}
  .fp  {{ color: #ff7070; }}
  .fn  {{ color: #ffb347; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:12px;
            font-size:0.75rem; font-weight:600; }}
  .best {{ background:#1a3a2a; color:#7ee8a2; }}
</style>
</head>
<body>
<h1>🏆 REDRED 리더보드</h1>
<p class="subtitle">파이프라인 버전별 정확도 추적 — python tools/score.py --desc "설명" 으로 갱신</p>

<table id="lb">
<thead>
<tr>
  <th>#</th>
  <th>설명</th>
  <th>F1</th>
  <th>Precision</th>
  <th>Recall</th>
  <th>TP</th><th>FP</th><th>FN</th>
  <th>제출</th><th>GT</th>
  <th>기록 시각</th>
</tr>
</thead>
<tbody id="tbody"></tbody>
</table>

<script>
const data = {data_json};

function pct(s) {{ return parseFloat(s.replace('%','')) || 0; }}

data.sort((a,b) => pct(b.F1) - pct(a.F1));

const bestF1 = pct(data[0]?.F1 || '0');
const tbody = document.getElementById('tbody');

data.forEach((row, i) => {{
  const rank = i + 1;
  const f1v  = pct(row.F1);
  const prev = pct(row.Precision);
  const recv = pct(row.Recall);
  const isB  = f1v === bestF1;

  const rankClass = rank===1?'gold':rank===2?'silver':rank===3?'bronze':'';

  const bar = (v, cls, color) =>
    `<div class="bar-wrap">
       <div class="bar ${{cls}}" style="width:${{Math.round(v*1.4)}}px;background:${{color}}"></div>
       <span class="val ${{cls}}">${{v.toFixed(1)}}%</span>
     </div>`;

  tbody.innerHTML += `<tr>
    <td class="rank ${{rankClass}}">${{rank}}</td>
    <td class="desc">${{row.description}} ${{isB ? '<span class="badge best">BEST</span>' : ''}}</td>
    <td>${{bar(f1v,  'f1',  '#7ee8a2')}}</td>
    <td>${{bar(prev, 'pre', '#79c8ff')}}</td>
    <td>${{bar(recv, 'rec', '#ffb347')}}</td>
    <td class="tp">${{row.TP}}</td>
    <td class="fp">${{row.FP}}</td>
    <td class="fn">${{row.FN}}</td>
    <td>${{row.total_sub}}</td>
    <td>${{row.total_gt}}</td>
    <td class="ts">${{row.timestamp}}</td>
  </tr>`;
}});
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub",  default="output/submission.csv")
    parser.add_argument("--gt",   default=GT_PATH)
    parser.add_argument("--desc", default="", help="이번 버전 설명")
    parser.add_argument("--lb",   default=LEADERBOARD_PATH)
    parser.add_argument("--html", default=HTML_PATH)
    args = parser.parse_args()

    gt_events  = load_gt(args.gt)
    sub_events = load_submission(args.sub)

    tp, fp, fn, precision, recall, f1, details = score(gt_events, sub_events)

    print(f"\n{'='*55}")
    print(f"  GT 이벤트:     {len(gt_events):>4}개")
    print(f"  제출 이벤트:   {len(sub_events):>4}개")
    print(f"{'─'*55}")
    print(f"  TP:            {tp:>4}")
    print(f"  FP:            {fp:>4}")
    print(f"  FN:            {fn:>4}")
    print(f"{'─'*55}")
    print(f"  Precision:     {precision*100:>6.1f}%")
    print(f"  Recall:        {recall*100:>6.1f}%")
    print(f"  F1:            {f1*100:>6.1f}%")
    print(f"{'='*55}\n")

    if details:
        print("불일치 항목 (클래스 / 액션 / GT수 / 제출수):")
        for cls, action, g, s in sorted(details, key=lambda x: -abs(x[2]-x[3]))[:15]:
            diff = s - g
            mark = "↑" if diff > 0 else "↓"
            print(f"  {mark}{abs(diff):>2}  {cls:<50} {action}  GT={g} Sub={s}")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    row_dict = {
        "timestamp":   ts,
        "description": args.desc,
        "total_gt":    len(gt_events),
        "total_sub":   len(sub_events),
        "TP":          tp,
        "FP":          fp,
        "FN":          fn,
        "Precision":   f"{precision*100:.1f}%",
        "Recall":      f"{recall*100:.1f}%",
        "F1":          f"{f1*100:.1f}%",
    }
    append_leaderboard(args.lb, row_dict)

    lb_rows = load_leaderboard(args.lb)
    generate_html(lb_rows, args.html)

    print(f"리더보드 갱신 완료")
    print(f"  CSV  → {args.lb}")
    print(f"  HTML → {args.html}  (브라우저에서 열기)")


if __name__ == "__main__":
    main()
