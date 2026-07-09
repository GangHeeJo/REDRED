"""
Score a submission against ground truth and update leaderboard.

채점 기준 (총 100점):
  정확도 40점  — F1 기반 추정 (F1 * 40)
  RTF    20점  — RTF ≤ 1 → 만점(20점), RTF > 1 → 상대평가(미공개, 추정 불가)
  발표   40점  — 직접 측정 불가

Usage:
  python tools/score.py --desc "UNKNOWN 제거 + history reset" --rtf 0.742
  python tools/score.py --sub output/submission.csv --desc "iou=0.3 테스트" --rtf 0.80
  python tools/score.py --desc "campbells confirm=74" --rtf 0.79 \
      --motivation "campbells가 monster_energy보다 늦게 발화해 순서 역전" \
      --issue "campbells를 더 당기면 late zone 최솟값이 monster_energy보다 늦음" \
      --next_step "monster_energy confirm을 늘려 뒤로 미루는 방향으로 전환"

  (--motivation/--issue/--next_step은 선택 — 리더보드 웹에서 행 오른쪽 ▼로 펼쳐서 봄)

Output:
  - 터미널에 TP/FP/FN/Precision/Recall/추정점수 출력
  - output/leaderboard.csv 누적 기록
  - output/leaderboard.html 갱신 (브라우저에서 바로 열기)
"""

import csv
import argparse
from datetime import datetime
from collections import Counter
import os
import json

GT_PATH          = "data/ground_truth_v2.csv"
LEADERBOARD_PATH = "output/leaderboard.csv"
HTML_PATH        = "output/leaderboard.html"


# ── 채점 공식 ────────────────────────────────────────────────────
def calc_accuracy_score(f1_pct: float) -> float:
    """정확도 점수: F1(%) × 0.4  →  최대 40점"""
    return round(f1_pct * 0.4, 1)


def calc_rtf_score(rtf: float) -> float:
    """RTF 점수: RTF ≤ 1 → 20점 만점. RTF > 1 → 상대평가(공식 미공개, None 반환)."""
    if rtf <= 1.0:
        return 20.0
    return None  # 상대평가 — 추정 불가


# ── 데이터 로드 ──────────────────────────────────────────────────
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
            action = _ACTION_MAP.get(action, action)
            if name and action:
                events.append((name, action))
    return events


def score_lcs(gt_events, sub_events):
    """Method 2: LCS-based order F1 (class-agnostic sequence matching)."""
    n, m = len(gt_events), len(sub_events)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gt_events[i - 1] == sub_events[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    tp = dp[n][m]
    fp = m - tp
    fn = n - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return tp, fp, fn, precision, recall, f1


def score_count(gt_events, sub_events):
    """Method 1: count-only F1 (Counter, order-agnostic). For reference only."""
    gt_counter  = Counter(gt_events)
    sub_counter = Counter(sub_events)
    all_keys = set(gt_counter) | set(sub_counter)

    tp = fp = fn = 0
    details = []
    for key in sorted(all_keys):
        g = gt_counter[key]
        s = sub_counter[key]
        tp += min(g, s)
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
    with open(path, encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def append_leaderboard(path, row_dict):
    """새 행 추가. row_dict에 기존 CSV에 없던 컬럼이 있으면(스키마 확장)
    전체를 다시 써서 과거 행에도 빈 값으로 채워 넣는다."""
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if exists:
        with open(path, encoding="utf-8-sig") as f:
            header = next(csv.reader(f))
    else:
        header = list(row_dict.keys())

    new_cols = [k for k in row_dict.keys() if k not in header]
    if exists and new_cols:
        rows = load_leaderboard(path)
        for r in rows:
            for k in new_cols:
                r.setdefault(k, "")
        fieldnames = header + new_cols
        rows.append(row_dict)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
        return

    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row_dict.get(k, "") for k in header})


# ── HTML 생성 ────────────────────────────────────────────────────
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
  html, body {{ max-width: 100%; overflow-x: hidden; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; padding: 24px; }}
  h1 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: 4px; color: #fff; }}
  .subtitle {{ color: #888; font-size: 0.82rem; margin-bottom: 8px; }}
  .formula  {{ color: #556; font-size: 0.78rem; margin-bottom: 24px;
               background:#141820; padding:8px 14px; border-radius:6px; display:inline-block; }}
  table {{ width: 100%; table-layout: fixed; border-collapse: collapse; font-size: 0.83rem; }}
  th {{ background: #1e2130; color: #aaa; padding: 8px 8px; text-align: left;
        font-weight: 600; border-bottom: 2px solid #2a2d3e; overflow: hidden; text-overflow: ellipsis; }}
  th.score-col {{ background:#1a2520; color:#7ee8a2; }}
  tr {{ border-bottom: 1px solid #1e2130; transition: background 0.15s; }}
  tr:hover {{ background: #1a1d2e; }}
  tr.history {{ opacity: 0.5; }}
  td {{ padding: 8px 8px; vertical-align: middle; overflow: hidden; text-overflow: ellipsis; }}
  .rank {{ font-weight: 700; color: #888; width: 30px; text-align: center; }}
  .rank.gold   {{ color: #f5c518; }}
  .rank.silver {{ color: #b0b0b0; }}
  .rank.bronze {{ color: #cd7f32; }}
  .desc {{ font-weight: 500; color: #ddd; white-space: normal; word-break: keep-all; line-height: 1.4; }}
  .ts   {{ color: #555; font-size: 0.72rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .bar-wrap {{ display: flex; align-items: center; gap: 5px; }}
  .bar {{ height: 7px; border-radius: 3px; min-width: 2px; max-width: 46px; }}
  .val {{ font-weight: 600; width: 38px; text-align: right; font-size:0.8rem; flex-shrink: 0; }}
  .f1  {{ color: #7ee8a2; }}
  .pre {{ color: #79c8ff; }}
  .rec {{ color: #ffb347; }}
  .tp  {{ color: #7ee8a2; }}
  .fp  {{ color: #ff7070; }}
  .fn  {{ color: #ffb347; }}
  .na  {{ color: #444; font-size: 0.8rem; }}
  /* Score columns */
  .acc-score  {{ color: #7ee8a2; font-weight: 700; }}
  .rtf-score  {{ color: #79c8ff; font-weight: 700; }}
  .rtf-val    {{ color: #aaa; }}
  .tot-score  {{ color: #fff;  font-weight: 800; font-size: 1rem; }}
  .score-cell {{ background: rgba(126,232,162,0.04); }}
  .badge {{ display:inline-block; padding:2px 7px; border-radius:10px; font-size:0.73rem; font-weight:600; }}
  .best      {{ background:#1a3a2a; color:#7ee8a2; }}
  .hist-badge {{ background:#1a1a2e; color:#445; border:1px solid #2a2d3e; }}
  .divider td {{ background:#151820; color:#445; font-size:0.73rem; padding:4px 12px; letter-spacing:.07em; }}
  .note {{ color:#445; font-size:0.72rem; margin-top:16px; }}

  /* 펼치기 화살표 + 상세 패널 */
  .expand-col {{ width: 34px; text-align: center; }}
  .expand-btn {{ background: none; border: none; color: #556; cursor: pointer; font-size: 0.9rem;
                 padding: 4px 8px; border-radius: 4px; transition: transform 0.18s, color 0.15s, background 0.15s; }}
  .expand-btn:hover {{ color: #7ee8a2; background: #1a1d2e; }}
  .expand-btn.open {{ transform: rotate(180deg); color: #7ee8a2; }}
  tr.main-row {{ cursor: pointer; }}
  tr.detail-row {{ display: none; }}
  tr.detail-row.open {{ display: table-row; }}
  tr.detail-row td {{ padding: 0; background: #12151f; }}
  .detail-box {{ padding: 14px 20px 16px 46px; border-left: 2px solid #2a2d3e; margin: 4px 12px 8px 12px; }}
  .detail-section {{ margin-bottom: 10px; font-size: 0.82rem; line-height: 1.55; }}
  .detail-section:last-child {{ margin-bottom: 0; }}
  .detail-label {{ display: inline-block; min-width: 74px; font-weight: 700; font-size: 0.76rem;
                    letter-spacing: .02em; }}
  .lbl-motivation {{ color: #79c8ff; }}
  .lbl-issue      {{ color: #ff8a8a; }}
  .lbl-next       {{ color: #ffb347; }}
  .detail-text {{ color: #ccc; }}
  .detail-empty {{ color: #445; font-style: italic; }}
  .detail-desc {{ color: #778; font-size: 0.78rem; margin-bottom: 12px; padding-bottom: 10px;
                  border-bottom: 1px solid #232640; }}
</style>
</head>
<body>
<h1>🏆 REDRED 리더보드</h1>
<p class="subtitle">파이프라인 버전별 정확도 + 속도 추적 &nbsp;|&nbsp; python tools/score.py --desc "설명" --rtf 0.742</p>
<div class="formula">
  정확도 40점 = F1 × 0.4 (공식 미공개, 추정) &nbsp;|&nbsp;
  RTF 20점 = RTF ≤ 1 → 만점(20점) / RTF > 1 → 상대평가(추정 불가) &nbsp;|&nbsp;
  발표 40점 = 직접 측정 불가 &nbsp;|&nbsp;
  <b>추정 총점 = 정확도 + RTF (최대 60점)</b>
</div>

<table>
<colgroup>
  <col style="width:3.5%"><col style="width:23%"><col style="width:8%">
  <col style="width:8%"><col style="width:8%"><col style="width:4%"><col style="width:4%"><col style="width:4%">
  <col style="width:8%"><col style="width:6%"><col style="width:8%">
  <col style="width:8%"><col style="width:11%"><col style="width:3.5%">
</colgroup>
<thead>
<tr>
  <th>#</th>
  <th>설명</th>
  <th>F1</th>
  <th>Precision</th>
  <th>Recall</th>
  <th>TP</th><th>FP</th><th>FN</th>
  <th class="score-col">정확도점수<br><small>/40</small></th>
  <th class="score-col">RTF</th>
  <th class="score-col">RTF점수<br><small>/20</small></th>
  <th class="score-col">추정총점<br><small>/60</small></th>
  <th>기록 시각</th>
  <th class="expand-col"></th>
</tr>
</thead>
<tbody id="tbody"></tbody>
</table>
<p class="note">* 추정 총점 = 정확도 + RTF (발표 40점 제외). 정확도 = order/LCS F1 × 0.4 (대회 공식 미공개, 추정). F1 컬럼은 order/LCS 기준.</p>

<script>
const data = {data_json};

function pct(s) {{
  if (!s || s === '-') return null;
  const v = parseFloat(s.replace('%',''));
  return isNaN(v) ? null : v;
}}
function num(s) {{
  if (!s || s === '-') return null;
  const v = parseFloat(s);
  return isNaN(v) ? null : v;
}}

const scored  = data.filter(r => pct(r.F1) !== null).sort((a,b) => {{
  const ta = (num(a.accuracy_score)||0) + (num(a.rtf_score)||0);
  const tb = (num(b.accuracy_score)||0) + (num(b.rtf_score)||0);
  return tb - ta;
}});
const history = data.filter(r => pct(r.F1) === null);

const tbody = document.getElementById('tbody');

function bar(v, cls, color) {{
  if (v === null) return `<span class="na">—</span>`;
  return `<div class="bar-wrap">
    <div class="bar" style="width:${{Math.round(v*1.3)}}px;background:${{color}}"></div>
    <span class="val ${{cls}}">${{v.toFixed(1)}}%</span>
  </div>`;
}}
function na(v, cls='', suffix='') {{
  if (v === null || v === '-' || v === undefined || v === '') return `<span class="na">—</span>`;
  return `<span class="${{cls}}">${{v}}${{suffix}}</span>`;
}}
function esc(s) {{
  if (s === undefined || s === null) return '';
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}}
function detailSection(label, cls, text) {{
  const body = text && text.trim()
    ? `<span class="detail-text">${{esc(text)}}</span>`
    : `<span class="detail-empty">기록 없음</span>`;
  return `<div class="detail-section"><span class="detail-label ${{cls}}">${{label}}</span>${{body}}</div>`;
}}
function detailRowHtml(rowId, colspan, row) {{
  return `<tr class="detail-row" id="detail-${{rowId}}">
    <td colspan="${{colspan}}">
      <div class="detail-box">
        <div class="detail-desc">${{esc(row.description)}}</div>
        ${{detailSection('🎯 계기', 'lbl-motivation', row.motivation)}}
        ${{detailSection('⚠️ 문제', 'lbl-issue', row.issue)}}
        ${{detailSection('➡️ 다음 단계', 'lbl-next', row.next_step)}}
      </div>
    </td>
  </tr>`;
}}

let uid = 0;
function toggleDetail(rowId) {{
  document.getElementById(`detail-${{rowId}}`).classList.toggle('open');
  document.getElementById(`btn-${{rowId}}`).classList.toggle('open');
}}

scored.forEach((row, i) => {{
  const rank = i + 1;
  const f1v  = pct(row.F1);
  const prev = pct(row.Precision);
  const recv = pct(row.Recall);
  const acc  = num(row.accuracy_score);
  const rtfv = num(row.RTF);
  const rtfs = num(row.rtf_score);
  const tot  = (acc !== null && rtfs !== null) ? +(acc + rtfs).toFixed(1) : null;
  const bestTot = num(scored[0].accuracy_score) + num(scored[0].rtf_score);
  const isB  = tot !== null && tot === +(bestTot).toFixed(1);
  const rankClass = rank===1?'gold':rank===2?'silver':rank===3?'bronze':'';
  const rid = `s${{uid++}}`;

  tbody.innerHTML += `<tr class="main-row" onclick="toggleDetail('${{rid}}')">
    <td class="rank ${{rankClass}}">${{rank}}</td>
    <td class="desc">${{row.description}} ${{isB ? '<span class="badge best">BEST</span>' : ''}}</td>
    <td>${{bar(f1v,  'f1',  '#7ee8a2')}}</td>
    <td>${{bar(prev, 'pre', '#79c8ff')}}</td>
    <td>${{bar(recv, 'rec', '#ffb347')}}</td>
    <td class="tp">${{row.TP ?? '<span class="na">—</span>'}}</td>
    <td class="fp">${{row.FP ?? '<span class="na">—</span>'}}</td>
    <td class="fn">${{row.FN ?? '<span class="na">—</span>'}}</td>
    <td class="score-cell">${{na(acc, 'acc-score', '점')}}</td>
    <td class="score-cell">${{na(rtfv, 'rtf-val')}}</td>
    <td class="score-cell">${{na(rtfs, 'rtf-score', '점')}}</td>
    <td class="score-cell">${{tot !== null ? `<span class="tot-score">${{tot}}점</span>` : '<span class="na">—</span>'}}</td>
    <td class="ts">${{row.timestamp}}</td>
    <td class="expand-col"><button class="expand-btn" id="btn-${{rid}}" onclick="event.stopPropagation(); toggleDetail('${{rid}}')">▼</button></td>
  </tr>`;
  tbody.innerHTML += detailRowHtml(rid, 14, row);
}});

if (history.length) {{
  tbody.innerHTML += `<tr class="divider"><td colspan="14">▼ 이전 기록 (점수 미측정 — 제출 CSV 없음)</td></tr>`;
  history.forEach(row => {{
    const rtfv = num(row.RTF);
    const rtfs = num(row.rtf_score);
    const rid = `h${{uid++}}`;
    tbody.innerHTML += `<tr class="history main-row" onclick="toggleDetail('${{rid}}')">
      <td class="rank"><span class="na">—</span></td>
      <td class="desc">${{row.description}} <span class="badge hist-badge">이벤트: ${{row.total_sub}}</span></td>
      <td colspan="6"><span class="na" style="font-size:0.8rem">정확도 미측정</span></td>
      <td class="score-cell"><span class="na">—</span></td>
      <td class="score-cell">${{na(rtfv, 'rtf-val')}}</td>
      <td class="score-cell">${{na(rtfs, 'rtf-score', '점')}}</td>
      <td class="score-cell"><span class="na">—</span></td>
      <td class="ts">${{row.timestamp}}</td>
      <td class="expand-col"><button class="expand-btn" id="btn-${{rid}}" onclick="event.stopPropagation(); toggleDetail('${{rid}}')">▼</button></td>
    </tr>`;
    tbody.innerHTML += detailRowHtml(rid, 14, row);
  }});
}}
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


# ── main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub",  default="output/submission.csv")
    parser.add_argument("--gt",   default=GT_PATH)
    parser.add_argument("--desc", default="", help="이번 버전 설명")
    parser.add_argument("--motivation", default="", help="이 시도를 하게 된 계기/풀려던 문제")
    parser.add_argument("--issue", default="", help="이 시도로 드러난 문제/부작용 (없으면 빈 문자열)")
    parser.add_argument("--next_step", default="", help="다음에 시도할 것")
    parser.add_argument("--rtf",  type=float, default=None, help="측정된 RTF 값 (선택)")
    parser.add_argument("--lb",   default=LEADERBOARD_PATH)
    parser.add_argument("--html", default=HTML_PATH)
    args = parser.parse_args()

    gt_events  = load_gt(args.gt)
    sub_events = load_submission(args.sub)

    # Primary metric: order (LCS) F1
    tp, fp, fn, precision, recall, f1 = score_lcs(gt_events, sub_events)
    # Reference: count F1
    ctp, cfp, cfn, cprecision, crecall, cf1, details = score_count(gt_events, sub_events)

    f1_pct      = f1 * 100
    cf1_pct     = cf1 * 100
    acc_score   = calc_accuracy_score(f1_pct)
    rtf_s       = calc_rtf_score(args.rtf) if args.rtf is not None else None
    total_score = round(acc_score + rtf_s, 1) if rtf_s is not None else None

    print(f"\n{'='*58}")
    print(f"  GT 이벤트:      {len(gt_events):>4}개")
    print(f"  제출 이벤트:    {len(sub_events):>4}개")
    print(f"{'─'*58}")
    print(f"  [order/LCS]  TP={tp} FP={fp} FN={fn}")
    print(f"  Precision:      {precision*100:>6.1f}%")
    print(f"  Recall:         {recall*100:>6.1f}%")
    print(f"  F1 (order):     {f1_pct:>6.1f}%  ← 정확도 점수 기준")
    print(f"{'─'*58}")
    print(f"  [count ref]  F1={cf1_pct:.1f}%  TP={ctp} FP={cfp} FN={cfn}")
    print(f"{'─'*58}")
    print(f"  정확도 점수:    {acc_score:>5.1f}점  / 40점  (order F1 기준)")
    if args.rtf is not None:
        print(f"  RTF:            {args.rtf:>5.3f}")
        if rtf_s is not None:
            print(f"  RTF 점수:       {rtf_s:>5.1f}점  / 20점")
            print(f"  추정 총점:      {total_score:>5.1f}점  / 60점  (발표 40점 제외)")
        else:
            print(f"  RTF 점수:       상대평가 (RTF > 1, 공식 미공개)")
    else:
        print(f"  RTF:            (--rtf 미입력)")
    print(f"{'='*58}\n")

    if details:
        print("count 불일치 항목 (클래스 / 액션 / GT수 / 제출수):")
        for cls, action, g, s in sorted(details, key=lambda x: -abs(x[2]-x[3]))[:15]:
            diff = s - g
            mark = "↑" if diff > 0 else "↓"
            print(f"  {mark}{abs(diff):>2}  {cls:<50} {action}  GT={g} Sub={s}")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    row_dict = {
        "timestamp":      ts,
        "description":    args.desc,
        "total_gt":       len(gt_events),
        "total_sub":      len(sub_events),
        "TP":             tp,
        "FP":             fp,
        "FN":             fn,
        "Precision":      f"{precision*100:.1f}%",
        "Recall":         f"{recall*100:.1f}%",
        "F1":             f"{f1_pct:.1f}%",
        "RTF":            args.rtf if args.rtf is not None else "-",
        "accuracy_score": acc_score,
        "rtf_score":      rtf_s if rtf_s is not None else ("상대평가" if args.rtf is not None else "-"),
        "total_score":    total_score if total_score is not None else "-",
        "motivation":     args.motivation,
        "issue":          args.issue,
        "next_step":      args.next_step,
    }
    append_leaderboard(args.lb, row_dict)

    lb_rows = load_leaderboard(args.lb)
    generate_html(lb_rows, args.html)

    print(f"리더보드 갱신 완료")
    print(f"  CSV  → {args.lb}")
    print(f"  HTML → {args.html}  (브라우저에서 열기)")


if __name__ == "__main__":
    main()
