from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

# ai_samples.py로 만든 샘플을 ai-server run_report로 하나씩 돌리고, 정답(labels.csv)과 비교한 성적표를 만든다
# 표준 라이브러리만 쓴다 — ai-server 가상환경의 python으로 그대로 실행할 수 있게

ML_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = ML_DIR / "exports" / "ai_samples"
REPORT_PATH = ML_DIR / "reports" / "ai_recommendation_v3.md"
DEFAULT_AI_SERVER = ML_DIR.parent / "ai-server"
RESULT_COLUMNS = ("caseId", "status", "recommendation", "overall", "seconds", "error")
TIMEOUT_SECONDS = 900

REJECT, CHECK, ACCEPT = "REJECT_REVIEW", "NEEDS_CHECK", "ACCEPT_REVIEW"
CONDITION_KO = {
    "NORMAL": "정상",
    "ODD_NEW_CARD_DEVICE": "정상 · 새 카드+기기",
    "ABROAD_TRAVEL": "정상 · 해외 여행",
    "ABROAD_RESIDENT": "정상 · 해외 거주",
    "ODD_BURST": "정상 · 몰아서 결제",
    "ODD_BIG_FIRST": "정상 · 처음 큰 금액",
    "FRAUD_SAME_COUNTRY": "사기 · 평소 국가로 위장",
    "FRAUD_FOREIGN": "사기 · 해외",
    "FRAUD_HIGH_RISK": "사기 · 고위험 국가(NG)",
}


# 채점 — 사기에 매입 진행은 위험한 놓침, 정상에 승인 취소는 잘못된 취소, 추가 확인은 어느 쪽이든 보류
def grade(fraud: bool, recommendation: str) -> str:
    if recommendation == CHECK:
        return "보류"
    if fraud:
        return "맞음" if recommendation == REJECT else "위험한 놓침"
    return "맞음" if recommendation == ACCEPT else "잘못된 취소"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


# run_report는 실패 이유를 stdout에 JSON({"code", "reason"})으로 찍는다 — stderr는 진행 로그라 이유가 아니다
def failure_reason(stdout: str, stderr: str) -> str:
    try:
        error = json.loads(stdout)
        return f"{error.get('code')}: {error.get('reason')}"[:300]
    except (json.JSONDecodeError, AttributeError):
        return (stderr or stdout).strip()[-300:]


def run_one(sample: Path, ai_server: Path, python: str) -> dict:
    started = time.monotonic()
    try:
        done = subprocess.run(
            [python, "-m", "scripts.run_report", "--sample", str(sample)],
            cwd=ai_server,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {"status": "fail", "error": f"{TIMEOUT_SECONDS}초 초과", "seconds": f"{time.monotonic() - started:.1f}"}
    seconds = f"{time.monotonic() - started:.1f}"
    if done.returncode != 0:
        return {"status": "fail", "error": failure_reason(done.stdout, done.stderr), "seconds": seconds}
    report = json.loads(done.stdout)
    analyses = report.get("paymentAnalyses") or []
    return {
        "status": "ok",
        "recommendation": analyses[0]["recommendation"] if analyses else "",
        "overall": report["verdict"]["overallRecommendation"],
        "seconds": seconds,
        "error": "",
    }


# 이미 성공한 사건은 건너뛴다 — 오래 걸리는 Ollama 실행을 끊었다가 이어서 돌릴 수 있게
def run(label: str, ai_server: Path, python: str, limit: int | None, sample_dir: Path = SAMPLE_DIR) -> Path:
    labels = read_csv(sample_dir / "labels.csv")[:limit]
    result_path = sample_dir / f"results-{label}.csv"
    done = {r["caseId"]: r for r in read_csv(result_path)} if result_path.exists() else {}
    for number, row in enumerate(labels, start=1):
        if done.get(row["caseId"], {}).get("status") == "ok":
            continue
        result = {"caseId": row["caseId"], "recommendation": "", "overall": "", **run_one(sample_dir / row["sampleFile"], ai_server, python)}
        done[row["caseId"]] = result
        answer = "사기" if row["fraudLabel"] == "1" else "정상"
        shown = result["recommendation"] or f"실패 — {result['error'][:120]}"
        print(f"[{number}/{len(labels)}] {label} · case {row['caseId']} ({answer}) → {shown} · {result['seconds']}초", flush=True)
        with result_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(done.values())
    return result_path


def _line(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def build_report(labels: list[dict], runs: dict[str, list[dict]]) -> str:
    by_case = {r["caseId"]: r for r in labels}
    out = [
        "# AI 조사 보고서 권고 성적표 (v3 합성 데이터)",
        "",
        "- 만든 방법: `python ai_samples.py` → `python score_ai_samples.py --label <이름>` (README 19 · 20번)",
        f"- 샘플: v3 REVIEW 결제 {len(labels)}건(사건 1개 = 결제 1개) · 정상 {sum(r['fraudLabel'] == '0' for r in labels)} · "
        f"사기 {sum(r['fraudLabel'] == '1' for r in labels)}",
        "- 채점: 사기에 승인 취소 = 맞음 · 매입 진행 = **위험한 놓침**, 정상에 매입 진행 = 맞음 · 승인 취소 = **잘못된 취소**, 추가 확인 = 보류",
        "- 드문 정황(#490 기준): 거절 후 재시도 · 카드 테스팅 · 같은 카드나 기기를 다른 고객이 씀. 흔한 신호만 있는 사기는 추가 확인이 기준대로의 답",
        "- 단서: 비슷한 과거 사건 Tool은 정답이 새지 않게 비워 둠(\"비슷한 사건 없음\"). IP 원문이 없어 같은 IP 비교는 \"정보 없음\"",
        "",
    ]
    for label, results in runs.items():
        ok = [r for r in results if r["status"] == "ok"]
        grades = Counter(grade(by_case[r["caseId"]]["fraudLabel"] == "1", r["recommendation"]) for r in ok)
        seconds = [float(r["seconds"]) for r in ok]
        out += [
            f"## {label}",
            "",
            f"- 완료 {len(ok)} / {len(results)}건 · 실패 {len(results) - len(ok)} · 평균 {sum(seconds) / len(seconds) if seconds else 0:.1f}초",
            f"- 맞음 {grades['맞음']} · 보류 {grades['보류']} · 위험한 놓침 {grades['위험한 놓침']} · 잘못된 취소 {grades['잘못된 취소']}",
            "",
            _line(["상황", "드문 정황", "건수", "승인 취소", "추가 확인", "매입 진행", "실패"]),
            _line(["---", "---", "---:", "---:", "---:", "---:", "---:"]),
        ]
        groups = Counter((by_case[r["caseId"]]["condition"], by_case[r["caseId"]]["rareSignal"]) for r in results)
        for condition in CONDITION_KO:
            for rare in ("1", "0"):
                members = [r for r in results if (by_case[r["caseId"]]["condition"], by_case[r["caseId"]]["rareSignal"]) == (condition, rare)]
                if not groups[(condition, rare)]:
                    continue
                counts = Counter(r["recommendation"] if r["status"] == "ok" else "fail" for r in members)
                out.append(
                    _line([CONDITION_KO[condition], "있음" if rare == "1" else "없음", len(members), counts[REJECT], counts[CHECK], counts[ACCEPT], counts["fail"]])
                )
        out.append("")
    if len(runs) > 1:
        out += ["## 실행끼리 같은 권고가 나온 비율", ""]
        for a, b in combinations(runs, 2):
            left = {r["caseId"]: r["recommendation"] for r in runs[a] if r["status"] == "ok"}
            right = {r["caseId"]: r["recommendation"] for r in runs[b] if r["status"] == "ok"}
            shared = left.keys() & right.keys()
            same = sum(left[c] == right[c] for c in shared)
            out.append(f"- {a} ↔ {b}: {same} / {len(shared)}건")
        out.append("")
    return "\n".join(out)


def write_report(sample_dir: Path = SAMPLE_DIR, report_path: Path = REPORT_PATH) -> Path:
    labels = read_csv(sample_dir / "labels.csv")
    runs = {p.stem.removeprefix("results-"): read_csv(p) for p in sorted(sample_dir.glob("results-*.csv"))}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(labels, runs), encoding="utf-8")
    return report_path


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description="AI 권고를 v3 샘플 정답과 비교")
    parser.add_argument("--label", help="실행 이름 (예: ollama-1, openai-1). 없으면 성적표만 다시 만든다")
    parser.add_argument("--limit", type=int, help="앞에서부터 N건만 (처음 시간 재기용)")
    parser.add_argument("--ai-server", type=Path, default=DEFAULT_AI_SERVER)
    parser.add_argument("--python", default=sys.executable, help="ai-server를 실행할 python")
    args = parser.parse_args()
    if args.label:
        if not (args.ai_server / "scripts" / "run_report.py").exists():
            parser.error(f"ai-server 폴더를 찾지 못했습니다: {args.ai_server} (--ai-server로 지정)")
        run(args.label, args.ai_server.resolve(), args.python, args.limit)
    print(f"성적표 → {write_report()}")


if __name__ == "__main__":
    main()
