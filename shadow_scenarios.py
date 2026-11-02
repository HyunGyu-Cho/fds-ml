# Ⓚ 만약 비교용 시나리오 결제 묶음 — 돈워리 로컬 서버에 정상 · 사기 결제를 섞어 보내고 정답 라벨을 CSV로 남긴다
# 서명 · 전송은 팀 저장소 scripts/demo/cakeshop_pay.py의 load_key · send를 그대로 쓴다 (팀 저장소는 고치지 않음)
import argparse
import csv
import importlib.util
import json
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
DEFAULT_DEMO = ML_DIR.parent / "scripts" / "demo" / "cakeshop_pay.py"


@dataclass(frozen=True)
class Payment:
    scenario: str
    label: str  # 정상 / 사기
    customer: str
    amount: int
    card: str
    device: str
    country: str = "KR"


def load_demo(path: Path):
    if not path.exists():
        sys.exit(f"cakeshop_pay.py를 찾지 못했어요: {path} (--demo 로 위치를 알려 주세요)")
    spec = importlib.util.spec_from_file_location("cakeshop_pay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def new_customer() -> str:
    return str(random.randint(10**11, 10**12 - 1))


# 손님마다 기본 카드 · 기기 하나. 탈취 · 공유 카드 시나리오만 바꿔 끼운다
def build_payments(run_id: str, risk_country: str, foreign_country: str) -> list[Payment]:
    payments: list[Payment] = []

    def own(scenario, label, customer, amount, **change):
        base = dict(card=f"tok_whatif_{customer}", device=f"device-whatif-{customer[-6:]}")
        base.update(change)
        payments.append(Payment(scenario, label, customer, amount, **base))

    for _ in range(8):
        own("N1 소액 새 손님", "정상", new_customer(), 30_000)
    for _ in range(3):
        c = new_customer()
        own("N2 단골 소액", "정상", c, 45_000)
        own("N2 단골 소액", "정상", c, 45_000)
    for _ in range(3):
        c = new_customer()
        own("N3 단골 고액", "정상", c, 40_000)
        own("N3 단골 고액", "정상", c, 700_000)
    for _ in range(2):
        c = new_customer()
        own("N4 단골 해외 결제", "정상", c, 50_000)
        own("N4 단골 해외 결제", "정상", c, 50_000, country=foreign_country)
    for _ in range(2):
        c = new_customer()
        for _ in range(3):
            own("F1 고액 연속 (기존 시연)", "사기", c, 600_000)
    for _ in range(3):
        c = new_customer()
        own("F2 탈취 전 본인 결제", "정상", c, 40_000)
        own("F2 계정 탈취 (새 기기 · 새 카드 · 고액)", "사기", c, 800_000,
            card=f"tok_whatif_stolen_{uuid.uuid4().hex[:8]}", device=f"device-whatif-x{uuid.uuid4().hex[:5]}")
    for _ in range(3):
        own("F3 고위험 국가 새 손님", "사기", new_customer(), 200_000, country=risk_country)
    shared = f"tok_whatif_shared_{run_id}"
    for _ in range(3):
        own("F4 손님 여럿이 같은 카드", "사기", new_customer(), 150_000, card=shared)
    for _ in range(2):
        c = new_customer()
        for _ in range(4):
            own("F5 소액 반복", "사기", c, 90_000)
    return payments


def send_payment(demo, key: bytes, payment: Payment, n: int) -> tuple[int, dict]:
    order_ref = f"WHATIF-{datetime.now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}-{n}"
    created_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = {
        "orderRef": order_ref,
        "customerRef": payment.customer,
        "amount": payment.amount,
        "orderCreatedAt": created_at,
        "cardToken": payment.card,
        "cardDisplayLabel": f"시나리오카드 ****{payment.card[-4:]}",
        "deviceId": payment.device,
        "ipAddress": "203.0.113.5",
        "ipCountryCode": payment.country,
        "buyerEmail": f"whatif-{payment.customer}@example.test",
    }
    raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
    status, data = demo.send(key, "POST", "/api/payments", raw, str(uuid.uuid4()))
    # 202는 처리 중 — 끝날 때까지 다시 조회한다 (cakeshop_pay.pay와 같음)
    for _ in range(10):
        if status != 202 or "paymentId" not in data:
            break
        time.sleep(1)
        status, data = demo.send(key, "GET", f"/api/payments/{data['paymentId']}")
    return status, data


def main() -> int:
    parser = argparse.ArgumentParser(description="만약 비교용 시나리오 결제 묶음")
    parser.add_argument("--demo", type=Path, default=Path(os.environ.get("DONWORRY_DEMO_PAY", DEFAULT_DEMO)))
    parser.add_argument("--risk-country", default="NG", help="fds_high_risk_countries에 있는 국가 코드")
    parser.add_argument("--foreign-country", default="US", help="고위험 목록에 없는 해외 국가 코드")
    parser.add_argument("--out", type=Path, default=ML_DIR / "data" / "shadow" / "scenario_labels.csv")
    parser.add_argument("--dry-run", action="store_true", help="보내지 않고 시나리오만 출력")
    args = parser.parse_args()

    run_id = uuid.uuid4().hex[:8]
    payments = build_payments(run_id, args.risk_country, args.foreign_country)
    if args.dry_run:
        for p in payments:
            print(f"{p.scenario:<32} {p.label}  {p.amount:>9,}원  국가 {p.country}")
        print(f"\n총 {len(payments)}건")
        return 0

    demo = load_demo(args.demo)
    key = demo.load_key()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    print(f"시나리오 결제 {len(payments)}건을 보냅니다 (묶음 {run_id})\n")
    with open(args.out, "a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(["payment_id", "scenario", "label", "payment_status", "amount", "run_id"])
        for n, payment in enumerate(payments, start=1):
            status, data = send_payment(demo, key, payment, n)
            if status >= 400:
                print(f"{n}번째 실패: HTTP {status} {data.get('code', '')} {data.get('detail', data)}")
                print("→ cakeshop_pay.py가 먼저 성공하는지 확인해 주세요 (서명 키 · payment.http.enabled)")
                return 1
            writer.writerow([data.get("paymentId"), payment.scenario, payment.label,
                             data.get("paymentStatus"), payment.amount, run_id])
            file.flush()
            print(f"{n:>2}. {payment.scenario:<32} {payment.label}  {payment.amount:>9,}원  → {data.get('paymentStatus')}")
            time.sleep(1)
    print(f"\n라벨 저장: {args.out}")
    print("ML Shadow 스케줄러가 채점하도록 1~2분 기다린 뒤 sql/shadow_export.sql을 실행해 주세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
