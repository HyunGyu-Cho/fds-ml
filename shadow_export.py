# Ⓚ Shadow 기록 내보내기 — mariadb 클라이언트로 sql/shadow_export.sql을 돌려 data/shadow/shadow_export.csv로 저장한다
# 새 pip 패키지 없이 MariaDB 설치에 들어 있는 mariadb.exe를 쓰고, 비밀번호는 실행할 때 입력받는다
import argparse
import csv
import getpass
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
SQL_PATH = ML_DIR / "sql" / "shadow_export.sql"
WINDOWS_CLIENTS = r"C:\Program Files\MariaDB *\bin\mariadb.exe"
BATCH_ESCAPES = {"t": "\t", "n": "\n", "0": "\0", "\\": "\\"}


# --batch는 탭 · 줄바꿈 · 역슬래시를 \t · \n · \\ 로 바꿔 내보내므로 되돌린다
def unescape(value: str) -> str:
    out, i = [], 0
    while i < len(value):
        if value[i] == "\\" and i + 1 < len(value) and value[i + 1] in BATCH_ESCAPES:
            out.append(BATCH_ESCAPES[value[i + 1]])
            i += 2
        else:
            out.append(value[i])
            i += 1
    return "".join(out)


# 첫 줄은 머리글. NULL 글자는 빈칸으로 바꾼다 (shadow_whatif.load_rows가 빈칸을 없는 값으로 읽음)
def batch_to_rows(lines: list[str]) -> list[list[str]]:
    rows = [lines[0].split("\t")]
    for line in lines[1:]:
        if line:
            rows.append(["" if cell == "NULL" else unescape(cell) for cell in line.split("\t")])
    return rows


def write_csv(rows: list[list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        csv.writer(file).writerows(rows)


def find_client(given: str | None) -> str:
    if given:
        return given
    found = shutil.which("mariadb") or shutil.which("mysql") or next(iter(sorted(glob.glob(WINDOWS_CLIENTS), reverse=True)), None)
    if not found:
        sys.exit("mariadb 클라이언트를 찾지 못했어요. --client \"C:\\Program Files\\MariaDB 12.3\\bin\\mariadb.exe\" 처럼 알려 주세요.")
    return found


# 한글 주석이 Windows 명령 인자에서 깨지지 않게 -- 주석 줄은 빼고 보낸다
def query_text() -> str:
    lines = SQL_PATH.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("--"))


def run_query(client: str, args: argparse.Namespace, password: str) -> list[str]:
    # 비밀번호는 명령 줄에 안 남게 임시 설정 파일로 넘기고 바로 지운다
    handle, option_file = tempfile.mkstemp(suffix=".cnf")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(f"[client]\npassword={password}\n")
        command = [
            client, f"--defaults-extra-file={option_file}", "--protocol=TCP", "-h", args.host, "-P", str(args.port),
            "-u", args.user, "--batch", "--default-character-set=utf8mb4", args.database,
            "-e", query_text(),
        ]
        result = subprocess.run(command, capture_output=True)
    finally:
        os.remove(option_file)
    if result.returncode != 0:
        sys.exit(f"조회 실패 (종료 코드 {result.returncode}): {result.stderr.decode('utf-8', 'replace').strip()}")
    return result.stdout.decode("utf-8").splitlines()


def main() -> int:
    parser = argparse.ArgumentParser(description="Shadow 기록을 CSV로 내보내기 (mariadb 클라이언트 사용)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3307)
    parser.add_argument("--user", default="root")
    parser.add_argument("--database", default="donworry")
    parser.add_argument("--client", help="mariadb.exe 경로 (없으면 PATH → C:\\Program Files\\MariaDB *\\bin 순으로 찾음)")
    parser.add_argument("--out", type=Path, default=ML_DIR / "data" / "shadow" / "shadow_export.csv")
    args = parser.parse_args()

    client = find_client(args.client)
    password = getpass.getpass(f"{args.user}@{args.host}:{args.port}/{args.database} 비밀번호: ")
    lines = run_query(client, args, password)
    if not lines:
        sys.exit("결과가 비어 있어요. DB 이름과 ml_shadow_scores 기록을 확인해 주세요.")
    rows = batch_to_rows(lines)
    write_csv(rows, args.out)
    print(f"클라이언트: {client}")
    print(f"저장: {args.out} ({len(rows) - 1}행)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
