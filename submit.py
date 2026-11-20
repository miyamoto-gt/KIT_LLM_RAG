"""KIT RAG コンペの提出クライアント（標準ライブラリのみ）。

使い方:
  python submit.py login --url http://192.168.38.239:3000
  python submit.py questions [-o questions.csv]
  python submit.py submit submission.csv [--code my_rag/] [--name 提出名] [--no-wait]
  python submit.py run my_rag/            # 提出前にローカルで main.py を本番と同じ条件で試す
  python submit.py status
  python submit.py final <session_id> [--off]
  python submit.py leaderboard [--private]
"""
import argparse
import csv
import getpass
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import base64
import subprocess
import zipfile
from pathlib import Path

CONFIG_PATH = Path.home() / ".kitrag.json"


def load_config():
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit("先に `python submit.py login --url <サーバーURL>` を実行してください。")


def save_config(config):
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except OSError:
        pass


CODE_STATUS = {"queued": "コード実行待ち", "running": "コード実行中", "succeeded": "コード実行済み", "failed": "コード実行失敗"}
CODE_MAX_BYTES = 30 * 1024 * 1024
SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", ".ipynb_checkpoints", "node_modules"}


def build_code_zip(path):
    """フォルダなら zip にまとめ、.zip ならそのまま読む。main.py が直下に必要。"""
    path = Path(path)
    if path.is_file():
        data = path.read_bytes()
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        if "main.py" not in names and not any(n.count("/") == 1 and n.endswith("/main.py") for n in names):
            sys.exit("zip の直下に main.py がありません。")
    elif path.is_dir():
        if not (path / "main.py").is_file():
            sys.exit(f"{path} に main.py がありません。")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(path.rglob("*")):
                if file.is_file() and not (set(file.relative_to(path).parts) & SKIP_DIRS):
                    archive.write(file, file.relative_to(path).as_posix())
        data = buffer.getvalue()
    else:
        sys.exit(f"{path} が見つかりません。")
    if len(data) > CODE_MAX_BYTES:
        sys.exit(f"コードの zip が {len(data) / 1024 / 1024:.1f}MB あります（30MBまで）。")
    return data


def request(config, method, path, body=None, token=True):
    headers = {"content-type": "application/json"} if body is not None else {}
    if token:
        headers["authorization"] = f"Bearer {config['token']}"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(config["url"].rstrip("/") + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            payload = json.loads(error.read().decode("utf-8"))
        except ValueError:
            payload = {"error": error.reason}
        if error.code == 401 and token:
            sys.exit("ログインの有効期限が切れました。`python submit.py login` をやり直してください。")
        detail = payload.get("error", "")
        if payload.get("missing"):
            detail += f"\n  不足している question_id: {', '.join(payload['missing'])}"
        sys.exit(f"エラー ({error.code} {payload.get('code', '')}): {detail}")
    except urllib.error.URLError as error:
        sys.exit(f"サーバーに接続できません: {error.reason}")


def score_text(score):
    return "—" if score is None else f"{score['percentage']:.1f} ({score['points']}/{score['total']})"


def cmd_login(args):
    config = {"url": args.url}
    username = args.username or input("ユーザー名: ").strip()
    password = getpass.getpass("パスワード: ")
    result = request(config, "POST", "/api/auth/login", {"username": username, "password": password}, token=False)
    config["token"] = result["access_token"]
    save_config(config)
    print(f"{result['user']['username']} でログインしました（{result['expires_in'] // 3600}時間有効）。")


def cmd_questions(args):
    config = load_config()
    data = request(config, "GET", "/api/competition/questions")
    with open(args.output, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["question_id", "question"])
        for item in data["questions"]:
            writer.writerow([item["question_id"], item["question"]])
    print(f"{data['total']}問を {args.output} に保存しました。")


def read_submission(path):
    # Excel on Japanese Windows saves plain "CSV" as Shift_JIS (cp932); accept both.
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("cp932")
        except UnicodeDecodeError:
            sys.exit("CSVの文字コードを判別できません。UTF-8 で保存してください。")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not reader.fieldnames or "question_id" not in reader.fieldnames or "answer" not in reader.fieldnames:
        sys.exit("CSVの1行目に question_id と answer の列が必要です。")
    return [{"question_id": row["question_id"].strip(), "answer": row["answer"] or ""} for row in reader if row["question_id"]]


def cmd_submit(args):
    config = load_config()
    answers = read_submission(args.file)
    body = {"answers": answers}
    if args.name:
        body["submission_name"] = args.name
    if args.code:
        code = build_code_zip(args.code)
        body["code"] = {"filename": Path(args.code).name, "content_base64": base64.b64encode(code).decode("ascii")}
        print(f"コード: {args.code}（{len(code) / 1024:.0f}KB）")
    created = request(config, "POST", "/api/submissions", body)
    print(f"「{created['submission_name']}」を受け付けました（{len(answers)}問, session_id={created['session_id']}）。")
    if args.no_wait:
        return
    while True:
        sessions = request(config, "GET", "/api/me/sessions")["sessions"]
        current = next(item for item in sessions if item["session_id"] == created["session_id"])
        if current["status"] == "completed":
            print(f"\n採点が完了しました。Public スコア: {score_text(current['public_score'])}")
            if current.get("code_status") == "failed":
                print(f"コードの実行に失敗しました: {current.get('code_error')}（Private の回答は0点になります）")
            elif current.get("code_status") == "succeeded":
                print("コードは正常に実行されました。Private スコアは結果発表後に表示されます。")
            return
        code = CODE_STATUS.get(current.get("code_status"), "")
        print(f"\r採点中 {current['answered']}/{current['total']}  {code}   ", end="", flush=True)
        time.sleep(4)


def cmd_status(args):
    config = load_config()
    data = request(config, "GET", "/api/me/sessions")
    usage = request(config, "GET", "/api/me/usage")
    print(f"本日の提出: {usage['new_sessions']} / {usage['new_session_limit']}")
    for item in data["sessions"]:
        final = "★" if item["selected_for_final"] else " "
        private = f"  Private {score_text(item['private_score'])}" if data.get("private_revealed") else ""
        code = f"  [{CODE_STATUS.get(item['code_status'], item['code_status'])}]" if item.get("code_status") else ""
        error = f"\n      {item['code_error']}" if item.get("code_error") else ""
        print(f"{final} {item['created_at'][:16]}  {item['status']:<9}  Public {score_text(item['public_score'])}{private}{code}  {item['submission_name']}  ({item['session_id']}){error}")


def cmd_run(args):
    """本番のサンドボックスと同じ環境変数で main.py を手元で実行する（質問は Public だけ）。"""
    config = load_config()
    folder = Path(args.folder).resolve()
    if not (folder / "main.py").is_file():
        sys.exit(f"{folder} に main.py がありません。")
    questions = Path(args.questions).resolve()
    if not questions.is_file():
        sys.exit(f"{questions} がありません。先に `python submit.py questions` を実行してください。")
    models = [m["id"] for m in request(config, "GET", "/api/llm/v1/models")["data"]]
    llm_url = config["url"].rstrip("/") + "/api/llm/v1"
    output = Path(args.output).resolve()
    env = {**os.environ, "INPUT_PATH": str(questions), "OUTPUT_PATH": str(output),
           "LLM_BASE_URL": llm_url, "OPENAI_BASE_URL": llm_url, "OPENAI_API_KEY": config["token"],
           "LLM_MODEL": next((m for m in models if "embed" not in m and "bge" not in m), models[0]),
           "EMBEDDING_MODEL": next((m for m in models if "embed" in m or "bge" in m), models[-1])}
    print(f"LLM: {env['LLM_MODEL']} / 埋め込み: {env['EMBEDDING_MODEL']} / 出力: {output}")
    started = time.time()
    result = subprocess.run([sys.executable, "main.py"], cwd=folder, env=env)
    print(f"終了コード {result.returncode}（{time.time() - started:.0f}秒）")
    if result.returncode == 0 and output.is_file():
        print(f"{len(read_submission(output))}問の回答を {output} に書き出しました。これを submit に渡せます。")


def cmd_final(args):
    config = load_config()
    result = request(config, "POST", f"/api/submissions/{args.session_id}/final", {"selected": not args.off})
    print("最終提出に選びました。" if result["selected_for_final"] else "最終提出から外しました。")


def cmd_leaderboard(args):
    config = load_config()
    board = "private" if args.private else "public"
    data = request(config, "GET", f"/api/leaderboard?board={board}")
    for entry in data["leaderboard"]:
        print(f"{entry['rank']:>3}. {entry['username']:<24} {entry['score']['percentage']:>5.1f}  (提出 {entry['submission_count']}回)")


def main():
    parser = argparse.ArgumentParser(description="KIT RAG コンペの提出クライアント")
    commands = parser.add_subparsers(dest="command", required=True)
    login = commands.add_parser("login", help="ログインしてトークンを保存する")
    login.add_argument("--url", required=True, help="例: http://192.168.38.239:3000")
    login.add_argument("--username")
    login.set_defaults(handler=cmd_login)
    questions = commands.add_parser("questions", help="質問CSVを保存する")
    questions.add_argument("-o", "--output", default="questions.csv")
    questions.set_defaults(handler=cmd_questions)
    submit = commands.add_parser("submit", help="回答CSV（question_id,answer）を提出する")
    submit.add_argument("file")
    submit.add_argument("--code", help="main.py を含むフォルダか zip（Private 問題はサーバーでこのコードを実行して評価）")
    submit.add_argument("--name", help="提出名（80文字以内）")
    submit.add_argument("--no-wait", action="store_true", help="採点の完了を待たない")
    submit.set_defaults(handler=cmd_submit)
    run = commands.add_parser("run", help="main.py を本番と同じ条件でローカル実行する")
    run.add_argument("folder", help="main.py があるフォルダ")
    run.add_argument("--questions", default="questions.csv")
    run.add_argument("-o", "--output", default="submission.csv")
    run.set_defaults(handler=cmd_run)
    commands.add_parser("status", help="自分の提出とスコアを表示する").set_defaults(handler=cmd_status)
    final = commands.add_parser("final", help="最終提出を選ぶ（最大2件）")
    final.add_argument("session_id")
    final.add_argument("--off", action="store_true", help="選択を外す")
    final.set_defaults(handler=cmd_final)
    board = commands.add_parser("leaderboard", help="ランキングを表示する")
    board.add_argument("--private", action="store_true", help="結果発表後の Private ランキング")
    board.set_defaults(handler=cmd_leaderboard)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
