# 01_prepare_data_dual_input.py

import json
import re
import random
from datetime import datetime
from pathlib import Path

# ============================
# 1. KONFIGURATION
# ============================
SCRIPT_DIR = Path(__file__).parent

LOGS_DIR = SCRIPT_DIR / "logs"     # z.B. moderator-only.txt
DAATA_DIR = SCRIPT_DIR / "daata"   # Discord Export TXT
OUTPUT_DIR = SCRIPT_DIR / "data2"

# ============================
# 2. PARSER – logs/
# ============================

LOGS_REGEX = re.compile(
    r"\[(?P<timestamp>[\d\-T:\.]+)\]\s+(?P<author>[^:]+):\s*(?P<message>.*)"
)

def parse_logs_line(line: str):
    match = LOGS_REGEX.match(line)
    if not match:
        return None

    try:
        timestamp = datetime.fromisoformat(match.group("timestamp"))
    except ValueError:
        return None

    author = match.group("author").strip()
    message = match.group("message").strip()

    if (
        not message
        or message.startswith("[ATTACHMENT")
        or message.startswith("<@")
    ):
        return None

    return timestamp, author, message


# ============================
# 3. PARSER – daata/
# ============================

DAATA_HEADER_REGEX = re.compile(
    r"\[(?P<date>\d{1,2}/\d{1,2}/\d{4})\s+"
    r"(?P<time>\d{1,2}:\d{2})\s*"
    r"(?P<ampm>AM|PM)\]\s+"
    r"(?P<author>.+)"
)

def parse_daata_file(path: Path):
    messages = []

    current_timestamp = None
    current_author = None
    current_message_lines = []

    def flush():
        nonlocal current_timestamp, current_author, current_message_lines

        if (
            current_timestamp is None
            or current_author is None
            or not current_message_lines
        ):
            return

        content = "\n".join(current_message_lines).strip()

        if (
            not content
            or content.startswith("{Attachments}")
            or content.startswith("{Embed}")
            or content.startswith("{Reactions}")
        ):
            return

        messages.append((current_timestamp, current_author, content))

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip()

            header_match = DAATA_HEADER_REGEX.match(line)
            if header_match:
                flush()

                date_str = header_match.group("date")
                time_str = header_match.group("time")
                ampm = header_match.group("ampm")
                author = header_match.group("author").strip()

                try:
                    timestamp = datetime.strptime(
                        f"{date_str} {time_str} {ampm}",
                        "%m/%d/%Y %I:%M %p"
                    )
                except ValueError:
                    current_timestamp = None
                    current_author = None
                    current_message_lines = []
                    continue

                current_timestamp = timestamp
                current_author = author
                current_message_lines = []
            else:
                if line and not line.startswith("{"):
                    current_message_lines.append(line)

        flush()

    return messages


# ============================
# 4. KONVERSATIONEN
# ============================

def build_conversations(parsed_messages):
    conversations = []
    current = {"messages": []}

    last_author = None
    last_time = None

    for timestamp, author, message in parsed_messages:
        if last_time and (timestamp - last_time).total_seconds() > 600:
            if len(current["messages"]) >= 2:
                conversations.append(current)
            current = {"messages": []}

        if not current["messages"]:
            role = "user"
        else:
            if author == last_author:
                role = current["messages"][-1]["role"]
            else:
                role = (
                    "assistant"
                    if current["messages"][-1]["role"] == "user"
                    else "user"
                )

        current["messages"].append({
            "role": role,
            "content": message
        })

        last_author = author
        last_time = timestamp

    if len(current["messages"]) >= 2:
        conversations.append(current)

    return conversations


# ============================
# 5. MAIN
# ============================

if __name__ == "__main__":
    print(f"📂 Skript-Ordner: {SCRIPT_DIR}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    all_messages = []

    # ---- logs/ ----
    if LOGS_DIR.exists():
        log_files = list(LOGS_DIR.glob("*.txt"))
        print(f"📄 logs/: {len(log_files)} Dateien")

        for path in log_files:
            print(f"   Lese {path.name}")
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parsed = parse_logs_line(line.strip())
                    if parsed:
                        all_messages.append(parsed)

    # ---- daata/ ----
    if DAATA_DIR.exists():
        daata_files = list(DAATA_DIR.glob("*.txt"))
        print(f"📄 daata/: {len(daata_files)} Dateien")

        for path in daata_files:
            print(f"   Lese {path.name}")
            all_messages.extend(parse_daata_file(path))

    if not all_messages:
        print("❌ Keine Nachrichten gefunden.")
        exit(1)

    # Sicherheitsfilter
    all_messages = [
        m for m in all_messages
        if isinstance(m[0], datetime)
    ]

    print(f"✅ Nachrichten gesamt: {len(all_messages)}")

    all_messages.sort(key=lambda x: x[0])

    print("🔨 Baue Konversationen...")
    conversations = build_conversations(all_messages)

    print(f"✅ Konversationen generiert: {len(conversations)}")
    if not conversations:
        exit(1)

    random.shuffle(conversations)

    split_idx = int(0.9 * len(conversations))
    train_data = conversations[:split_idx]
    val_data = conversations[split_idx:]

    print("💾 Speichere JSONL...")
    with open(OUTPUT_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for conv in train_data:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    with open(OUTPUT_DIR / "val.jsonl", "w", encoding="utf-8") as f:
        for conv in val_data:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"🎉 Fertig! Train: {len(train_data)} | Val: {len(val_data)}")
    print(f"📁 Daten liegen in: {OUTPUT_DIR}")
