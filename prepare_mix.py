import json
import re
import random
from datetime import datetime
from pathlib import Path

# ======================================================
# 1. KONFIGURATION
# ======================================================
SCRIPT_DIR = Path(__file__).parent

LOGS_DIR = SCRIPT_DIR / "logs"
DAATA_DIR = SCRIPT_DIR / "daata"
PERSONA_DIR = SCRIPT_DIR / "persona"

OUTPUT_DIR = SCRIPT_DIR / "data3"

MAX_GAP_SECONDS = 600
TRAIN_SPLIT = 0.9
PERSONA_OVERSAMPLE = 20   # ggf. 10–20

# ======================================================
# 2. LINK / GIF FILTER (HART)
# ======================================================
URL_REGEX = re.compile(r"https?://\S+", re.IGNORECASE)

def clean_text(text: str) -> str:
    """
    Entfernt ALLE URLs (Tenor, Discord CDN, alles).
    Gibt bereinigten Text zurück oder "".
    """
    text = URL_REGEX.sub("", text)

    # Whitespaces normalisieren
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

# ======================================================
# 3. PARSER – logs/
# ======================================================
LOGS_REGEX = re.compile(
    r"\[(?P<timestamp>[\d\-T:\.]+)\]\s+(?P<author>[^:]+):\s*(?P<message>.*)"
)

def parse_logs_line(line: str):
    m = LOGS_REGEX.match(line)
    if not m:
        return None

    try:
        ts = datetime.fromisoformat(m.group("timestamp"))
    except ValueError:
        return None

    msg = clean_text(m.group("message").strip())
    if not msg:
        return None

    if msg.startswith("[ATTACHMENT") or msg.startswith("<@"):
        return None

    return ts, m.group("author").strip(), msg

# ======================================================
# 4. PARSER – daata/
# ======================================================
DAATA_HEADER_REGEX = re.compile(
    r"\[(?P<date>\d{1,2}/\d{1,2}/\d{4})\s+"
    r"(?P<time>\d{1,2}:\d{2})\s*"
    r"(?P<ampm>AM|PM)\]\s+"
    r"(?P<author>.+)"
)

def parse_daata_file(path: Path):
    messages = []

    current_ts = None
    current_author = None
    buffer = []

    def flush():
        nonlocal current_ts, current_author, buffer
        if not current_ts or not current_author or not buffer:
            return

        content = clean_text("\n".join(buffer).strip())
        if not content:
            return

        messages.append((current_ts, current_author, content))

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip()

            m = DAATA_HEADER_REGEX.match(line)
            if m:
                flush()
                try:
                    current_ts = datetime.strptime(
                        f"{m.group('date')} {m.group('time')} {m.group('ampm')}",
                        "%m/%d/%Y %I:%M %p"
                    )
                except ValueError:
                    current_ts = None
                    current_author = None
                    buffer = []
                    continue

                current_author = m.group("author").strip()
                buffer = []
            else:
                if line and not line.startswith("{"):
                    buffer.append(line)

        flush()

    return messages

# ======================================================
# 5. KONVERSATIONEN BAUEN
# ======================================================
def build_conversations(messages):
    conversations = []
    current = {"messages": []}

    last_author = None
    last_time = None

    for ts, author, msg in messages:
        if last_time and (ts - last_time).total_seconds() > MAX_GAP_SECONDS:
            if len(current["messages"]) >= 2:
                conversations.append(current)
            current = {"messages": []}

        if not current["messages"]:
            role = "user"
        else:
            if author == last_author:
                role = current["messages"][-1]["role"]
            else:
                role = "assistant" if current["messages"][-1]["role"] == "user" else "user"

        current["messages"].append({
            "role": role,
            "content": msg
        })

        last_author = author
        last_time = ts

    if len(current["messages"]) >= 2:
        conversations.append(current)

    return conversations

# ======================================================
# 6. PERSONA LADEN
# ======================================================
def load_persona():
    persona = []

    if not PERSONA_DIR.exists():
        print("ℹ️ Kein persona/-Ordner gefunden.")
        return persona

    for file in PERSONA_DIR.glob("*.jsonl"):
        print(f"🎭 Lade Persona-Datei: {file.name}")
        with open(file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()

                # 1. Leere Zeilen ignorieren
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(
                        f"⚠️ Persona-JSON übersprungen "
                        f"({file.name}:{line_num}): {e}"
                    )
                    continue

                # 2. Minimal-Validierung
                if not isinstance(obj, dict):
                    continue
                if "messages" not in obj:
                    continue
                if not isinstance(obj["messages"], list):
                    continue
                if len(obj["messages"]) < 2:
                    continue

                persona.append(obj)

    return persona


# ======================================================
# 7. MAIN
# ======================================================
if __name__ == "__main__":
    print(f"📂 Skript-Ordner: {SCRIPT_DIR}")
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_messages = []

    # ---- logs/ ----
    if LOGS_DIR.exists():
        files = list(LOGS_DIR.glob("*.txt"))
        print(f"📄 logs/: {len(files)} Dateien")
        for p in files:
            print(f"   Lese {p.name}")
            with open(p, encoding="utf-8") as f:
                for line in f:
                    parsed = parse_logs_line(line.strip())
                    if parsed:
                        all_messages.append(parsed)

    # ---- daata/ ----
    if DAATA_DIR.exists():
        files = list(DAATA_DIR.glob("*.txt"))
        print(f"📄 daata/: {len(files)} Dateien")
        for p in files:
            print(f"   Lese {p.name}")
            all_messages.extend(parse_daata_file(p))

    all_messages = [m for m in all_messages if isinstance(m[0], datetime)]
    print(f"✅ Nachrichten gesamt (bereinigt): {len(all_messages)}")

    all_messages.sort(key=lambda x: x[0])

    print("🔨 Baue Discord-Konversationen...")
    discord_convs = build_conversations(all_messages)
    print(f"   → {len(discord_convs)}")

    # ---- Persona ----
    persona_convs = load_persona()
    print(f"🎭 Persona-Convs: {len(persona_convs)} (x{PERSONA_OVERSAMPLE})")

    persona_weighted = persona_convs * PERSONA_OVERSAMPLE

    # ---- Merge ----
    all_convs = discord_convs + persona_weighted
    #all_convs =  persona_weighted
    random.shuffle(all_convs)

    split = int(len(all_convs) * TRAIN_SPLIT)
    train = all_convs[:split]
    val = all_convs[split:]

    # ---- Save ----
    with open(OUTPUT_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for c in train:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    with open(OUTPUT_DIR / "val.jsonl", "w", encoding="utf-8") as f:
        for c in val:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print("\n🎉 FERTIG")
    print(f"   Train: {len(train)}")
    print(f"   Val:   {len(val)}")
    print(f"   Persona Oversample: x{PERSONA_OVERSAMPLE}")
    print(f"📁 Output: {OUTPUT_DIR}")
