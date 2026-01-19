# 01_prepare_data.py
import json
import re
from datetime import datetime
from pathlib import Path
import random

# ============================
# 1. KONFIGURATION
# ============================
SCRIPT_DIR = Path(__file__).parent
LOGS_DIR = SCRIPT_DIR / "logs"
OUTPUT_DIR = SCRIPT_DIR / "data"

# ============================
# 2. HELPER-FUNKTIONEN
# ============================

def parse_log_line(line: str):
    """Parst eine einzelne Zeile aus dem Discord-Log."""
    # Versucht, Timestamp, Author und Message zu extrahieren
    match = re.match(r"\[(?P<timestamp>.*?)\]\s+(?P<author>[^:]+):\s+(?P<message>.*)", line)
    if not match:
        return None
    
    timestamp_str = match.group("timestamp")
    author = match.group("author").strip()
    message = match.group("message").strip()
    
    try:
        # Passt auf Formate wie "2024-01-20 15:30:00"
        timestamp = datetime.fromisoformat(timestamp_str)
    except ValueError:
        return None
    
    # Filter: Ignoriere Anhänge, leere Nachrichten oder Mentions am Anfang
    if not message or message.startswith("[ATTACHMENT") or message.startswith("<@"):
        return None
    
    return timestamp, author, message


def build_conversations(parsed_messages):
    """Wandelt Nachrichten in Konversationen um."""
    conversations = []
    current_conv = {"messages": []}
    last_author = None
    last_time = None
    
    for timestamp, author, message in parsed_messages:
        # Neue Konversation, wenn > 10 Minuten (600s) Pause war
        if (last_time and (timestamp - last_time).total_seconds() > 600):
            if len(current_conv["messages"]) >= 2:
                conversations.append(current_conv)
            current_conv = {"messages": []}
        
        # Rolle bestimmen (User vs Assistant)
        # Wir nehmen einfach an: Wechselnder Autor = Wechselnde Rolle
        if not current_conv["messages"]:
            role = "user" # Erste Nachricht immer User
        else:
            # Wenn der Autor der gleiche ist wie vorher, bleibt die Rolle gleich
            if author == last_author:
                role = current_conv["messages"][-1]["role"]
            else:
                # Sonst wechsel die Rolle
                role = "assistant" if current_conv["messages"][-1]["role"] == "user" else "user"
        
        current_conv["messages"].append({
            "role": role,
            "content": message
            # Timestamp entfernen wir hier direkt, brauchen wir im JSONL nicht
        })
        
        last_author = author
        last_time = timestamp
    
    # Letzte Konversation hinzufügen
    if len(current_conv["messages"]) >= 2:
        conversations.append(current_conv)
    
    return conversations

# ============================
# 3. MAIN
# ============================
if __name__ == "__main__":
    print(f"📂 Skript-Ordner: {SCRIPT_DIR}")
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    if not LOGS_DIR.exists():
        print(f"❌ Ordner nicht gefunden: {LOGS_DIR}")
        print("   Bitte erstelle den Ordner 'logs' und lege deine .txt Dateien dort ab.")
        exit(1)

    log_files = list(LOGS_DIR.glob("*.txt"))
    print(f"📄 Gefunden: {len(log_files)} Log-Dateien")
    
    all_messages = []
    for file_path in log_files:
        print(f"   Lese {file_path.name}...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = parse_log_line(line.strip())
                if parsed:
                    all_messages.append(parsed)
    
    print(f"✅ Nachrichten gesamt: {len(all_messages)}")
    
    # Sortiere chronologisch
    all_messages.sort(key=lambda x: x[0])
    
    # Baue Konversationen
    print("🔨 Baue Konversationen...")
    conversations = build_conversations(all_messages)
    print(f"✅ Konversationen generiert: {len(conversations)}")
    
    if len(conversations) == 0:
        print("❌ Keine Konversationen generiert! Prüfe das Datumsformat in den Logs.")
        exit(1)

    # Shuffle für bessere Verteilung (optional)
    random.shuffle(conversations)

    # Train/Val Split (90/10)
    split_idx = int(0.9 * len(conversations))
    train_data = conversations[:split_idx]
    val_data = conversations[split_idx:]
    
    # Speichern
    print("💾 Speichere JSONL Dateien...")
    with open(OUTPUT_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for conv in train_data:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    
    with open(OUTPUT_DIR / "val.jsonl", "w", encoding="utf-8") as f:
        for conv in val_data:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    
    print(f"🎉 Fertig! Train: {len(train_data)} | Val: {len(val_data)}")
    print(f"📁 Daten liegen in: {OUTPUT_DIR}")