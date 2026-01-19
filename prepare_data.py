# 01_prepare_data.py
import os
import json
import re
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================
# 1. KONFIGURATION & HYPERPARAMETER
# ============================
# WICHTIG: Pfade relativ zum Skript-Ordner
SCRIPT_DIR = Path(__file__).parent  # ← NEU: Ordner, in dem dieses Skript liegt
LOGS_DIR = SCRIPT_DIR / "logs"      # ← NEU: Kombiniere Pfade
OUTPUT_DIR = SCRIPT_DIR / "data"
CACHE_DIR = SCRIPT_DIR / "cache"
PPL_THRESHOLD = 50.0
MODEL_ID = "Qwen/Qwen3-0.6B"

# ============================
# 2. HELPER-FUNKTIONEN
# ============================

def parse_log_line(line: str):
    """Parst eine einzelne Zeile aus dem Discord-Log."""
    match = re.match(r"\[(?P<timestamp>.*?)\]\s+(?P<author>[^:]+):\s+(?P<message>.*)", line)
    if not match:
        return None
    
    timestamp_str = match.group("timestamp")
    author = match.group("author").strip()
    message = match.group("message").strip()
    
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
    except ValueError:
        return None
    
    if not message or message.startswith("[ATTACHMENT") or message.startswith("<@"):
        return None
    
    return timestamp, author, message


def build_conversations(parsed_messages):
    """Wandelt Nachrichten in Konversationen im ChatML Format um."""
    conversations = []
    current_conv = {"messages": []}
    last_author = None
    last_time = None
    
    for timestamp, author, message in parsed_messages:
        # Starte neue Konversation nach 10 Minuten Pause oder gleicher Author
        if (last_time and (timestamp - last_time).total_seconds() > 600) or \
           (last_author == author):
            if len(current_conv["messages"]) >= 2:
                conversations.append(current_conv)
            current_conv = {"messages": []}
        
        # Weise Rolle zu
        if not current_conv["messages"]:
            role = "user"
        else:
            last_role = current_conv["messages"][-1]["role"]
            role = "assistant" if last_role == "user" else "user"
        
        current_conv["messages"].append({
            "role": role,
            "content": message,
            "timestamp": timestamp.isoformat()
        })
        
        last_author = author
        last_time = timestamp
    
    # Füge letzte Konversation hinzu
    if len(current_conv["messages"]) >= 2:
        conversations.append(current_conv)
    
    return conversations


def filter_by_perplexity(conversations, tokenizer, model, threshold=50.0):
    """Filtert Konversationen basierend auf ihrer Perplexität."""
    filtered = []
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"🔄 Filtere {len(conversations)} Konversationen...")  # ← NEU: Progress Info
    
    with torch.no_grad():
        for i, conv in enumerate(conversations):
            # Progress-Log alle 1000 Konversationen
            if i % 1000 == 0 and i > 0:
                print(f"   {i}/{len(conversations)} verarbeitet...")
            
            text = ""
            for msg in conv["messages"]:
                text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            
            tokens = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
            tokens = tokens.to(device)
            
            outputs = model(tokens, labels=tokens)
            ppl = torch.exp(outputs.loss).item()
            
            if ppl > threshold:
                # Entferne Timestamps
                for msg in conv["messages"]:
                    if "timestamp" in msg:
                        del msg["timestamp"]
                filtered.append(conv)
    
    model.train()
    return filtered

# ============================
# 3. HAUPT-AUSFÜHRUNG
# ============================
if __name__ == "__main__":
    print("📂 Debug: Prüfe Pfade...")
    print(f"   Skript-Ordner: {SCRIPT_DIR}")
    print(f"   Logs-Ordner: {LOGS_DIR}")
    print(f"   Existiert Logs/? {LOGS_DIR.exists()}")  # ← NEU: Zeigt an, ob der Ordner gefunden wird
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Sammle alle TXT-Dateien
    log_files = list(LOGS_DIR.glob("*.txt"))
    print(f"📄 Gefunden: {len(log_files)} Log-Dateien")
    
    if len(log_files) == 0:
        print("❌ FEHLER: Keine TXT-Dateien gefunden!")
        print(f"   Bitte prüfe, ob '{LOGS_DIR}' existiert und .txt-Dateien enthält.")
        exit(1)  # ← NEU: Beende mit Fehlercode
    
    # Parse alle Nachrichten
    all_messages = []
    for file_path in log_files:
        print(f"   Lese {file_path.name}...")  # ← NEU: Zeigt nur Dateinamen
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = parse_log_line(line.strip())
                if parsed:
                    all_messages.append(parsed)
    
    print(f"✅ Nachrichten gesamt: {len(all_messages)}")  # ← NEU: Zeigt Gesamtzahl
    
    # Sortiere nach Zeit
    all_messages.sort(key=lambda x: x[0])
    
    # Baue Konversationen
    print("🔨 Baue Konversationen...")
    conversations = build_conversations(all_messages)
    print(f"✅ Konversationen vor Filter: {len(conversations)}")
    
    # Lade Modell mit FP8-Fallback
    print("🤖 Lade Modell für Perplexitätsfilter...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True)
    
    # ← NEU: FP8-Fallback Block
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float8_e4m3fn,
            device_map="auto",
            cache_dir=CACHE_DIR,
            trust_remote_code=True
        )
    except TypeError as e:
        if "Float8_e4m3fnStorage" in str(e):
            print("⚠️  FP8 nicht unterstützt, falle zurück auf float16...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,  # float16 ist sicher
                device_map="auto",
                cache_dir=CACHE_DIR,
                trust_remote_code=True
            )
        else:
            raise e
    
    # Filtere mit Progress-Anzeige
    filtered_conversations = filter_by_perplexity(conversations, tokenizer, model, PPL_THRESHOLD)
    print(f"✅ Konversationen nach Filter: {len(filtered_conversations)}")
    
    if len(filtered_conversations) == 0:
        print("⚠️  WARNUNG: Alle Konversationen wurden gefiltert!")
        print(f"   Senke PPL_THRESHOLD (aktuell: {PPL_THRESHOLD}) oder deaktiviere den Filter.")
        response = input("   Trotzdem fortfahren? (y/n): ")  # ← NEU: Interaktive Abfrage
        if response.lower() != 'y':
            exit(1)
    
    # Train/Val Split
    split_idx = int(0.9 * len(filtered_conversations))
    train_data = filtered_conversations[:split_idx]
    val_data = filtered_conversations[split_idx:]
    
    # Speichere
    with open(OUTPUT_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for conv in train_data:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    
    with open(OUTPUT_DIR / "val.jsonl", "w", encoding="utf-8") as f:
        for conv in val_data:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    
    print(f"🎉 Erfolg! Train: {len(train_data)} | Val: {len(val_data)}")
    print(f"📁 Gespeichert in: {OUTPUT_DIR}")