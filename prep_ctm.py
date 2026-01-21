import json
from collections import Counter
from pathlib import Path

# ============================
# KONFIG
# ============================

CHUNK_SIZE = 100
OUTPUT_DIR = Path("data_chunks")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"

# ============================
# CHUNKING
# ============================

def chunk_messages(messages, size):
    for i in range(0, len(messages), size):
        yield messages[i:i + size]

# ============================
# MAIN
# ============================

print("🔨 Baue Chunks & schreib JSONL...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for idx, chunk in enumerate(chunk_messages(all_messages, CHUNK_SIZE), start=1):

        if not chunk:
            continue

        authors = [author for _, author, _ in chunk]
        counter = Counter(authors)
        active_user = counter.most_common(1)[0][0]

        messages = []
        for _, author, message in chunk:
            messages.append({
                "username": author,
                "content": message
            })

        json_obj = {
            "chunk_id": idx,
            "active_user": active_user,
            "messages": messages
        }

        f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

print(f"💪 Fertig. JSONL liegt da: {OUTPUT_FILE}")
