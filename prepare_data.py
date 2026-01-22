import json
import re

# Read the entire content
with open('kitten.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# Split by lines that start with "{" and end with "}"
# This handles multi-line objects correctly
objects = []
current = ""
in_object = False

for line in content.splitlines():
    line = line.strip()
    if not line:
        continue
    if line.startswith('{') and not in_object:
        current = line
        in_object = True
    elif line.endswith('}') and in_object:
        current += " " + line
        try:
            obj = json.loads(current)
            objects.append(obj)
            current = ""
            in_object = False
        except json.JSONDecodeError:
            print(f"⚠️ Failed to parse: {current[:100]}...")
            current = ""
            in_object = False
    elif in_object:
        current += " " + line

# Write fixed JSONL
with open('kitten_fixed.jsonl', 'w', encoding='utf-8') as f:
    for obj in objects:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')

print(f"✅ Fixed {len(objects)} conversations. Saved to kitten_fixed.jsonl")