# test.py
from pathlib import Path
from kitten_model import HOPEModel
from kitten_lora import HOPEConfig

CACHE_DIR = Path(__file__).parent.parent / "cache"

model = HOPEModel(
    model_id="Qwen/Qwen3-0.6B",
    config=HOPEConfig(),
    cache_dir=str(CACHE_DIR),
)

# Checkpoint laden
model.load_hope_weights("../models/kitten_full/best")
model.reset_memory(1)

print("\n🧪 Memory Test (Mit korrektem Format)")

# Richtiges Format wie im Training (<|im_start|>user\n...<|im_end|>\n)
setup_prompt = "<|im_start|>user\nMy name is Kitten and I love neural networks.<|im_end|>\n<|im_start|>assistant\n"
_ = model.generate(setup_prompt, max_new_tokens=30, reset_memory=False)

question_prompt = "<|im_start|>user\nWhat is my name?<|im_end|>\n<|im_start|>assistant\n"
response = model.generate(question_prompt, max_new_tokens=50, temperature=1.7, reset_memory=False)

print(f"Frage: What is my name?")
print(f"Antwort: {response[:100]}")

if "kitten" in response.lower():
    print("✅ PASSED")
else:
    print("❌ FAILED (Modell weiß es nicht)")