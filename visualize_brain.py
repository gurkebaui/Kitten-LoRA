# src/visualize_brain.py
"""
Interaktive CLI um zu beobachten, wie das Memory (F/M/S) reagiert.
"""

import torch
from kitten_model import HOPEModel
from kitten_lora import HOPEConfig
from pathlib import Path

def draw_memory_bar(label: str, value: float, max_val: float = 3.0):
    """Zeigt eine Lade-Balken an."""
    pct = min(value / max_val, 1.0)
    bar_length = 20
    filled = int(bar_length * pct)
    empty = bar_length - filled
    return f"{label:8} |{'█' * filled}{'░' * empty}| {value:.2f}"

def main():
    print("\n" + "="*60)
    print("    🐱 HOPE BRAIN MONITOR V1.0")
    print("="*60)
    
    # 1. Modell laden
    CACHE_DIR = Path(__file__).parent.parent / "cache"

    model = HOPEModel(
        model_id="Qwen/Qwen3-0.6B",
        config=HOPEConfig(),
        cache_dir=str(CACHE_DIR),
    )
    
    # Bestes Modell laden (von Simple Trainer)
    checkpoint_path = Path("/home/henry/Documents/Kitten-LoRA/models/kitten_simple/best")
    
    if checkpoint_path.exists():
        if model.load_hope_weights(str(checkpoint_path)):
            print("✅ Gewichte geladen (Simple HOPE).")
        else:
            print("❌ Fehler beim Laden der Gewichte.")
            return
    else:
        print(f"❌ Kein Checkpoint gefunden: {checkpoint_path}")
        return
    
    model.model.eval()
    device = next(model.model.parameters()).device
    
    print("\n" + "="*60)
    print("    📝 MODELL: START (Memory ist leer!)")
    print("    TIP: Sag 'reset' um das Memory zu löschen.")
    print("="*60 + "\n")
    
    # Main Loop
    while True:
        # Input abwarten
        try:
            user_input = input("Du: ")
        except KeyboardInterrupt:
            print("\n👋 Beendet.")
            break
        
        user_input = user_input.strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == "/quit":
            break
        
        if user_input.lower() == "/reset":
            model.reset_memory(1)
            print("🧹 Memory gelöscht.")
            continue
        
        if user_input.lower() == "/stats":
            stats = model.get_memory_stats()
            print("\n" + "-"*40)
            print(draw_memory_bar("Fast", stats['fast_norm_avg'], max_val=1.0))
            print(draw_memory_bar("Medium", stats['medium_norm_avg'], max_val=3.0))
            print(draw_memory_bar("Slow", stats['slow_norm_avg'], max_val=1.0))
            print("-"*40 + "\n")
            continue

        # Prompt formatieren (für Chat-Style)
        # Wir benutzen das Template direkt
        messages = [{"role": "user", "content": user_input}]
        text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generieren
        inputs = model.tokenizer(text, return_tensors="pt").to(device)
        
        # WICHTIG: reset_memory=False!
        # Wir wollen sehen, wie das Memory sich anhäuft.
        with torch.no_grad():
            outputs = model.model.generate(
                **inputs,
                max_new_tokens=100,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        response = model.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Stats holen
        stats = model.get_memory_stats()
        
        # Ausgabe
        print(f"Bot: {response}")
        
        print("-" * 40)
        print(draw_memory_bar("Fast", stats['fast_norm_avg'], max_val=1.0))
        print(draw_memory_bar("Medium", stats['medium_norm_avg'], max_val=3.0))
        print(draw_memory_bar("Slow", stats['slow_norm_avg'], max_val=1.0))
        print("-" * 40 + "\n")

if __name__ == "__main__":
    main()