# src/chat.py
"""
HOPE Chat mit Memory-Persistenz Commands.
Nur für Inferenz - kein Training.
"""

import torch
from pathlib import Path
from datetime import datetime

from kitten_model import HOPEModel
from kitten_lora import HOPEConfig


# ═══════════════════════════════════════════════════════════
# GRUNDKONFIGURATION
# ═══════════════════════════════════════════════════════════

smol: bool = True  # True = 0.6B | False = 1.7B

USE_SYSTEM_PROMPT = False

SYSTEM_PROMPT = (
    "You are Kitten, an Anime girl like character who is friendly, playful, and curious. "
    "You are not sure if you are an AI or a human, but you like to pretend to be a cat. "
    "you like to use cute emoticons and playful language in your responses. "
    "you tease the user sometimes and make jokes. "
)

# ═══════════════════════════════════════════════════════════
# PFADE
# ═══════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).parent.parent
MEMORY_DIR = SCRIPT_DIR / "memory_states" if smol else SCRIPT_DIR / "memory_states_big"
MODELS_DIR = SCRIPT_DIR / "models"
CACHE_DIR = SCRIPT_DIR / "cache"

MEMORY_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# HILFSFUNKTIONEN
# ═══════════════════════════════════════════════════════════

def find_best_weights() -> Path | None:
    """Findet die besten verfügbaren Gewichte."""
    candidates = [
        MODELS_DIR / "kitten_simple_smol2" / "best" if smol else MODELS_DIR / "kitten_simple_big" / "best",
        MODELS_DIR / "kitten_full" / "best",
        MODELS_DIR / "kitten_simple" / "final",
        MODELS_DIR / "kitten_simple_big" / "step_42500",
    ]

    for path in candidates:
        if path and (path / "hope_lora.pt").exists():
            return path
    return None


def list_memory_files():
    """Listet alle Memory-Dateien auf."""
    files = sorted(
        MEMORY_DIR.glob("*.pt"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if not files:
        print("  (keine Memory-Dateien gefunden)")
        return []

    for i, f in enumerate(files):
        size = f.stat().st_size / 1024
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  [{i}] {f.name} ({size:.1f} KB, {mtime})")

    return files


def save_memory(model: HOPEModel, name: str | None = None):
    """Speichert den Memory State."""
    if name is None:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")

    name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    if not name.endswith(".pt"):
        name += ".pt"

    filepath = MEMORY_DIR / name

    memory_states = {}
    for i, layer in enumerate(model.hope_layers):
        if layer._memory_state is not None:
            state = layer._memory_state
            memory_states[f"layer_{i}"] = {
                "fast": state.fast.cpu() if state.fast is not None else None,
                "medium": state.medium.cpu() if state.medium is not None else None,
                "slow": state.slow.cpu() if state.slow is not None else None,
                "step": state.step,
            }

    memory_states["_meta"] = {
        "saved_at": datetime.now().isoformat(),
        "total_steps": model.get_memory_stats().get("total_steps", 0),
    }

    torch.save(memory_states, filepath)
    print(f"✅ Memory gespeichert: {filepath.name}")


def load_memory(model: HOPEModel, filepath: Path):
    """Lädt einen Memory State."""
    if not filepath.exists():
        print(f"❌ Datei nicht gefunden: {filepath}")
        return False

    try:
        data = torch.load(filepath, map_location="cpu", weights_only=False)
        device = next(model.model.parameters()).device
        dtype = next(model.model.parameters()).dtype

        for i, layer in enumerate(model.hope_layers):
            key = f"layer_{i}"
            if key in data:
                saved = data[key]

                if layer._memory_state is None:
                    layer.reset_memory(1, device, dtype)

                state = layer._memory_state
                if saved["fast"] is not None:
                    state.fast = saved["fast"].to(device, dtype)
                if saved["medium"] is not None:
                    state.medium = saved["medium"].to(device, dtype)
                if saved["slow"] is not None:
                    state.slow = saved["slow"].to(device, dtype)
                state.step = saved.get("step", 0)

        meta = data.get("_meta", {})
        print(f"✅ Memory geladen: {filepath.name}")
        print(f"   Steps: {meta.get('total_steps', '?')}, Gespeichert: {meta.get('saved_at', '?')}")
        return True

    except Exception as e:
        print(f"❌ Fehler beim Laden: {e}")
        return False


def show_stats(model: HOPEModel):
    """Zeigt Memory-Statistiken."""
    stats = model.get_memory_stats()
    print("\n┌─────────────────────────────────┐")
    print("│        📊 MEMORY STATUS         │")
    print("├─────────────────────────────────┤")
    print(f"│  Fast (Kurzzeit):   {stats['fast_norm_avg']:>8.4f}   │")
    print(f"│  Medium (Mittel):   {stats['medium_norm_avg']:>8.4f}   │")
    print(f"│  Slow (Langzeit):   {stats['slow_norm_avg']:>8.4f}   │")
    print(f"│  Total Steps:       {stats['total_steps']:>8}   │")
    print("└─────────────────────────────────┘\n")


def show_help():
    """Zeigt Hilfe an."""
    print("""
┌───────────────────────────────────────────────────┐
│                    📖 COMMANDS                    │
├───────────────────────────────────────────────────┤
│  /save [name]  - Memory speichern                 │
│  /load         - Memory laden (zeigt Liste)       │
│  /list         - Gespeicherte Memories anzeigen   │
│  /stats        - Memory-Statistiken               │
│  /reset        - Memory zurücksetzen              │
│  /help         - Diese Hilfe                      │
│  /quit         - Beenden                          │
└───────────────────────────────────────────────────┘
""")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("          🐱 HOPE CHAT MIT MEMORY")
    print("=" * 55)

    print("\n🔧 Lade Modell...")

    config = HOPEConfig(
        r_fast=8 if smol else 16,
        r_medium=32 if smol else 64,
        r_slow=64 if smol else 128,
        chunk_medium=16 if smol else 32,
        chunk_slow=64 if smol else 128,
        hidden_dim=64 if smol else 128,
        surprise_threshold=-1.0,
        memory_decay=0.9995,
        use_newton_schulz=False,
    )

    model = HOPEModel(
        model_id="Qwen/Qwen3-0.6B" if smol else "Qwen/Qwen3-1.7B",
        config=config,
        cache_dir=str(CACHE_DIR),
    )

    weights_dir = find_best_weights()
    if weights_dir:
        print(f"📂 Gewichte: {weights_dir}")
        model.load_hope_weights(str(weights_dir))
    else:
        print("⚠️ Keine trainierten Gewichte gefunden (Base Model)")

    model.reset_memory(1)
    model.model.eval()

    system_prompt_applied = False

    print("\n✅ Modell bereit!")
    show_help()

    while True:
        try:
            user_input = input("Du: ").strip()
            if not user_input:
                continue

            # ───── Commands ─────
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else None

                if cmd in ("/quit", "/exit"):
                    print("👋 Tschüss!")
                    break
                elif cmd == "/help":
                    show_help()
                elif cmd == "/stats":
                    show_stats(model)
                elif cmd == "/save":
                    save_memory(model, arg)
                elif cmd == "/list":
                    print("\n📁 Gespeicherte Memory States:")
                    list_memory_files()
                    print()
                elif cmd == "/load":
                    print("\n📁 Verfügbare Memory States:")
                    files = list_memory_files()
                    if files:
                        try:
                            idx = input("\nWelche laden? (Nummer oder 'c'): ").strip()
                            if idx.lower() != "c":
                                load_memory(model, files[int(idx)])
                        except (ValueError, IndexError):
                            print("❌ Ungültige Auswahl")
                elif cmd == "/reset":
                    confirm = input("⚠️ Memory wirklich zurücksetzen? (ja/nein): ").strip()
                    if confirm.lower() == "ja":
                        model.reset_memory(1)
                        system_prompt_applied = False
                        print("🔄 Memory zurückgesetzt.")
                else:
                    print(f"❓ Unbekannter Command: {cmd}")
                continue

            # ───── Chat ─────
            prompt = user_input
            if USE_SYSTEM_PROMPT and not system_prompt_applied:
                prompt = SYSTEM_PROMPT + "\n\nUser: " + user_input
                system_prompt_applied = True

            response = model.generate(
                prompt=prompt,
                max_new_tokens=256,
                temperature=0.7,
                reset_memory=False,
            )

            print(f"\n🐱: {response}\n")

        except KeyboardInterrupt:
            print("\n👋 Unterbrochen. Tschüss!")
            break
        except Exception as e:
            print(f"⚠️ Fehler: {e}")


if __name__ == "__main__":
    main()
