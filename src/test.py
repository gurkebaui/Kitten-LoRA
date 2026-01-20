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


smol: bool = False  # Setze auf True für das kleine Modell (0.6B), False für 1.7B
# ═══════════════════════════════════════════════════════════
# Pfade
# ═══════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).parent.parent
MEMORY_DIR = SCRIPT_DIR / "memory_states" if smol else SCRIPT_DIR / "memory_states_big"
MODELS_DIR = SCRIPT_DIR / "models"
CACHE_DIR = SCRIPT_DIR / "cache"

MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def find_best_weights() -> Path:
    """Findet die besten verfügbaren Gewichte."""
    candidates = [
        MODELS_DIR / "kitten_simple" / "best" if smol else MODELS_DIR / "kitten_simple_big" / "best",
        MODELS_DIR / "kitten_full" / "best",
        MODELS_DIR / "kitten_simple" / "final",
    ]
    
    for path in candidates:
        if (path / "hope_lora.pt").exists():
            return path
    return None


def list_memory_files():
    """Listet alle Memory-Dateien auf."""
    files = sorted(MEMORY_DIR.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not files:
        print("  (keine Memory-Dateien gefunden)")
        return []
    
    for i, f in enumerate(files):
        size = f.stat().st_size / 1024  # KB
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  [{i}] {f.name} ({size:.1f} KB, {mtime})")
    
    return files


def save_memory(model: HOPEModel, name: str = None):
    """Speichert den Memory State."""
    if name is None:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Dateiname bereinigen
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


def main():
    print("="*55)
    print("          🐱 HOPE CHAT MIT MEMORY")
    print("="*55)
    
    # Modell laden
    print("\n🔧 Lade Modell...")
    
    config = HOPEConfig(
        r_fast=8 if smol else 16,
        r_medium=32 if smol else 64,
        r_slow=64 if smol else 128,  # AUF 64 GEÄNDERT!
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
    
    # Gewichte laden
    weights_dir = find_best_weights()
    if weights_dir:
        print(f"📂 Gewichte: {weights_dir}")
        model.load_hope_weights(str(weights_dir))
    else:
        print("⚠️ Keine trainierten Gewichte gefunden (Base Model)")
    
    model.reset_memory(1)
    model.model.eval()
    
    print("\n✅ Modell bereit!")
    show_help()
    
    # Chat Loop
    while True:
        try:
            user_input = input("Du: ").strip()
            
            if not user_input:
                continue
            
            # ─────────────────────────────────────────────
            # Commands
            # ─────────────────────────────────────────────
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else None
                
                if cmd == "/quit" or cmd == "/exit":
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
                            idx = input("\nWelche laden? (Nummer oder 'c' zum Abbrechen): ").strip()
                            if idx.lower() != 'c':
                                load_memory(model, files[int(idx)])
                        except (ValueError, IndexError):
                            print("❌ Ungültige Auswahl")
                    print()
                
                elif cmd == "/reset":
                    confirm = input("⚠️ Memory wirklich zurücksetzen? (ja/nein): ").strip()
                    if confirm.lower() == "ja":
                        model.reset_memory(1)
                        print("🔄 Memory zurückgesetzt.")
                
                else:
                    print(f"❓ Unbekannter Command: {cmd}")
                    print("   Tippe /help für alle Commands")
                
                continue
            
            # ─────────────────────────────────────────────
            # Chat
            # ─────────────────────────────────────────────
            response = model.generate(
                prompt=user_input,
                max_new_tokens=256,
                temperature=0.7,
                reset_memory=False,
            )
            
            print(f"\n🐱: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Unterbrochen. Tschüss!")
            break
        except Exception as e:
            print(f"⚠️ Fehler: {e}")
            continue


if __name__ == "__main__":
    main()