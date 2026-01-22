# src/train_kitten_simple.py
"""
Simple HOPE Trainer: Ohne Deep Optimizer, ohne Meta-Learning.
Nur die nackte Kraft der HOPE-LoRA Architektur.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import json
import time
import gc
import shutil
import sys

from kitten_model import HOPEModel
from kitten_lora import HOPEConfig

# ═════════════════════════════════════════════════════════
# 1. KONFIGURATION & STEUERUNG
# ═════════════════════════════════════════════════════════
smol = True
big_data = True

# --- NEU: PHASEN-STEUERUNG ---
# Phase 1: TRAIN_LORA = True, LOAD_CHECKPOINT_PATH = None
# Phase 2: TRAIN_LORA = False, LOAD_CHECKPOINT_PATH = ".../pfad/zu/best"

TRAIN_LORA = False  # True = Alles trainieren | False = Nur Memory (LoRA frozen)
LOAD_CHECKPOINT_PATH = "/home/henry/Documents/Kitten-LoRA/models/kitten_simple_smol2/best" # none wenn neu starten 
# Beispiel für Phase 2:
# LOAD_CHECKPOINT_PATH = "/home/henry/Documents/Kitten-LoRA/models/kitten_simple_smol2/best" 

NUM_EPOCHS = 3    # Wie viele Epochen soll DIESER Run laufen?
# -----------------------------

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data5" if big_data else SCRIPT_DIR.parent / "data"
OUTPUT_DIR = (
    SCRIPT_DIR.parent / "models" / "kitten_simple_big"
    if not smol
    else SCRIPT_DIR.parent / "models" / "kitten_simple_smol2_mem"
)
CACHE_DIR = SCRIPT_DIR.parent / "cache"

MODEL_ID = "Qwen/Qwen3-1.7B" if not smol else "Qwen/Qwen3-0.6B"

# Training Settings
MAX_SEQ_LEN = 512
BATCH_SIZE = 2 if smol else 1
LOG_INTERVAL = 10
SAVE_INTERVAL = 500
MAX_STEP_CHECKPOINTS = 2
GC_INTERVAL = 10
EMPTY_CACHE_INTERVAL = 100

# HOPE Config
HOPE_CONFIG = HOPEConfig(
    r_fast=8 if smol else 16,
    r_medium=32 if smol else 64,
    r_slow=64 if smol else 128,
    chunk_medium=16 if smol else 32,
    chunk_slow=64 if smol else 128,
    hidden_dim=64 if smol else 128,
    use_newton_schulz=False,
    memory_decay=0.9995,
    surprise_threshold=-1.0,
    lr_fast=0.2,
    lr_medium=0.05,
    lr_slow=0.01,
)

# ═════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════
def load_dataset_kitten(tokenizer):
    print("📂 Lade Dataset...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(DATA_DIR / "train.jsonl"),
            "validation": str(DATA_DIR / "val.jsonl"),
        }
    )

    def tokenize(batch):
        text = ""
        for msg in batch["messages"]:
            text += (
                f"<|im_start|>{msg.get('role','user')}\n"
                f"{msg.get('content','')}<|im_end|>\n"
            )

        out = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        out = {k: v.squeeze(0) for k, v in out.items()}
        out["labels"] = out["input_ids"].clone()
        out["labels"][out["labels"] == tokenizer.pad_token_id] = -100
        return out

    tokenized = dataset.map(
        tokenize,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )
    tokenized.set_format("torch")

    print(f"✅ {len(tokenized['train']):,} train / {len(tokenized['validation']):,} val")
    return tokenized


# ═════════════════════════════════════════════════════════
# Trainer
# ═════════════════════════════════════════════════════════
class SimpleHOPETrainer:
    def __init__(self, model, train_loader, val_loader, train_lora_weights=True): 
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = next(model.model.parameters()).device
        self.train_lora_weights = train_lora_weights

        memory_params = []
        lora_params = []

        for layer in model.hope_layers:
            # Memory (wird IMMER trainiert)
            memory_params += (
                list(layer.update_fast.parameters())
                + list(layer.update_medium.parameters())
                + list(layer.update_slow.parameters())
                + list(layer.surprise_net.parameters())
                + list(layer.gate_net.parameters())
            )
            
            # LoRA (Nur wenn Flag True)
            if self.train_lora_weights:
                lora_params += (
                    list(layer.proj_down.parameters())
                    + list(layer.proj_up.parameters())
                )
                # Sicherstellen, dass Grads aktiv sind
                layer.proj_down.weight.requires_grad = True
                layer.proj_up.weight.requires_grad = True
            else:
                # Einfrieren!
                layer.proj_down.weight.requires_grad = False
                layer.proj_up.weight.requires_grad = False

        self.memory_optimizer = torch.optim.AdamW(
            memory_params, lr=5e-5, weight_decay=0.01
        )
        
        if self.train_lora_weights:
            # Hier: 2e-4 basierend auf dem "LoRA Without Regret" Paper Empfehlung
            self.lora_optimizer = torch.optim.AdamW(lora_params, lr=2e-4, weight_decay=0.01)
        else:
            self.lora_optimizer = None

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": []}

        print("\n🐱 Simple HOPE Trainer Initialized")
        print(f"   Mode: {'FULL TRAINING (LoRA + Memory)' if train_lora_weights else 'MEMORY ONLY (LoRA Frozen)'}")
        print(f"   Device: {self.device}")
        print(f"   Memory Params (Trainable): {sum(p.numel() for p in memory_params):,}")
        if self.train_lora_weights:
            print(f"   LoRA Params   (Trainable): {sum(p.numel() for p in lora_params):,}")
        else:
            # Zähle trotzdem wie viele parameter eingefroren sind
            frozen_count = sum(p.numel() for layer in model.hope_layers for p in [layer.proj_down.weight, layer.proj_up.weight])
            print(f"   LoRA Params   (FROZEN):    {frozen_count:,}")

    # ─────────────────────────────────────────────
    def _rotate_step_checkpoints(self):
        step_dirs = sorted(
            [p for p in OUTPUT_DIR.glob("step_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
        )

        while len(step_dirs) > MAX_STEP_CHECKPOINTS:
            old = step_dirs.pop(0)
            shutil.rmtree(old, ignore_errors=True)
            print(f"🗑️ Entfernt alten Checkpoint: {old.name}")

    # ─────────────────────────────────────────────
    def _save_checkpoint(self, name: str):
        path = OUTPUT_DIR / name
        path.mkdir(parents=True, exist_ok=True)

        self.model.save_hope_weights(str(path))
        
        # Optimizer State speichern (nur wenn vorhanden)
        opt_state = {
            "memory_opt": self.memory_optimizer.state_dict(),
        }
        if self.lora_optimizer:
            opt_state["lora_opt"] = self.lora_optimizer.state_dict()
            
        torch.save(opt_state, path / "optimizers.pt")

        with open(path / "training_state.json", "w") as f:
            json.dump(
                {
                    "global_step": self.global_step,
                    "best_val_loss": self.best_val_loss,
                    "history": self.history,
                },
                f,
                indent=2,
            )

        print(f"   💾 Saved: {path}")

        if name.startswith("step_"):
            self._rotate_step_checkpoints()

    # ─────────────────────────────────────────────
    def train_epoch(self, epoch):
        self.model.model.train()
        total_loss = 0
        num_batches = 0

        if epoch == 0:
            self.model.reset_memory(BATCH_SIZE)

        progress = tqdm(
            self.train_loader, desc=f"🐱 Epoch {epoch+1}/{NUM_EPOCHS}"
        )

        for step, batch in enumerate(progress):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 1. Zero Grads
            self.memory_optimizer.zero_grad()
            if self.train_lora_weights:
                self.lora_optimizer.zero_grad()

            # 2. Forward
            outputs = self.model.model(**batch)
            loss = outputs.loss
            
            # 3. Backward
            loss.backward()

            # 4. Clip & Step
            params_to_clip = list(self.memory_optimizer.param_groups[0]["params"])
            if self.train_lora_weights:
                params_to_clip += list(self.lora_optimizer.param_groups[0]["params"])
            
            torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)

            self.memory_optimizer.step()
            if self.train_lora_weights:
                self.lora_optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            if step % LOG_INTERVAL == 0:
                mem = self.model.get_memory_stats()
                progress.set_postfix(
                    {
                        "loss": f"{total_loss/num_batches:.4f}",
                        "F": f"{mem['fast_norm_avg']:.2f}",
                        "M": f"{mem['medium_norm_avg']:.2f}",
                    }
                )

            if step % GC_INTERVAL == 0:
                gc.collect()
            if step % EMPTY_CACHE_INTERVAL == 0:
                torch.cuda.empty_cache()

            if self.global_step % SAVE_INTERVAL == 0:
                self._save_checkpoint(f"step_{self.global_step}")

        return total_loss / num_batches

    # ─────────────────────────────────────────────
    @torch.no_grad()
    def validate(self):
        self.model.model.eval()
        total_loss = 0
        num_batches = 0

        self.model.reset_memory(BATCH_SIZE)

        for batch in tqdm(self.val_loader, desc="📋 Validation", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model.model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1

        gc.collect()
        torch.cuda.empty_cache()
        return total_loss / num_batches

    # ─────────────────────────────────────────────
    def train(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        for epoch in range(NUM_EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            print(f"\n📊 Epoch {epoch+1}")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best")
                print("   ✅ New Best Model")

        self._save_checkpoint("final")


# ═════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("🐱 SIMPLE HOPE PIPELINE")
    print("=" * 60)

    # 1. Modell initialisieren
    model = HOPEModel(
        model_id=MODEL_ID,
        config=HOPE_CONFIG,
        cache_dir=str(CACHE_DIR),
    )

    # 2. CHECKPOINT LADEN (Falls gewünscht)
    if LOAD_CHECKPOINT_PATH:
        print(f"\n🔄 Setze Training fort von: {LOAD_CHECKPOINT_PATH}")
        path = Path(LOAD_CHECKPOINT_PATH)
        if not path.exists():
            print("❌ Fehler: Checkpoint Pfad existiert nicht!")
            sys.exit(1)
            
        success = model.load_hope_weights(str(path))
        if success:
            print("✅ Gewichte erfolgreich geladen.")
        else:
            print("❌ Fehler beim Laden der Gewichte.")
            sys.exit(1)
    else:
        print("\n✨ Starte neues Training von Scratch.")

    # 3. Daten laden
    tokenized = load_dataset_kitten(model.tokenizer)
    train_loader = DataLoader(
        tokenized["train"], batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        tokenized["validation"], batch_size=BATCH_SIZE
    )

    # 4. Trainer starten (Der entscheidet basierend auf TRAIN_LORA was eingefroren wird)
    trainer = SimpleHOPETrainer(model, train_loader, val_loader, train_lora_weights=TRAIN_LORA)
    trainer.train()


if __name__ == "__main__":
    main()