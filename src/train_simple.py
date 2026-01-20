# src/train_kitten_simple.py
"""
Simple HOPE Trainer: Ohne Deep Optimizer, ohne Meta-Learning.
Nur die nackte Kraft der HOPE-LoRA Architektur.

Ziel:
1. LoRA-Adapter lernen, wie man die Daten repräsentiert.
2. Update-Netzwerke lernen, wie man das Memory füllt.
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

from kitten_model import HOPEModel
from kitten_lora import HOPEConfig

# ═════════════════════════════════════════════════════════
# Konfiguration
# ═════════════════════════════════════════════════════════
smol = False  # Setze auf True für das kleine Modell (0.6B), False für 1.7B


SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_DIR = SCRIPT_DIR.parent / "models" / "kitten_simple_big" if smol == False else SCRIPT_DIR.parent / "models" / "kitten_simple_smol"
CACHE_DIR = SCRIPT_DIR.parent / "cache"



MODEL_ID = "Qwen/Qwen3-1.7B" if smol == False else "Qwen/Qwen3-0.6B"

# Training
MAX_SEQ_LEN = 512
BATCH_SIZE = 2 if smol else 1      # Absolut sicher für 16GB VRAM
NUM_EPOCHS = 50
LOG_INTERVAL = 10
SAVE_INTERVAL = 500

# Garbage Collection
GC_INTERVAL = 10
EMPTY_CACHE_INTERVAL = 100
# ═════════════════════════════════════════════════════════
# HOPE Config (Bleibt gleich - die Architektur ist perfekt)
# ═════════════════════════════════════════════════════════
HOPE_CONFIG = HOPEConfig(
    
    
    r_fast=8 if smol else 16,
    r_medium=32 if smol else 64,
    r_slow=64 if smol else 128,
    chunk_medium=16 if smol else 32,
    chunk_slow=64 if smol else 128,
    hidden_dim=64 if smol else 128,
    
    # WICHTIG: Kein Newton-Schulz für mehr "natürliche" Updates
    use_newton_schulz=False, 
    
    # WICHTIG: Langsamerer Decay, damit sich Wissen aufbaut
    memory_decay=0.9995,
    
    # Wir filtern jetzt nicht so hart, das machen die Gewichte selbst
    surprise_threshold=-1.0, 
    
    # Die LRs hier werden nur für die *Initialisierung* der Update-Netzwerke verwendet
    lr_fast=0.2,
    lr_medium=0.05,
    lr_slow=0.01,
)

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
            text += f"<|im_start|>{msg.get('role', 'user')}\n{msg.get('content', '')}<|im_end|>\n"
        
        out = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length", return_tensors="pt")
        out = {k: v.squeeze(0) for k, v in out.items()}
        out["labels"] = out["input_ids"].clone()
        out["labels"][out["labels"] == tokenizer.pad_token_id] = -100
        return out
    
    tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names, desc="Tokenizing")
    tokenized.set_format("torch")
    print(f"✅ {len(tokenized['train']):,} train / {len(tokenized['validation']):,} val")
    return tokenized

class SimpleHOPETrainer:
    def __init__(self, model: HOPEModel, train_loader: DataLoader, val_loader: DataLoader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = next(model.model.parameters()).device
        
        # ═════════════════════════════════════════════════════════
        # 1. MEMORY OPTIMIZER (Der "Writer")
        # Trainiert update_fast, update_medium, update_slow, gates, surprise
        # ═════════════════════════════════════════════════════════
        memory_params = []
        for layer in model.hope_layers:
            memory_params.extend(layer.update_fast.parameters())
            memory_params.extend(layer.update_medium.parameters())
            memory_params.extend(layer.update_slow.parameters())
            memory_params.extend(layer.surprise_net.parameters())
            memory_params.extend(layer.gate_net.parameters())
            
        self.memory_optimizer = torch.optim.AdamW(
            memory_params,
            lr=5e-5,  # Standard Learning Rate
            weight_decay=0.01,
        )
        
        # ═════════════════════════════════════════════════════════
        # 2. LORA OPTIMIZER (Der "Adapter")
        # Trainiert proj_down, proj_up
        # ═════════════════════════════════════════════════════════
        lora_params = []
        for layer in model.hope_layers:
            lora_params.extend(layer.proj_down.parameters())
            lora_params.extend(layer.proj_up.parameters())
            
        self.lora_optimizer = torch.optim.AdamW(
            lora_params,
            lr=5e-5,  # Etwas höher für die Projektion
            weight_decay=0.01,
        )
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {"train_loss": [], "val_loss": []}
        
        print(f"\n🐱 Simple HOPE Trainer Initialized")
        print(f"   Device: {self.device}")
        print(f"   Memory Optimizer (Writer): {sum(p.numel() for p in memory_params):,} params")
        print(f"   LoRA Optimizer (Adapter): {sum(p.numel() for p in lora_params):,} params")
    
    def train_epoch(self, epoch: int) -> float:
        self.model.model.train()
        total_loss = 0
        num_batches = 0
        
        if epoch == 0:
            self.model.reset_memory(BATCH_SIZE)
        
        progress = tqdm(self.train_loader, desc=f"🐱 Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for step, batch in enumerate(progress):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # ═════════════════════════════════════════════════════════
            # Einfacher Trainings-Loop
            # ═════════════════════════════════════════════════════════
            
            # 1. Zero Grads
            self.memory_optimizer.zero_grad()
            self.lora_optimizer.zero_grad()
            
            # 2. Forward
            outputs = self.model.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            
            loss = outputs.loss
            
            # 3. Backward
            loss.backward()
            
            # 4. Clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.memory_optimizer.param_groups[0]['params']) + \
                list(self.lora_optimizer.param_groups[0]['params']),
                1.0
            )
            
            # 5. Step (Beide Optimizer machen ihren Job)
            self.memory_optimizer.step()
            self.lora_optimizer.step()
            
            # ═════════════════════════════════════════════════════════
            # Tracking
            # ═════════════════════════════════════════════════════════
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if step % LOG_INTERVAL == 0:
                mem_stats = self.model.get_memory_stats()
                progress.set_postfix({
                    "loss": f"{(total_loss/num_batches):.4f}",
                    "F": f"{mem_stats['fast_norm_avg']:.2f}",
                    "M": f"{mem_stats['medium_norm_avg']:.2f}",
                    "S": f"{mem_stats['slow_norm_avg']:.2f}",
                })
            
            # GC
            if step % GC_INTERVAL == 0:
                gc.collect()
            if step % EMPTY_CACHE_INTERVAL == 0:
                torch.cuda.empty_cache()
            
            # Checkpoints
            if self.global_step % SAVE_INTERVAL == 0:
                self._save_checkpoint(f"step_{self.global_step}")
        
        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> float:
        self.model.model.eval()
        total_loss = 0
        num_batches = 0
        
        # Reset für saubere Validation (Optional)
        self.model.reset_memory(BATCH_SIZE)
        
        for batch in tqdm(self.val_loader, desc="📋 Validation", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            total_loss += outputs.loss.item()
            num_batches += 1
            
            del outputs, batch
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return total_loss / num_batches
    
    def train(self):
        print("\n" + "="*60)
        print("🐱 SIMPLE HOPE TRAINING (No Meta-Learning)")
        print("="*60)
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        
        for epoch in range(NUM_EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            
            print(f"\n📊 Epoch {epoch+1}:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")
            
            mem_stats = self.model.get_memory_stats()
            print(f"   Memory - F/M/S: {mem_stats['fast_norm_avg']:.2f} / {mem_stats['medium_norm_avg']:.2f} / {mem_stats['slow_norm_avg']:.2f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best")
                print(f"   ✅ New Best Model!")
        
        self._save_checkpoint("final")
        
        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print(f"✅ TRAINING COMPLETE in {elapsed/60:.1f} min")
        print(f"   Best Val Loss: {self.best_val_loss:.4f}")
        print("="*60)
    
    def _save_checkpoint(self, name: str):
        path = OUTPUT_DIR / name
        path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_hope_weights(str(path))
        torch.save({
            "memory_opt": self.memory_optimizer.state_dict(),
            "lora_opt": self.lora_optimizer.state_dict(),
        }, path / "optimizers.pt")
        
        with open(path / "training_state.json", "w") as f:
            json.dump({
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "history": self.history,
            }, f, indent=2)
        
        print(f"   💾 Saved: {path}")

def main():
    print("="*60)
    print("🐱 SIMPLE HOPE PIPELINE")
    print("="*60)
    
    model = HOPEModel(
        model_id=MODEL_ID,
        config=HOPE_CONFIG,
        cache_dir=str(CACHE_DIR),
    )
    
    tokenized = load_dataset_kitten(model.tokenizer)
    
    train_loader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(tokenized["validation"], batch_size=BATCH_SIZE, num_workers=0)
    
    trainer = SimpleHOPETrainer(model, train_loader, val_loader)
    
    # Optional: Resetten falls was schiefgelaufen ist
    # trainer.reset_cheating_agents() # (Du hast diese Methode noch im Code drin, wenn du sie willst)
    
    trainer.train()

if __name__ == "__main__":
    main()