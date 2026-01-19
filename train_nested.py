# src/train_kitten_full.py
"""
Kitten-LoRA Training mit korrektem Deep Optimizer.

FIXES:
- Klare Trennung: Deep Opt → LoRA, Base Opt → Update-Netze
- Aggressives Memory Management
- Korrekter Meta-Learning Loop
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
from deep_optimizer import DeepOptimizerManager, DeepOptimizerConfig


# ═══════════════════════════════════════════════════════════
# Konfiguration
# ═══════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_DIR = SCRIPT_DIR.parent / "models" / "kitten_full"
CACHE_DIR = SCRIPT_DIR.parent / "cache"

MODEL_ID = "Qwen/Qwen3-0.6B"

# Training
MAX_SEQ_LEN = 512
BATCH_SIZE = 1
NUM_EPOCHS = 3
LOG_INTERVAL = 50
SAVE_INTERVAL = 500
GC_INTERVAL = 50  # Häufiger aufräumen
MEMORY_RESET_INTERVAL = 32

# HOPE Config
HOPE_CONFIG = HOPEConfig(
    r_fast=4,
    r_medium=16,
    r_slow=32,
    chunk_medium=8,
    chunk_slow=32,
    hidden_dim=32,
    use_newton_schulz=True,
    memory_decay=0.999,
)

# Deep Optimizer Config
DEEP_OPT_CONFIG = DeepOptimizerConfig(
    hidden_dim=64,
    num_levels=3,
    meta_lr=1e-4,
    max_update_norm=0.1,
    momentum_beta=0.9,
    state_detach_interval=10,
    consolidation_interval=100,
    slow_consolidation_interval=500,
)


def load_dataset_kitten(tokenizer):
    """Lädt das Dataset."""
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


def get_base_optimizer_params(model: HOPEModel) -> list:
    """
    Sammelt Parameter für den Base Optimizer.
    
    WICHTIG: NUR die Update-Netzwerke, Surprise-Net, Gate-Net.
    NICHT die LoRA-Projektionen (die macht der Deep Optimizer).
    """
    params = []
    
    for layer in model.hope_layers:
        # Update Networks
        params.extend(layer.update_fast.parameters())
        params.extend(layer.update_medium.parameters())
        params.extend(layer.update_slow.parameters())
        
        # Surprise Network
        params.extend(layer.surprise_net.parameters())
        
        # Gate Network
        params.extend(layer.gate_net.parameters())
    
    return params


class KittenFullTrainer:
    """Trainer mit korrekter Optimizer-Trennung."""
    
    def __init__(self, model: HOPEModel, train_loader: DataLoader, val_loader: DataLoader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = next(model.model.parameters()).device
        
        # Deep Optimizer (NUR für LoRA-Projektionen)
        self.deep_optimizer = DeepOptimizerManager(
            config=DEEP_OPT_CONFIG,
            device=self.device,
        )
        
        # Base Optimizer (NUR für Update-Netzwerke etc.)
        base_params = get_base_optimizer_params(model)
        self.base_optimizer = torch.optim.AdamW(
            base_params,
            lr=2e-5,
            weight_decay=0.01,
        )
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {"train_loss": [], "val_loss": []}
        
        # Parameter-Statistik
        deep_opt_params = sum(p.numel() for p in self.deep_optimizer.optimizer_net.parameters())
        base_opt_params = sum(p.numel() for p in base_params)
        lora_params = sum(
            layer.proj_down.weight.numel() + layer.proj_up.weight.numel()
            for layer in model.hope_layers
        )
        
        print(f"\n🐱 Kitten Full Trainer")
        print(f"   Deep Optimizer params: {deep_opt_params:,}")
        print(f"   Base Optimizer params: {base_opt_params:,}")
        print(f"   LoRA params (Deep Opt): {lora_params:,}")
    
    def train_epoch(self, epoch: int) -> float:
        """Trainiert eine Epoche."""
        self.model.model.train()
        total_loss = 0
        num_batches = 0
        
        self.model.reset_memory(BATCH_SIZE)
        self.deep_optimizer.reset()
        
        progress = tqdm(self.train_loader, desc=f"🐱 Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for step, batch in enumerate(progress):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # ═══════════════════════════════════════════════════════
            # 1. Forward + Backward
            # ═══════════════════════════════════════════════════════
            self.base_optimizer.zero_grad()
            
            outputs = self.model.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            
            loss = outputs.loss
            loss.backward()
            
            # ═══════════════════════════════════════════════════════
            # 2. Deep Optimizer Step (LoRA-Projektionen)
            # ═══════════════════════════════════════════════════════
            self.deep_optimizer.step(
                hope_layers=self.model.hope_layers,
                main_loss=loss,
                do_meta_update=(self.global_step % 10 == 0),
            )
            
            # ═══════════════════════════════════════════════════════
            # 3. Base Optimizer Step (Update-Netzwerke etc.)
            # ═══════════════════════════════════════════════════════
            torch.nn.utils.clip_grad_norm_(
                get_base_optimizer_params(self.model),
                1.0
            )
            self.base_optimizer.step()
            
            # ═══════════════════════════════════════════════════════
            # Tracking & Cleanup
            # ═══════════════════════════════════════════════════════
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if self.global_step % LOG_INTERVAL == 0:
                deep_stats = self.deep_optimizer.get_stats()
                mem_stats = self.model.get_memory_stats()
                
                progress.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{deep_stats['lr_avg']:.4f}",
                    "surp": f"{deep_stats['surprise_avg']:.2f}",
                    "F": f"{mem_stats['fast_norm_avg']:.2f}",
                })
            
            # Memory Reset (simuliert neue Konversation)
            if (step + 1) % MEMORY_RESET_INTERVAL == 0:
                self.model.reset_memory(BATCH_SIZE)
                self.deep_optimizer.reset()
            
            # Garbage Collection
            if self.global_step % GC_INTERVAL == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Checkpoint
            if self.global_step % SAVE_INTERVAL == 0:
                self._save_checkpoint(f"step_{self.global_step}")
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validierung."""
        self.model.model.eval()
        total_loss = 0
        num_batches = 0
        
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
        
        return total_loss / num_batches
    
    def train(self):
        """Vollständiges Training."""
        print("\n" + "="*60)
        print("🐱 KITTEN-LoRA FULL TRAINING")
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
            print(f"   Val Loss: {val_loss:.4f}")
            
            deep_stats = self.deep_optimizer.get_stats()
            print(f"   Deep Opt - LR: {deep_stats['lr_avg']:.4f}, Momentum: {deep_stats['momentum_avg']:.4f}")
            print(f"   Consolidations: {deep_stats['consolidations']}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best")
                print(f"   ✅ Best checkpoint!")
            
            # Epoch-Ende GC
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self._save_checkpoint("final")
        
        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print(f"✅ TRAINING COMPLETE in {elapsed/60:.1f} min")
        print(f"   Best Val Loss: {self.best_val_loss:.4f}")
        print("="*60)
    
    def _save_checkpoint(self, name: str):
        """Speichert Checkpoint."""
        path = OUTPUT_DIR / name
        path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_hope_weights(str(path))
        self.deep_optimizer.save(str(path / "deep_optimizer.pt"))
        
        # Base Optimizer State
        torch.save(self.base_optimizer.state_dict(), path / "base_optimizer.pt")
        
        with open(path / "training_state.json", "w") as f:
            json.dump({
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "history": self.history,
            }, f, indent=2)
        
        print(f"   💾 Saved: {path}")


def quick_test(model: HOPEModel):
    """Schneller Test."""
    print("\n🧪 Quick Test")
    
    model.reset_memory(1)
    
    _ = model.generate("My name is Kitten!", max_new_tokens=20, reset_memory=False)
    response = model.generate("What is my name?", max_new_tokens=30, temperature=0, reset_memory=False)
    
    print(f"   Q: What is my name?")
    print(f"   A: {response[:100]}")
    
    passed = "kitten" in response.lower()
    print(f"   {'✅ Pass' if passed else '❌ Fail'}")
    
    stats = model.get_memory_stats()
    print(f"   Memory: F={stats['fast_norm_avg']:.3f} M={stats['medium_norm_avg']:.3f} S={stats['slow_norm_avg']:.3f}")


def main():
    print("="*60)
    print("🐱 KITTEN-LoRA TRAINING")
    print("="*60)
    
    model = HOPEModel(
        model_id=MODEL_ID,
        config=HOPE_CONFIG,
        cache_dir=str(CACHE_DIR),
    )
    
    tokenized = load_dataset_kitten(model.tokenizer)
    
    train_loader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(tokenized["validation"], batch_size=BATCH_SIZE, num_workers=0)
    
    trainer = KittenFullTrainer(model, train_loader, val_loader)
    trainer.train()
    
    quick_test(model)


if __name__ == "__main__":
    main()