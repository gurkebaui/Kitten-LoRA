# src/train_kitten.py
"""
Kitten-LoRA Training mit HOPE-Integration.
Trainiert die Update-Netze für Memory-basiertes Lernen.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import json
import time
import random

from kitten_model import HOPEModel
from kitten_lora import HOPEConfig


# ═══════════════════════════════════════════════════════════
# Konfiguration
# ═══════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_DIR = SCRIPT_DIR.parent / "models" / "kitten_hope"
CACHE_DIR = SCRIPT_DIR.parent / "cache"

MODEL_ID = "Qwen/Qwen3-0.6B"

# Training Hyperparameter
MAX_SEQ_LEN = 512
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
MEMORY_RESET_INTERVAL = 32  # Reset Memory alle N Batches (simuliert neue Konversationen)

# HOPE Config - angepasst für 0.6B Modell
HOPE_CONFIG = HOPEConfig(
    r_fast=4,
    r_medium=16,
    r_slow=32,      # Reduziert für kleineres Modell
    chunk_medium=8,
    chunk_slow=32,  # Reduziert
    alpha=1.0,
    surprise_threshold=0.3,
    lr_fast=0.1,
    lr_medium=0.05,
    lr_slow=0.01,
    hidden_dim=32,  # Reduziert für kleineres Modell
)


# ═══════════════════════════════════════════════════════════
# Dataset Vorbereitung
# ═══════════════════════════════════════════════════════════
class KittenDataset:
    """Dataset-Handler für Kitten-LoRA Training."""
    
    def __init__(self, tokenizer, max_length: int = MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load(self):
        """Lädt und tokenisiert das Dataset."""
        print("📂 Lade Kitten Dataset...")
        
        train_path = DATA_DIR / "train.jsonl"
        val_path = DATA_DIR / "val.jsonl"
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data nicht gefunden: {train_path}")
        
        dataset = load_dataset(
            "json",
            data_files={
                "train": str(train_path),
                "validation": str(val_path) if val_path.exists() else str(train_path),
            }
        )
        
        tokenized = dataset.map(
            self._tokenize,
            remove_columns=dataset["train"].column_names,
            desc="🔧 Tokenizing",
            num_proc=1,
        )
        tokenized.set_format("torch")
        
        print(f"✅ Dataset geladen:")
        print(f"   Train: {len(tokenized['train']):,} samples")
        print(f"   Val: {len(tokenized['validation']):,} samples")
        
        return tokenized
    
    def _tokenize(self, batch):
        """Tokenisiert eine Konversation."""
        text = ""
        for msg in batch["messages"]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        out = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        out = {k: v.squeeze(0) for k, v in out.items()}
        out["labels"] = out["input_ids"].clone()
        out["labels"][out["labels"] == self.tokenizer.pad_token_id] = -100
        
        return out


# ═══════════════════════════════════════════════════════════
# Parameter-Verwaltung
# ═══════════════════════════════════════════════════════════
def get_trainable_params(model: HOPEModel) -> tuple:
    """Extrahiert alle trainierbaren HOPE-Parameter."""
    params = []
    param_groups = {
        "projections": [],
        "update_nets": [],
        "surprise_net": [],
        "gate_net": [],
    }
    
    for layer in model.hope_layers:
        # Projektionen (LoRA-Kern)
        param_groups["projections"].extend([
            layer.proj_down.weight,
            layer.proj_up.weight,
        ])
        
        # Update-Netze (Memory-Lernen)
        for p in layer.update_fast.parameters():
            param_groups["update_nets"].append(p)
        for p in layer.update_medium.parameters():
            param_groups["update_nets"].append(p)
        for p in layer.update_slow.parameters():
            param_groups["update_nets"].append(p)
        
        # Surprise-Netz
        for p in layer.surprise_net.parameters():
            param_groups["surprise_net"].append(p)
        
        # Gate-Netz
        for p in layer.gate_net.parameters():
            param_groups["gate_net"].append(p)
    
    # Alle Parameter kombinieren
    all_params = []
    for group_name, group_params in param_groups.items():
        all_params.extend(group_params)
    
    # Statistik
    stats = {name: sum(p.numel() for p in params) for name, params in param_groups.items()}
    total = sum(stats.values())
    
    print(f"\n📊 Trainierbare Parameter:")
    for name, count in stats.items():
        print(f"   {name}: {count:,} ({100*count/total:.1f}%)")
    print(f"   Total: {total:,}")
    
    return all_params, param_groups


# ═══════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════
class KittenTrainer:
    """Trainer für Kitten-LoRA mit HOPE."""
    
    def __init__(
        self,
        model: HOPEModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = LEARNING_RATE,
        num_epochs: int = NUM_EPOCHS,
        grad_accum: int = GRAD_ACCUM,
        warmup_ratio: float = WARMUP_RATIO,
        output_dir: Path = OUTPUT_DIR,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.grad_accum = grad_accum
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = next(model.model.parameters()).device
        
        # Parameter
        self.trainable_params, self.param_groups = get_trainable_params(model)
        
        # Optimizer mit unterschiedlichen LRs für verschiedene Gruppen
        self.optimizer = torch.optim.AdamW([
            {"params": self.param_groups["projections"], "lr": learning_rate},
            {"params": self.param_groups["update_nets"], "lr": learning_rate * 2},  # Höhere LR für Update-Netze
            {"params": self.param_groups["surprise_net"], "lr": learning_rate},
            {"params": self.param_groups["gate_net"], "lr": learning_rate},
        ], weight_decay=0.01)
        
        # Scheduler
        num_training_steps = len(train_loader) * num_epochs // grad_accum
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "memory_stats": [],
        }
        
        print(f"\n🔧 Trainer Setup:")
        print(f"   Epochs: {num_epochs}")
        print(f"   Effective Batch: {BATCH_SIZE} × {grad_accum} = {BATCH_SIZE * grad_accum}")
        print(f"   Training Steps: {num_training_steps}")
        print(f"   Warmup Steps: {num_warmup_steps}")
    
    def train_epoch(self, epoch: int) -> float:
        """Trainiert eine Epoche."""
        self.model.model.train()
        total_loss = 0
        num_batches = 0
        
        # Reset Memory zu Beginn der Epoche
        self.model.reset_memory(BATCH_SIZE)
        
        progress = tqdm(
            self.train_loader,
            desc=f"🐱 Epoch {epoch+1}/{self.num_epochs}",
            leave=True,
        )
        
        for step, batch in enumerate(progress):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward
            outputs = self.model.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            
            loss = outputs.loss / self.grad_accum
            loss.backward()
            
            total_loss += loss.item() * self.grad_accum
            num_batches += 1
            
            # Gradient Step
            if (step + 1) % self.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Progress Update
                current_loss = total_loss / num_batches
                stats = self.model.get_memory_stats()
                
                progress.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                    "mem": f"F{stats['fast_norm_avg']:.1f}/M{stats['medium_norm_avg']:.1f}/S{stats['slow_norm_avg']:.1f}",
                })
            
            # Memory Reset (simuliert neue Konversationen)
            if (step + 1) % MEMORY_RESET_INTERVAL == 0:
                self.model.reset_memory(BATCH_SIZE)
        
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
        print("🐱 KITTEN-LoRA TRAINING START")
        print("="*60)
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            self.history["train_loss"].append(train_loss)
            
            # Validation
            val_loss = self.validate()
            self.history["val_loss"].append(val_loss)
            
            # Memory Stats
            stats = self.model.get_memory_stats()
            self.history["memory_stats"].append(stats)
            
            # Logging
            print(f"\n📊 Epoch {epoch+1} Complete:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Memory: F={stats['fast_norm_avg']:.2f} M={stats['medium_norm_avg']:.2f} S={stats['slow_norm_avg']:.2f}")
            
            # Checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best")
                print(f"   ✅ Neuer Best Checkpoint!")
            
            # Periodic Save
            self.save_checkpoint(f"epoch_{epoch+1}")
        
        # Finale Speicherung
        self.save_checkpoint("final")
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE")
        print(f"   Duration: {elapsed/60:.1f} minutes")
        print(f"   Best Val Loss: {self.best_val_loss:.4f}")
        print(f"   Output: {self.output_dir}")
        print("="*60)
        
        return self.history
    
    def save_checkpoint(self, name: str):
        """Speichert einen Checkpoint."""
        checkpoint_dir = self.output_dir / name
        self.model.save_hope_weights(str(checkpoint_dir))
        
        # Training State
        state = {
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "config": {
                "r_fast": HOPE_CONFIG.r_fast,
                "r_medium": HOPE_CONFIG.r_medium,
                "r_slow": HOPE_CONFIG.r_slow,
                "hidden_dim": HOPE_CONFIG.hidden_dim,
            }
        }
        
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)


# ═══════════════════════════════════════════════════════════
# Quick Test
# ═══════════════════════════════════════════════════════════
def quick_test(model: HOPEModel):
    """Schneller Test nach dem Training."""
    print("\n" + "="*60)
    print("🧪 QUICK TEST")
    print("="*60)
    
    test_prompts = [
        "Hello! What's your name?",
        "Tell me a fun fact.",
        "What did I just ask you?",  # Memory-Test
    ]
    
    model.reset_memory(1)
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n[Turn {i+1}]")
        print(f"👤 {prompt}")
        
        response = model.generate(
            prompt,
            max_new_tokens=64,
            temperature=0.7,
            reset_memory=False,
        )
        print(f"🐱 {response}")
    
    stats = model.get_memory_stats()
    print(f"\n📊 Memory nach Test: F={stats['fast_norm_avg']:.2f} M={stats['medium_norm_avg']:.2f} S={stats['slow_norm_avg']:.2f}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():
    print("="*60)
    print("🐱 KITTEN-LoRA HOPE TRAINING")
    print("="*60)
    
    # Modell laden
    print("\n📦 Lade Modell...")
    model = HOPEModel(
        model_id=MODEL_ID,
        config=HOPE_CONFIG,
        cache_dir=str(CACHE_DIR),
    )
    
    # Dataset
    dataset_handler = KittenDataset(model.tokenizer)
    tokenized = dataset_handler.load()
    
    train_loader = DataLoader(
        tokenized["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    
    val_loader = DataLoader(
        tokenized["validation"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    # Trainer
    trainer = KittenTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    
    # Training
    history = trainer.train()
    
    # Quick Test
    quick_test(model)
    
    print("\n🐱 Kitten-LoRA Training abgeschlossen!")
    print(f"   Checkpoint: {OUTPUT_DIR}/best")


if __name__ == "__main__":
    main()