# src/train_kitten_optimal.py
"""
Kitten-LoRA Optimiertes Training für Infinite Context (Nested Learning).

FEATURES:
- Infinite Context: Kein Hard-Reset des Memory (CMS aktiv).
- HOPE/Nested Learning: Hohe Ranks & langsamer Memory Decay.
- Deep Optimizer: Stabilisiert für lange Sequenzen.
- RAM Safety: Aggressives GC für 16GB VRAM.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple, Any
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
BATCH_SIZE = 2        # 16GB VRAM sollte BS=2 problemlos schaffen
NUM_EPOCHS = 50
LOG_INTERVAL = 10
SAVE_INTERVAL = 500

# ═══════════════════════════════════════════════════════════
# MEMORY SETTINGS (KRITISCH FÜR INFINITE CONTEXT)
# ═══════════════════════════════════════════════════════════

# Wir resetten das Memory NIE während des Trainings.
# Das CMS (Continuum Memory System) übernimmt das "Vergessen" via Decay.
# Wir setzen das Intervall extrem hoch.
MEMORY_RESET_INTERVAL = 9999999 

# Garbage Collection
GC_INTERVAL = 5     
EMPTY_CACHE_INTERVAL = 100

# HOPE Config: Optimized for CMS Persistence & Nested Learning
HOPE_CONFIG = HOPEConfig(
    # Höhere Ranks für mehr "Schreibkapazität" im Memory
    r_fast=8,              
    r_medium=32,           
    r_slow=64,             
    
    # Selteneres Update = Langsamere Zeitskale (Theorie aus dem Paper)
    chunk_medium=16,       
    chunk_slow=64,         
    
    # Größerer Hidden State für komplexere Gate/Update Funktionen
    hidden_dim=64,         
    
    use_newton_schulz=False, # scützt vor Instabilitäten    wenn true 
    
    # Sehr langsamer Decay (0.9999 ≈ Unendlich, 0.9995 ist stabil)
    memory_decay=0.999,    
    
    surprise_threshold=-1.0,
    
    # Konservativere LRs für Stabilität bei wachsendem Memory
    lr_fast=0.2,          
    lr_medium=0.05,
    lr_slow=0.01,
)

# Deep Optimizer Config
DEEP_OPT_CONFIG = DeepOptimizerConfig(
    hidden_dim=128,
    num_levels=3,
    
    # 🔧 FIX 1: Meta LR drastisch erhöhen
    # Das Netzwerk muss lernen, die Gradienten zu interpretieren.
    # 5e-5 war zu passiv. 1e-3 ist ein guter Startpunkt für Meta-Learning.
    meta_lr=1e-4,      
    
    grad_clip=1.0,     # Sicherheitsgurt bleibt
    
    # 🔧 FIX 2: Max Update Norm erhöhen
    # Wir erlauben dem Deep Optimizer, größere Schritte vorzuschlagen.
    # Das hilft ihm, Einfluss auf die LoRA-Gewichte zu nehmen.
    max_update_norm=0.1, 
    
    # ... Rest wie gehabt
    update_reg_weight=0.1,
    surprise_threshold=-1.0, # Wird eh intern vom Netzwerk berechnet
    consolidation_interval=200,
    slow_consolidation_interval=1000,
    momentum_beta=0.9,
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
    """Sammelt Parameter für den Base Optimizer."""
    params = []
    
    for layer in model.hope_layers:
        params.extend(layer.update_fast.parameters())
        params.extend(layer.update_medium.parameters())
        params.extend(layer.update_slow.parameters())
        params.extend(layer.surprise_net.parameters())
        params.extend(layer.gate_net.parameters())
    
    return params


class KittenFullTrainer:
    """Trainer mit Phase A / Phase B Trennung."""
    
    def __init__(self, model: HOPEModel, train_loader: DataLoader, val_loader: DataLoader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = next(model.model.parameters()).device
        
        # Deep Optimizer
        self.deep_optimizer = DeepOptimizerManager(
            config=DEEP_OPT_CONFIG,
            device=self.device,
        )
        
        # Base Optimizer (für Memory-Netzwerke: update_fast, gate_net, etc.)
        base_params = get_base_optimizer_params(model)
        self.base_optimizer = torch.optim.AdamW(
            base_params,
            lr=2e-5,
            weight_decay=0.01,
        )
        
        # ═══════════════════════════════════════════════════════════
        # META OPTIMIZER (Separater Optimizer für den Deep Optimizer!)
        # ═══════════════════════════════════════════════════════════
        self.meta_optimizer = torch.optim.AdamW(
            self.deep_optimizer.optimizer_net.parameters(),
            lr=1e-3,  # Höher als meta_lr im Config, weil wir echte Signale haben
            weight_decay=0.01,
        )
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {"train_loss": [], "val_loss": [], "meta_loss": []}
        
        print(f"\n🐱 Kitten Full Trainer (Phase A/B) Initialized")
        print(f"   Base Optimizer: {sum(p.numel() for p in base_params):,} params")
        print(f"   Meta Optimizer: {sum(p.numel() for p in self.deep_optimizer.optimizer_net.parameters()):,} params")
    
    def train_epoch(self, epoch: int) -> float:
        """Training mit Phase A / Phase B Trennung."""
        self.model.model.train()
        total_loss = 0
        total_meta_loss = 0
        num_batches = 0
        meta_steps = 0
        
        if epoch == 0:
            self.model.reset_memory(BATCH_SIZE)
        self.deep_optimizer.reset()
        
        progress = tqdm(self.train_loader, desc=f"🐱 Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for step, batch in enumerate(progress):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # ═════════════════════════════════════════════════════
            # PHASE A: BASE TRAINING
            # ═════════════════════════════════════════════════════
            self.base_optimizer.zero_grad()
            
            outputs = self.model.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            
            # Backward für Base Optimizer
            loss.backward()
            
            # Deep Optimizer wendet Updates an (nicht-differenzierbar, Phase A)
            self.deep_optimizer.step(
                hope_layers=self.model.hope_layers,
                main_loss=loss,
                do_meta_update=False,  # KEIN Meta-Update in Phase A!
            )
            
            # Base Optimizer Step
            torch.nn.utils.clip_grad_norm_(
                get_base_optimizer_params(self.model),
                1.0
            )
            self.base_optimizer.step()
            
            # ═════════════════════════════════════════════════════
            # PHASE B: META TRAINING (alle N Steps)
            # ═════════════════════════════════════════════════════
            if step % 5 == 0 and step > 0:
                meta_loss = self._do_meta_learning_step(batch)
                if meta_loss is not None:
                    total_meta_loss += meta_loss
                    meta_steps += 1
            
            # Tracking
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if step % LOG_INTERVAL == 0:
                avg_meta = total_meta_loss / max(1, meta_steps)
                deep_stats = self.deep_optimizer.get_stats()
                
                progress.set_postfix({
                    "loss": f"{(total_loss/num_batches):.4f}",
                    "meta": f"{avg_meta:.4f}",
                    "d_lr": f"{deep_stats['lr_avg']:.4f}",
                })
            
            # Memory Management
            if step % GC_INTERVAL == 0:
                gc.collect()
            if step % EMPTY_CACHE_INTERVAL == 0:
                torch.cuda.empty_cache()
            
            # Checkpoints
            if self.global_step % SAVE_INTERVAL == 0:
                self._save_checkpoint(f"step_{self.global_step}")
        
        avg_meta = total_meta_loss / max(1, meta_steps)
        self.history["meta_loss"].append(avg_meta)
        
        return total_loss / num_batches
    
    def _do_meta_learning_step(self, batch) -> Optional[float]:
        """
        Phase B: Meta-Learning durch Simulation.
        
        Ablauf:
        1. Speichere aktuellen Zustand (Snapshot)
        2. Forward Pass 1 → Loss_before
        3. Berechne differenzierbares Update
        4. Wende Update an
        5. Forward Pass 2 → Loss_after
        6. Meta-Loss = Loss_after (wir wollen, dass es sinkt)
        7. Backprop durch den Deep Optimizer
        8. Rollback zum Snapshot
        """
        try:
            # ═══════════════════════════════════════════════════════
            # 1. SNAPSHOT erstellen
            # ═══════════════════════════════════════════════════════
            snapshot = self._create_snapshot()
            
            # ═══════════════════════════════════════════════════════
            # 2. Forward Pass 1 (Baseline)
            # ═══════════════════════════════════════════════════════
            self.model.model.zero_grad()
            
            outputs_before = self.model.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss_before = outputs_before.loss
            
            # Wir brauchen Gradienten für den Deep Optimizer
            loss_before.backward(retain_graph=False)
            
            # ═══════════════════════════════════════════════════════
            # 3. Berechne differenzierbare Updates
            # ═══════════════════════════════════════════════════════
            self.meta_optimizer.zero_grad()
            
            deltas, lr_tensors = self.deep_optimizer.compute_differentiable_update(
                self.model.hope_layers,
                detach_grads=True,  # Gradienten vom Hauptmodell trennen
            )
            
            if not deltas:
                self._restore_snapshot(snapshot)
                return None
            
            # ═══════════════════════════════════════════════════════
            # 4. Wende Updates an (differenzierbar!)
            # ═══════════════════════════════════════════════════════
            self.deep_optimizer.apply_deltas_differentiable(
                self.model.hope_layers,
                deltas,
            )
            
            # ═══════════════════════════════════════════════════════
            # 5. Forward Pass 2 (Nach Update)
            # ═══════════════════════════════════════════════════════
            outputs_after = self.model.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss_after = outputs_after.loss
            
            # ═══════════════════════════════════════════════════════
            # 6. Meta-Loss berechnen
            # ═══════════════════════════════════════════════════════
            # Wir wollen, dass loss_after < loss_before
            # Also minimieren wir loss_after direkt
            # (Alternative: loss_after - loss_before.detach())
            
            meta_loss = loss_after
            
            # Regularisierung: Verhindere zu große Updates
            reg_loss = sum(lr ** 2 for lr in lr_tensors.values()) * 0.1
            meta_loss = (loss_after - loss_before.detach()) + reg_loss
            
            # ═══════════════════════════════════════════════════════
            # 7. Backprop durch den Deep Optimizer
            # ═══════════════════════════════════════════════════════
            meta_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.deep_optimizer.optimizer_net.parameters(),
                1.0
            )
            self.meta_optimizer.step()
            
            # ═══════════════════════════════════════════════════════
            # 8. ROLLBACK zum Snapshot
            # ═══════════════════════════════════════════════════════
            self._restore_snapshot(snapshot)
            
            return meta_loss.item()
        
        except RuntimeError as e:
            print(f"⚠️ Meta-Learning Error: {e}")
            self._restore_snapshot(snapshot)
            return None
    
    def _create_snapshot(self) -> Dict:
        """Erstellt einen Snapshot des aktuellen Zustands."""
        snapshot = {
            "lora_weights": [],
            "memory_states": [],
        }
        
        for layer in self.model.hope_layers:
            snapshot["lora_weights"].append({
                "down": layer.proj_down.weight.data.clone(),
                "up": layer.proj_up.weight.data.clone(),
            })
            
            if layer._memory_state is not None:
                state = layer._memory_state
                snapshot["memory_states"].append({
                    "fast": state.fast.clone() if state.fast is not None else None,
                    "medium": state.medium.clone() if state.medium is not None else None,
                    "slow": state.slow.clone() if state.slow is not None else None,
                    "step": state.step,
                })
            else:
                snapshot["memory_states"].append(None)
        
        return snapshot
    
    def _restore_snapshot(self, snapshot: Dict):
        """Stellt den Zustand aus einem Snapshot wieder her."""
        for i, layer in enumerate(self.model.hope_layers):
            # LoRA Weights
            layer.proj_down.weight = torch.nn.Parameter(
                snapshot["lora_weights"][i]["down"]
            )
            layer.proj_up.weight = torch.nn.Parameter(
                snapshot["lora_weights"][i]["up"]
            )
            
            # Memory State
            if snapshot["memory_states"][i] is not None:
                saved = snapshot["memory_states"][i]
                state = layer._memory_state
                if state is not None:
                    if saved["fast"] is not None:
                        state.fast = saved["fast"]
                    if saved["medium"] is not None:
                        state.medium = saved["medium"]
                    if saved["slow"] is not None:
                        state.slow = saved["slow"]
                    state.step = saved["step"]
    
    @torch.no_grad()
    def validate(self) -> float:
        """
        Validierung.
        Resetten wir hier das Memory? Für 'True Infinite Context' Evaluation eher nicht,
        um zu sehen, wie das Modell auf den Training-Context reagiert.
        Für saubere Loss-Werte (Generalisierung) resetten wir hier aber.
        """
        self.model.model.eval()
        total_loss = 0
        num_batches = 0
        
        # Reset nur am Anfang der Validation für saubere Messung
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
        """Vollständiges Training."""
        print("\n" + "="*60)
        print("🐱 KITTEN-LoRA INFINITE CONTEXT TRAINING")
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
            
            deep_stats = self.deep_optimizer.get_stats()
            mem_stats = self.model.get_memory_stats()
            print(f"   Deep Opt - LR: {deep_stats['lr_avg']:.5f}, Mom: {deep_stats['momentum_avg']:.4f}")
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
        """Speichert Checkpoint."""
        path = OUTPUT_DIR / name
        path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_hope_weights(str(path))
        self.deep_optimizer.save(str(path / "deep_optimizer.pt"))
        torch.save(self.base_optimizer.state_dict(), path / "base_optimizer.pt")
        
        with open(path / "training_state.json", "w") as f:
            json.dump({
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "history": self.history,
            }, f, indent=2)
        
        print(f"   💾 Saved: {path}")


def quick_test(model: HOPEModel):
    """Schneller Inference Test."""
    print("\n🧪 Quick Inference Test (Infinite Context)")
    
    # NICHT resetten, um das Memory aus dem Training zu nutzen (sofern nicht grad enabled)
    # Aber für einen sauberen Test resetten wir hier manuell
    model.reset_memory(1)
    
    _ = model.generate("My name is Kitten and I love neural networks.", max_new_tokens=20, reset_memory=False)
    response = model.generate("What is my name and what do I love?", max_new_tokens=30, temperature=0, reset_memory=False)
    
    print(f"   Q: What is my name and what do I love?")
    print(f"   A: {response[:100]}")
    
    passed = "kitten" in response.lower() and "network" in response.lower()
    print(f"   {'✅ Memory Test Passed' if passed else '❌ Memory Test Failed'}")


def main():
    print("="*60)
    print("🐱 KITTEN-LoRA INFINITE CONTEXT PIPELINE")
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