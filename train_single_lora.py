
# 02_train_single_lora.py
from email.mime import text
from pyexpat import model
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import bitsandbytes as bnb  # Für 8-Bit Optimizer

# ============================
# 1. KONFIGURATION
# ============================
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "single_lora_model"

MODEL_ID = "Qwen/Qwen3-0.6B"
CACHE_DIR = SCRIPT_DIR / "cache"

# LoRA HYPERPARAMETER (Slow-LoRA)
LORA_R = 128
LORA_ALPHA = 256
LORA_DROPOUT = 0.05  # Leichtes Dropout gegen Overfitting

# TRAINING HYPERPARAMETER
BATCH_SIZE = 1  # Muss 1 sein für 16GB VRAM mit r=128
GRADIENT_ACCUMULATION_STEPS = 16  # Simuliert Batch-Size 16
LEARNING_RATE = 5e-5  # Konservativ für stabiles Training
NUM_EPOCHS = 3  # 3 Epochen sind genug für Overfitting-Check
MAX_SEQ_LENGTH = 512  # Kritisch für VRAM

# ============================
# 2. MODELL & TOKENIZER LADEN
# ============================
def load_model_and_tokenizer():
    """Lädt das FP8-Modell direkt – ohne weitere Quantisierung."""
    print("🤖 Lade Tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        padding_side="right",
        force_download=True,
        use_fast=False,      # ← NEU: Python-Tokeniser
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    print("🤖 Lade FP8-Modell (keine weitere 8-Bit-Quantisierung)...")
    # Wir laden das Modell einfach so, wie es auf HF liegt – FP8 ist schon drin
    # 02_train_single_lora.py
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="sdpa",
        use_cache=False,
    )

   

    # Trotzdem Gradient-Checkpointing für VRAM sparen
    #model.gradient_checkpointing_enable()
    
    return model, tokenizer

# ============================
# 3. LoRA KONFIGURATION
# ============================
def setup_lora(model):
    """Konfiguriert LoRA-Adapter."""
    print("🔧 Konfiguriere LoRA...")
    
    # Ziele ALLE Linearen Layer im Modell
    # Das ist aggressiv, aber für deinen Use-Case ideal
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",     # MLP
       # "embed_tokens", "lm_head"               # Optional: Embeddings (spääter testen)
    ]
    
    lora_config = LoraConfig(
        r=LORA_R,  # Rank 128 = hohe Kapazität
        lora_alpha=LORA_ALPHA,  # Alpha = 2*r ist Standard
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none",  # Bias-Parameter nicht trainieren
        task_type="CAUSAL_LM",  # Für Sprachmodelle
    )
    
    # Wickle Modell mit LoRA
    model = get_peft_model(model, lora_config)
    
    # Zeige trainierbare Parameter
    model.print_trainable_parameters()
    
    return model

# ============================
# 4. DATEN LADEN & FORMATIEREN
# ============================
def load_and_format_data(tokenizer):
    """Lädt train.jsonl und val.jsonl und formatiert für Training."""
    print("📂 Lade Trainingsdaten...")
    
    # Lade mit Hugging Face datasets (effizient & schnell)
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(DATA_DIR / "train.jsonl"),
            "validation": str(DATA_DIR / "val.jsonl")
        }
    )
    
    # Formatierungsfunktion: Wandelt ChatML in "text" um
    def format_chatml(example):
        """Erzeugt echtes user/assistant-Wechselspiel."""
        text = ""
        # Rolle wechselt bei jeder Nachricht
        for i, msg in enumerate(example["messages"]):
            role = "user" if i % 2 == 0 else "assistant"
            text += f"<|im_start|>{role}\n{msg['content']}<|im_end|>\n"

        result = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    
    # Wende Formatierung an
    tokenized_dataset = dataset.map(
        format_chatml,
        remove_columns=dataset["train"].column_names,  # Entferne Original-Spalten
        batched=False  # Nicht batched, da wir variable Längen haben
    )
    
    return tokenized_dataset

# ============================
# 5. TRAINING ARGUMENTE
# ============================
def setup_training_args():
    """Konfiguriert TrainingArguments für optimalen VRAM-Verbrauch."""
    return TrainingArguments(
        # Output-Verzeichnis
        output_dir=str(OUTPUT_DIR),
        
        # Training-Länge
        num_train_epochs=NUM_EPOCHS,
        max_steps=-1,  # -1 = nutze epochs
        
        # Batch-Größe & Akkumulation
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        
        # Learning Rate
        learning_rate=LEARNING_RATE,
        warmup_steps=100,  # Sanfter Start
        lr_scheduler_type="cosine",  # Cosine-Annealing
        
        # Optimizer
        optim="adamw_bnb_8bit",# ← WICHTIG: 8-Bit Optimizer spart 30% VRAM
        
        # Logging & Saving
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,  # Nur die letzten 2 Checkpoints behalten
        
        # VRAM-Optimierungen
        fp16=True,  # Mixed Precision Training
        gradient_checkpointing=True,  # ← WICHTIG: Spare 30% VRAM, 20% langsamer
        report_to="none",  # Deaktiviere wandb/comet
        
        # Kontroll-Parameter
        dataloader_pin_memory=False,  # Spare VRAM
        remove_unused_columns=True,
    )

# ============================
# 6. TRAINER & TRAINING
# ============================
def main():
    """Haupt-Training-Loop."""
    # 1. Lade Modell & Tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # 2. Setup LoRA
    model = setup_lora(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # 3. Lade Daten
    dataset = load_and_format_data(tokenizer)
    
    # 4. Training Arguments
    training_args = setup_training_args()
    
    # 5. Erstelle Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )
    
    # 6. Starte Training
    print("🚀 Starte Training...")
    trainer.train()
    
    # 7. Speichere Modell
    print("💾 Speichere Model...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    
    print(f"✅ Training abgeschlossen! Modell in {OUTPUT_DIR}")

# ============================
# AUSFÜHRUNG
# ============================
if __name__ == "__main__":
    main()