# src/kitten_model.py
"""
HOPE Model Wrapper mit Deep Optimizer Support.

INFINITE CONTEXT VERSION:
- Deprecation Warning für torch_dtype behoben
- Memory-Persistenz über Sessions hinweg
"""

from numpy import zeros
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from kitten_lora import HOPELoRALayer, HOPEConfig, MemoryState


class HOPEModel(nn.Module):
    """
    Wrapper der HOPE-LoRA auf ein Qwen/LLaMA-Style Modell anwendet.
    Unterstützt Deep Optimizer Integration und Infinite Context.
    """
    
    CACHE_DIR = Path(__file__).parent.parent / "cache"

    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-0.6B",  # Fallback auf korrekten Pfad
        config: HOPEConfig = None,
        target_modules: List[str] = None,
        device_map: str = "auto",
        dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        
        # ═════════════════════════════════════════════════════════════════
        # 1. ROBUSTER PATH HANDLING (Fix für models-- Fehler)
        # ═════════════════════════════════════════════════════════════════════
        # Wir gehen davon aus, dass das Skript in `src/` liegt.
        base_dir = Path(__file__).parent.parent.resolve()  # .../Kitten-LoRA
        cache_dir = base_dir / "cache"
        
        # Falls model_id versehentlich falsch ist (z.B. "models/Qwen..."):
        if "models/" in model_id:
            print(f"⚠️ Warnung: model_id enthält 'models/' Pfad. Korrigiere zu 'Qwen/Qwen3-0.6B'.")
            model_id = "Qwen/Qwen3-0.6B"
        
        # ═══════════════════════════════════════════════════════════════════
        # 2. SETUP
        # ═══════════════════════════════════════════════════════════════════════════
        self.config = config or HOPEConfig()
        self.target_modules = target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # ═══════════════════════════════════════════════════════════════════════════
        # 3. MODEL LOADING
        # ═══════════════════════════════════════════════════════════════════════════
        print(f"🔧 Lade Tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=str(cache_dir),
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"🚀 Lade Modell: {model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=str(cache_dir),
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=dtype,
            attn_implementation="eager",
            use_cache=True,
        )
        
        # ═════════════════════════════════════════════════════════════════════════
        # 4. HOPE LoRA INJECTION
        # ════════════════════════════════════════════════════════════════════════════════
        self.hope_layers: List[HOPELoRALayer] = []
        self._apply_hope_lora()
        
        self.deep_optimizer = None
        
        print(f"📊 196 Layer durch HOPE-LoRA ersetzt")
        print(f"✅ HOPE-Modell bereit mit {len(self.hope_layers)} HOPE-Layern")
    
    def _apply_hope_lora(self):
        """Ersetzt Linear-Layer durch HOPE-LoRA."""
        replaced = 0
        
        def replace_recursive(parent: nn.Module, prefix: str = ""):
            nonlocal replaced
            
            for name, child in list(parent.named_children()):
                full_name = f"{prefix}.{name}" if prefix else name
                
                if isinstance(child, nn.Linear):
                    if any(t in name for t in self.target_modules):
                        hope_layer = HOPELoRALayer(child, self.config)
                        setattr(parent, name, hope_layer)
                        self.hope_layers.append(hope_layer)
                        replaced += 1
                        if replaced <= 7 or replaced % 28 == 0:
                            print(f"   ✓ {full_name}")
                else:
                    replace_recursive(child, full_name)
        
        replace_recursive(self.model)
        print(f"📊 {replaced} Layer durch HOPE-LoRA ersetzt")
    
    def set_deep_optimizer(self, deep_optimizer):
        """Setzt den Deep Optimizer."""
        self.deep_optimizer = deep_optimizer
        print("🧠 Deep Optimizer verbunden")
    
    def reset_memory(self, batch_size: int = 1):
        """
        Setzt Memory für alle HOPE-Layer zurück.
        
        ACHTUNG für Infinite Context:
        Nur am Anfang einer neuen "Session" aufrufen,
        NICHT während des Trainings!
        """
        # ════════════════════════════════════════════════════════════════════
        # 1. Robuste Geräte-Erkennung (fixt deine Pfad-Probleme)
        # ══════════════════════════════════════════════════════════════════════
        # Wir nutzen next(...).device, da HOPEModel ein Wrapper ist.
        try:
            device = next(self.model.parameters()).device
            dtype = next(self.model.parameters()).dtype
        except StopIteration:
            print("⚠️ Keine Parameter im Modell gefunden. Abbruch.")
            return

        if batch_size is None:
            batch_size = 1
            
        # ══════════════════════════════════════════════════════════════════════
        # 2. Erzeugen eines neuen Memory States (WICHTIG)
        # ═══════   ═════════════════      ══════════════════════════════════════════════════════════════════════════
        # Wir bauen ein frisches Objekt, um "veraltete" Referenzen zu vermeiden.
        new_state = MemoryState()
        new_state.batch_size = batch_size
        new_state.device = device
        new_state.dtype = dtype
        new_state.step = 0
        
        # Initialisierung der Tensoren (ohne .grad() zu nutzen, da sie neu sind)
        new_state.fast = torch.zeros(batch_size, self.config.r_fast, device=device, dtype=dtype) 
        new_state.medium = torch.zeros(batch_size, self.config.r_medium, device=device, dtype=dtype) 
        new_state.slow = torch.zeros(batch_size, self.config.r_slow, device=device, dtype=dtype) 
        new_state.medium_accum = torch.zeros(batch_size, self.config.r_medium, device=device, dtype=dtype)
        new_state.slow_accum = torch.zeros(batch_size, self.config.r_slow, device=device, dtype=dtype)
        
        # ═════════════════ Referenz auf die Klasse (MemoryState)
        # ═══════════════════════.zeros(batch_size, self.config.r_slow, device=device, device=dtype)
        
        # ═════       ════════════════════
        # ════════════════════════     # Wir weisen das frische State-Objekt jedem Layer zu.
        #      # Wir weisen das frische State-Objekt jedem Layer zu.
        
        for layer in self.hope_layers:
            #         # Wir weisen das frische State-Objekt jedem Layer zu.
            layer._memory_state = new_state
            #         # Wir weisen das frische *shared* State-Objekt jedem Layer zu.
            #         # Wir weisen das frische *shared* State-Objekt jedem Layer zu.
            #             # (Dies ist wichtig für Nested Learning: Alle Layer teilen sich einen Memory-Zustand,
            #             #  dies ist wichtig für Nested Learning: Alle Layer teilen sich einen Memory-Zustand,
            #             # aber mit verschiedenen Frequen.)
            #              *aber mit verschiedenen Frequen.)
    
    def set_memory_update(self, enabled: bool):
        """Aktiviert/deaktiviert Memory-Updates."""
        for layer in self.hope_layers:
            layer._update_memory = enabled
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Aggregierte Memory-Statistiken."""
        if not self.hope_layers:
            return {"status": "no_hope_layers"}
        
        stats = {
            "num_layers": len(self.hope_layers),
            "total_steps": 0,
            "fast_norm_avg": 0.0,
            "medium_norm_avg": 0.0,
            "slow_norm_avg": 0.0,
        }
        
        for layer in self.hope_layers:
            layer_stats = layer.get_memory_stats()
            stats["total_steps"] = max(stats["total_steps"], layer_stats["step"])
            stats["fast_norm_avg"] += layer_stats["fast_norm"]
            stats["medium_norm_avg"] += layer_stats["medium_norm"]
            stats["slow_norm_avg"] += layer_stats["slow_norm"]
        
        n = len(self.hope_layers)
        stats["fast_norm_avg"] /= n
        stats["medium_norm_avg"] /= n
        stats["slow_norm_avg"] /= n
        
        return stats
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Forward pass."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        reset_memory: bool = False,
        show_memory_stats: bool = False,
        **kwargs
    ) -> str:
        """
        Generiert Text mit persistentem Memory.
        
        Args:
            reset_memory: Ob Memory zurückgesetzt werden soll.
                          Für Infinite Context auf False setzen!
        """
        if reset_memory:
            self.reset_memory(1)
        
        device = next(self.model.parameters()).device

        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
        ).to(device)
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["do_sample"] = False
        
        output_ids = self.model.generate(
            **inputs,
            **gen_kwargs,
        )
        
        response = self.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        if show_memory_stats:
            stats = self.get_memory_stats()
            print(f"📊 Memory: F={stats['fast_norm_avg']:.3f} M={stats['medium_norm_avg']:.3f} S={stats['slow_norm_avg']:.3f} | Steps={stats['total_steps']}")
        
        return response
    
    @torch.no_grad()
    def generate_streaming(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        reset_memory: bool = True,
    ):
        """Generator für Token-by-Token Streaming."""
        if reset_memory:
            self.reset_memory(1)
        
        device = next(self.model.parameters()).device
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(device)
        
        generated = inputs.input_ids
        
        for _ in range(max_new_tokens):
            outputs = self.model(generated)
            logits = outputs.logits[:, -1, :]
            
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            token_str = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            yield token_str
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
    
    def save_hope_weights(self, path: str):
        """Speichert HOPE-LoRA Gewichte."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        state_dict = {}
        for i, layer in enumerate(self.hope_layers):
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    state_dict[f"layer_{i}.{name}"] = param.data.cpu()
        
        torch.save(state_dict, path / "hope_lora.pt")
        torch.save(self.config, path / "hope_config.pt")
        print(f"💾 HOPE-Gewichte gespeichert: {path}")
    
    def load_hope_weights(self, path: str):
        """Lädt HOPE-LoRA Gewichte."""
        path = Path(path)
        
        weights_file = path / "hope_lora.pt"
        if not weights_file.exists():
            print(f"⚠️ Keine Gewichte gefunden: {weights_file}")
            return False
        
        state_dict = torch.load(weights_file, map_location="cpu", weights_only=False)
        
        loaded = 0
        for i, layer in enumerate(self.hope_layers):
            for name, param in layer.named_parameters():
                key = f"layer_{i}.{name}"
                if key in state_dict:
                    param.data = state_dict[key].to(param.device, param.dtype)
                    loaded += 1
        
        print(f"📂 HOPE-Gewichte geladen: {loaded} Tensoren von {path}")
        return True
    
    def load_deep_optimizer(self, path: str):
        """Lädt Deep Optimizer State."""
        path = Path(path)
        opt_file = path / "deep_optimizer.pt"
        
        if opt_file.exists() and self.deep_optimizer is not None:
            self.deep_optimizer.load(str(opt_file))
            print(f"🧠 Deep Optimizer geladen: {opt_file}")
            return True
        return False
    
    def save_memory_state(self, path: str):
        """
        Speichert den aktuellen Memory-State aller HOPE-Layer.
        Nützlich für Infinite Context: Session fortsetzen.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        memory_states = {}
        for i, layer in enumerate(self.hope_layers):
            if layer._memory_state is not None:
                state = layer._memory_state
                memory_states[f"layer_{i}"] = {
                    "fast": state.fast.cpu() if state.fast is not None else None,
                    "medium": state.medium.cpu() if state.medium is not None else None,
                    "slow": state.slow.cpu() if state.slow is not None else None,
                    "step": state.step,
                }
        
        torch.save(memory_states, path / "memory_state.pt")
        print(f"💾 Memory State gespeichert: {path}")
    
    def load_memory_state(self, path: str):
        """
        Lädt einen gespeicherten Memory-State.
        Nützlich für Infinite Context: Session fortsetzen.
        """
        path = Path(path)
        memory_file = path / "memory_state.pt"
        
        if not memory_file.exists():
            print(f"⚠️ Kein Memory State gefunden: {memory_file}")
            return False
        
        memory_states = torch.load(memory_file, map_location="cpu", weights_only=False)
        
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        
        for i, layer in enumerate(self.hope_layers):
            key = f"layer_{i}"
            if key in memory_states:
                saved = memory_states[key]
                
                # Initialisiere falls nötig
                if layer._memory_state is None:
                    layer.reset_memory(1, device, dtype)
                
                state = layer._memory_state
                if saved["fast"] is not None:
                    state.fast = saved["fast"].to(device, dtype)
                if saved["medium"] is not None:
                    state.medium = saved["medium"].to(device, dtype)
                if saved["slow"] is not None:
                    state.slow = saved["slow"].to(device, dtype)
                state.step = saved["step"]
        
        print(f"📂 Memory State geladen: {path}")
        return True