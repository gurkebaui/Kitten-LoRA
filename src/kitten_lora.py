# src/kitten_lora.py
"""
HOPE-LoRA: Hierarchical Online Parameter-Efficient Learning
Mit Newton-Schulz Stabilisierung für Infinite Context.

INFINITE CONTEXT VERSION:
- Höhere Ranks für mehr Speicherkapazität
- Langsamerer Decay für persistentes Memory
- Optimierte Chunk-Sizes für Nested Learning
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from dataclasses import dataclass, field
import math


# ═══════════════════════════════════════════════════════════
# Newton-Schulz Orthogonalisierung (Stabilität)
# ═══════════════════════════════════════════════════════════
def newton_schulz_(mat: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz Iteration für orthogonale Approximation.
    Hält die Memory-Updates stabil bei langen Kontexten.
    """
    original_shape = mat.shape
    original_dtype = mat.dtype
    
    if len(original_shape) == 1:
        mat = mat.unsqueeze(0).unsqueeze(0)
    elif len(original_shape) == 2:
        mat = mat.unsqueeze(0)
    
    mat = mat.float()
    
    b, n, d = mat.shape
    
    trace = (mat * mat).sum(dim=(-1, -2), keepdim=True).sqrt()
    mat = mat / (trace + eps)
    
    for _ in range(steps):
        mat_t = mat.transpose(-1, -2)
        prod = torch.matmul(mat, mat_t)
        term = torch.matmul(prod, mat)
        mat = 1.5 * mat - 0.5 * term
    
    mat = mat.view(original_shape).to(original_dtype)
    
    return mat


def stabilize_update(delta: torch.Tensor, use_newton_schulz: bool = True) -> torch.Tensor:
    """Stabilisiert ein Update-Signal."""
    if not use_newton_schulz:
        return torch.clamp(delta, -1.0, 1.0)
    
    if delta.dim() >= 2:
        return newton_schulz_(delta, steps=3)
    else:
        norm = delta.norm() + 1e-8
        if norm > 1.0:
            delta = delta / norm
        return delta


# ═══════════════════════════════════════════════════════════
# Memory State
# ═══════════════════════════════════════════════════════════
@dataclass
class MemoryState:
    """Rekurrenter Memory-State für CMS."""
    fast: Optional[torch.Tensor] = None
    medium: Optional[torch.Tensor] = None
    slow: Optional[torch.Tensor] = None
    medium_accum: Optional[torch.Tensor] = None
    slow_accum: Optional[torch.Tensor] = None
    step: int = 0
    batch_size: int = 1
    device: torch.device = field(default_factory=lambda: torch.device('cpu'))
    dtype: torch.dtype = field(default_factory=lambda: torch.float32)
    
    def reset(self):
        if self.fast is not None:
            self.fast.zero_()
        if self.medium is not None:
            self.medium.zero_()
        if self.slow is not None:
            self.slow.zero_()
        if self.medium_accum is not None:
            self.medium_accum.zero_()
        if self.slow_accum is not None:
            self.slow_accum.zero_()
        self.step = 0


# ═══════════════════════════════════════════════════════════
# HOPE Config - INFINITE CONTEXT DEFAULTS
# ═══════════════════════════════════════════════════════════
@dataclass
class HOPEConfig:
    """
    Konfiguration für HOPE-LoRA.
    
    INFINITE CONTEXT DEFAULTS:
    - Höhere Ranks (8/32/64 statt 4/16/32)
    - Langsamerer Decay (0.9995 statt 0.999)
    - Größere Chunks für langsamere Konsolidierung
    """
    # Memory Ranks (Kapazität)
    r_fast: int = 8       # Kurzzeitgedächtnis (war 4)
    r_medium: int = 32    # Mittelfristiges Gedächtnis (war 16)
    r_slow: int = 64      # Langzeitgedächtnis (war 32)
    
    # Chunk Sizes (Zeitskalen für Nested Learning)
    chunk_medium: int = 16  # Update alle 16 Tokens (war 8)
    chunk_slow: int = 64    # Update alle 64 Tokens (war 32)
    
    # LoRA Scaling
    alpha: float = 1.0
    
    # Surprise Threshold (wann ist etwas "neu genug" zum Speichern)
    surprise_threshold: float = -1.0  # Erhöht für selektiveres Lernen (war 0.3)
    
    # Learning Rates für Memory Updates (konservativer für Stabilität)
    lr_fast: float = 0.2    # (war 0.1)
    lr_medium: float = 0.05  # (war 0.05)
    lr_slow: float = 0.01    # (war 0.01)

    # Hidden Dimension für Update Networks
    hidden_dim: int = 64     # Größer für komplexere Muster (war 32)
    
    # Stabilisierung
    use_newton_schulz: bool = False  # (war True)
    
    # Memory Decay - KRITISCH für Infinite Context!
    # 0.9995 = Memory hält ~2000 Steps bevor es auf 37% fällt
    # 0.999  = Memory hält ~1000 Steps
    # 0.99   = Memory hält ~100 Steps (zu schnell!)
    memory_decay: float = 0.999  # (war 0.999)


# ═══════════════════════════════════════════════════════════
# HOPE LoRA Layer
# ═══════════════════════════════════════════════════════════
class HOPELoRALayer(nn.Module):
    """
    HOPE-Style LoRA Layer mit Continuum Memory System.
    Inkludiert Newton-Schulz Stabilisierung für Infinite Context.
    """
    
    def __init__(
        self,
        base_linear: nn.Linear,
        config: HOPEConfig = None,
    ):
        super().__init__()
        
        self.config = config or HOPEConfig()
        
        # Base Layer einfrieren
        self.base = base_linear
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
            
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        
        device = base_linear.weight.device
        dtype = base_linear.weight.dtype
        
        r_total = self.config.r_fast + self.config.r_medium + self.config.r_slow
        hidden = self.config.hidden_dim

        # Projektionen (trainiert vom Deep Optimizer)
        self.proj_down = nn.Linear(self.in_features, r_total, bias=False, device=device, dtype=dtype)
        self.proj_up = nn.Linear(r_total, self.out_features, bias=False, device=device, dtype=dtype)
        
        # Surprise Detector (trainiert vom Base Optimizer)
        self.surprise_net = nn.Sequential(
            nn.Linear(self.in_features, hidden, device=device, dtype=dtype),
            nn.LayerNorm(hidden, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden, 1, device=device, dtype=dtype),
            nn.Sigmoid(),
        )
        
        # Update Generators (trainiert vom Base Optimizer)
        self.update_fast = nn.Sequential(
            nn.Linear(self.in_features + self.config.r_fast, hidden, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden, self.config.r_fast, device=device, dtype=dtype),
            nn.Tanh(),
        )
        
        self.update_medium = nn.Sequential(
            nn.Linear(self.in_features + self.config.r_medium, hidden, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden, self.config.r_medium, device=device, dtype=dtype),
            nn.Tanh(),
        )
        
        self.update_slow = nn.Sequential(
            nn.Linear(self.in_features + self.config.r_slow, hidden, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden, self.config.r_slow, device=device, dtype=dtype),
            nn.Tanh(),
        )
        
        # Gate Network (trainiert vom Base Optimizer)
        self.gate_net = nn.Sequential(
            nn.Linear(self.in_features, hidden, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden, 3, device=device, dtype=dtype),
            nn.Softmax(dim=-1),
        )
        
        # Interner Memory State (PERSISTENT für Infinite Context!)
        self._memory_state: Optional[MemoryState] = None
        self._update_memory = True
        
        # Für Deep Optimizer
        self._last_input_features: Optional[torch.Tensor] = None
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.proj_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.proj_up.weight)
        
        for net in [self.update_fast, self.update_medium, self.update_slow]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def reset_memory(self, batch_size: int = None, device: torch.device = None, dtype: torch.dtype = None):
        """
        Setzt den internen Memory-State zurück.
        
        ACHTUNG: Für Infinite Context sollte dies nur am Anfang
        einer neuen "Session" aufgerufen werden, NICHT während des Trainings!
        """
        if device is None:
            device = self.proj_down.weight.device
        if dtype is None:
            dtype = self.proj_down.weight.dtype
        if batch_size is None:
            batch_size = 1
            
        self._memory_state = MemoryState()
        self._memory_state.batch_size = batch_size
        self._memory_state.device = device
        self._memory_state.dtype = dtype
        self._memory_state.fast = torch.zeros(batch_size, self.config.r_fast, device=device, dtype=dtype)
        self._memory_state.medium = torch.zeros(batch_size, self.config.r_medium, device=device, dtype=dtype)
        self._memory_state.slow = torch.zeros(batch_size, self.config.r_slow, device=device, dtype=dtype)
        self._memory_state.medium_accum = torch.zeros(batch_size, self.config.r_medium, device=device, dtype=dtype)
        self._memory_state.slow_accum = torch.zeros(batch_size, self.config.r_slow, device=device, dtype=dtype)
        self._memory_state.step = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - DROP-IN kompatibel mit nn.Linear."""
        original_shape = x.shape
        input_was_2d = (x.dim() == 2)
        
        if input_was_2d:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        
        if self._memory_state is None or self._memory_state.batch_size != batch_size:
            self.reset_memory(batch_size, device, dtype)
        
        state = self._memory_state
        if state.fast.dtype != dtype:
            state.fast = state.fast.to(dtype=dtype)
            state.medium = state.medium.to(dtype=dtype)
            state.slow = state.slow.to(dtype=dtype)
            state.medium_accum = state.medium_accum.to(dtype=dtype)
            state.slow_accum = state.slow_accum.to(dtype=dtype)
        
        # Speichere Input für Deep Optimizer
        self._last_input_features = x.detach().mean(dim=1)
        
        # Base Output
        base_out = self.base(x)
        
        # Low-Rank Projection
        proj = self.proj_down(x)
        
        r_f = self.config.r_fast
        r_m = self.config.r_medium
        proj_fast = proj[..., :r_f]
        proj_medium = proj[..., r_f:r_f+r_m]
        proj_slow = proj[..., r_f+r_m:]
        
        # Memory Modulation (Kernmechanismus von HOPE!)
        fast_exp = state.fast.unsqueeze(1).expand(-1, seq_len, -1)
        medium_exp = state.medium.unsqueeze(1).expand(-1, seq_len, -1)
        slow_exp = state.slow.unsqueeze(1).expand(-1, seq_len, -1)
        
        #scale_fast = torch.sigmoid(state.fast) * 1.0  # Bereich [0, 1] -> skaliert minimal
        #scale_medium = torch.sigmoid(state.medium) * 1.0
        #scale_slow = torch.sigmoid(state.slow) * 1.0

        #mod_fast = proj_fast * (1.0 + scale_fast)
        #mod_medium = proj_medium * (1.0 + scale_medium)
        #mod_slow = proj_slow * (1.0 + scale_slow)

        # ═══════════════════════════════════════════════════════════
        # old modulation (vorher)
        # ═══════════════════════════════════════════════════════════

        mod_fast = proj_fast * (1 + fast_exp)
        mod_medium = proj_medium * (1 + medium_exp)
        mod_slow = proj_slow * (1 + slow_exp)

        # ═══════════════════════════════════════════════════════════
        modulated = torch.cat([mod_fast, mod_medium, mod_slow], dim=-1)
        
        # Gated Output
        gates = self.gate_net(x)
        gate_scale = gates.mean(dim=-1, keepdim=True)
        
        lora_out = self.proj_up(modulated)
        output = base_out + self.config.alpha * lora_out * gate_scale
        
        # Memory Update (CMS - Continuum Memory System)
        if self._update_memory:
            self._do_memory_update(x, proj_fast, proj_medium, proj_slow)
        
        if input_was_2d:
            output = output.squeeze(1)
        
        return output
    
    def _do_memory_update(self, x, proj_fast, proj_medium, proj_slow):
        """
        Internes Memory-Update mit Newton-Schulz Stabilisierung.
        
        Implementiert das Continuum Memory System (CMS) aus dem Nested Learning Paper:
        - Fast: Updates jeden Token (Arbeitsgedächtnis)
        - Medium: Updates alle chunk_medium Tokens (Episodisches Gedächtnis)
        - Slow: Updates alle chunk_slow Tokens (Semantisches Gedächtnis)
        """
        state = self._memory_state
        batch_size, seq_len, _ = x.shape
        
        x_mean = x.mean(dim=1)
        state.step += seq_len
        
        with torch.no_grad():
            # ═══════════════════════════════════════════════════════
            # Memory Decay (Natürliches Vergessen - verhindert Explosion)
            # ═══════════════════════════════════════════════════════
            decay = self.config.memory_decay
            state.fast = state.fast * decay
            state.medium = state.medium * decay
            state.slow = state.slow * decay
            
            # ═══════════════════════════════════════════════════════
            # Surprise Detection (Was ist neu/wichtig?)
            # ═══════════════════════════════════════════════════════
            surprise = self.surprise_net(x_mean).squeeze(-1)
            surprise_gate = (surprise > self.config.surprise_threshold).float().unsqueeze(-1)
            
            # ═══════════════════════════════════════════════════════
            # Fast Memory Update (jeden Token - Arbeitsgedächtnis)
            # ═══════════════════════════════════════════════════════
            update_input = torch.cat([x_mean, state.fast], dim=-1)
            delta_fast = self.update_fast(update_input)
            
            # Newton-Schulz Stabilisierung
            delta_fast = stabilize_update(delta_fast, self.config.use_newton_schulz)
            
            # Nur updaten wenn "überraschend" genug
            state.fast = state.fast + surprise_gate * self.config.lr_fast * delta_fast
            
            # ═══════════════════════════════════════════════════════
            # 🔧 FIX 1: HARD CLIPPING der Memory States
            # Verhindert, dass F/M/S zu groß werden und das Modell destabilisieren
            # ═══════════════════════════════════════════════════════
            torch.clamp_(state.fast, -0.5, 0.5)    # Maximaler Skalierungsfaktor: 1.5x
            torch.clamp_(state.medium, -0.5, 0.5)
            torch.clamp_(state.slow, -0.5, 0.5)
            
            # ═══════════════════════════════════════════════════════
            # Medium Memory (alle chunk_medium Tokens - Episodisches Gedächtnis)
            # ═══════════════════════════════════════════════════════
            proj_medium_mean = proj_medium.mean(dim=1)
            state.medium_accum = state.medium_accum + proj_medium_mean
            
            if state.step % self.config.chunk_medium == 0:
                update_input = torch.cat([x_mean, state.medium], dim=-1)
                delta_medium = self.update_medium(update_input)
                compressed = state.medium_accum / max(1, self.config.chunk_medium)
                
                update_signal = delta_medium + 0.5 * compressed
                update_signal = stabilize_update(update_signal, self.config.use_newton_schulz)
                
                state.medium = state.medium + self.config.lr_medium * update_signal
                state.medium_accum.zero_()
            
            # ═══════════════════════════════════════════════════════
            # Slow Memory (alle chunk_slow Tokens - Semantisches Gedächtnis)
            # ═══════════════════════════════════════════════════════
            proj_slow_mean = proj_slow.mean(dim=1)
            state.slow_accum = state.slow_accum + proj_slow_mean
            
            if state.step % self.config.chunk_slow == 0:
                update_input = torch.cat([x_mean, state.slow], dim=-1)
                delta_slow = self.update_slow(update_input)
                compressed = state.slow_accum / max(1, self.config.chunk_slow)
                
                update_signal = delta_slow + 0.3 * compressed
                update_signal = stabilize_update(update_signal, self.config.use_newton_schulz)
                
                state.slow = state.slow + self.config.lr_slow * update_signal
                state.slow_accum.zero_()
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Gibt Memory-Statistiken zurück."""
        if self._memory_state is None:
            return {"fast_norm": 0, "medium_norm": 0, "slow_norm": 0, "step": 0}
        
        state = self._memory_state
        return {
            "fast_norm": state.fast.norm().item() if state.fast is not None else 0,
            "medium_norm": state.medium.norm().item() if state.medium is not None else 0,
            "slow_norm": state.slow.norm().item() if state.slow is not None else 0,
            "step": state.step,
        }
    
    def get_gradient_info(self) -> Dict[str, Optional[torch.Tensor]]:
        """Gibt Gradienten-Info für Deep Optimizer zurück."""
        return {
            "proj_down_grad": self.proj_down.weight.grad,
            "proj_up_grad": self.proj_up.weight.grad,
            "input_features": self._last_input_features,
        }