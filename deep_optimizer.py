# src/deep_optimizer.py
"""
Deep Optimizer für Kitten-LoRA / HOPE.

STABLE VERSION:
- Gradient Safety Valve (Clipping) verhindert NaN/Inf Abstürze
- Robuste Feature-Extraktion
- State Detaching gegen RAM-Leak
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class DeepOptimizerConfig:
    """
    Konfiguration für den Deep Optimizer.
    
    STABLE DEFAULTS:
    - meta_lr sehr klein (1e-5) für Robustheit
    - Features werden intern geclipped
    """
    hidden_dim: int = 128
    num_levels: int = 3
    level_embed_dim: int = 16
    meta_lr: float = 1e-4 # Sehr klein für maximale Stabilität (war 5e-5)
    grad_clip: float = 1.0
    update_reg_weight: float = 0.1
    surprise_threshold: float = -1.0 # Deaktiviert Filterung (war 0.5)
    consolidation_interval: int = 200
    slow_consolidation_interval: int = 1000
    svd_rank_ratio: float = 0.5

    max_update_norm: float = 0.1 # Strengeres Clipping für Stabilität (war 0.05)
    momentum_beta: float = 0.9
    state_detach_interval: int = 100


class DeepOptimizerNetwork(nn.Module):
    """Deep Optimizer Network."""
    
    def __init__(self, config: DeepOptimizerConfig):
        super().__init__()
        self.config = config
        
        self.level_embedding = nn.Embedding(config.num_levels, config.level_embed_dim)
        
        self.grad_encoder = nn.Sequential(
            nn.Linear(8, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        
        self.lstm = nn.LSTMCell(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
        )
        
        combined_dim = config.hidden_dim + config.level_embed_dim
        
        self.lr_head = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        self.momentum_head = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        self.surprise_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        self.momentum_buffers: Dict[str, torch.Tensor] = {}
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)# Kleiner gain für Stabilität
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)


    
    
    def extract_grad_features(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Extrahiert statistische Features - MIT NaN/INF SCHUTZ.
        """
        if grad is None or grad.numel() == 0:
            device = next(self.parameters()).device
            return torch.zeros(8, device=device)
        
        with torch.no_grad():
            flat = grad.flatten().float()
            
            # 🔧 FIX 1: Gradient Clipping (Safety Valve)
            # Wir clippen die Gradienten auf [-10, 10] VOR der Berechnung.
            # Das verhindert Inf-Werte in den Features und damit im Netzwerk.
            flat = torch.clamp(flat, min=-10.0, max=10.0)
            
            mean = flat.mean()
            std = flat.std() + 1e-8
            
            # 🔧 FIX 2: NaN/Inf Check auf Stats
            if torch.isnan(std) or torch.isinf(std):
                std = torch.tensor(1.0, device=flat.device)
            if torch.isnan(mean) or torch.isinf(mean):
                mean = torch.tensor(0.0, device=flat.device)
            
            norm = torch.norm(flat) / (flat.numel() ** 0.5 + 1e-8)
            min_val = flat.min()
            max_val = flat.max()
            
            if torch.isnan(norm): norm = torch.tensor(0.0, device=flat.device)
            if torch.isnan(min_val): min_val = torch.tensor(-10.0, device=flat.device)
            if torch.isnan(max_val): max_val = torch.tensor(10.0, device=flat.device)
            
            pos_ratio = (flat > 0).float().mean()
            small_ratio = (flat.abs() < 1e-6).float().mean()
            
            features = torch.stack([
                mean,
                std,
                flat.abs().mean(),
                norm,
                min_val,
                max_val,
                pos_ratio,
                small_ratio,
            ])
            
            # 🔧 FIX 3: Feature Clipping
            # Wir clippen auch den Feature-Vektor final, um das LSTM sicher zu halten.
            features = torch.clamp(features, -5.0, 5.0)
        
        return features
    
    def forward(
        self,
        grad_features: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
        level_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass.
        """
        # Detachen der Inputs (bereits in deinem Code, behalten wir)
        h_in, c_in = hidden_state
        h_detached = h_in.detach() if h_in is not None else None
        c_detached = c_in.detach() if c_in is not None else None
        
        # Encode
        encoded = self.grad_encoder(grad_features)
        
        # LSTM Update
        h_new, c_new = self.lstm(encoded, (h_detached, c_detached))
        
        # Level Context
        tau = self.level_embedding(level_idx)
        combined = torch.cat([h_new, tau], dim=-1)
        
        # Outputs
        learned_lr = self.lr_head(combined) * self.config.max_update_norm
        learned_momentum = self.momentum_head(combined)
        surprise = self.surprise_head(h_new)
        
        return learned_lr, learned_momentum, surprise, (h_new, c_new)
    
    def init_hidden(self, batch_size: int, device: torch.device):
        h = torch.zeros(batch_size, self.config.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.config.hidden_dim, device=device)
        return (h, c)
    
    def get_or_init_momentum(self, key: str, shape: torch.Size, device: torch.device) -> torch.Tensor:
        if key not in self.momentum_buffers:
            self.momentum_buffers[key] = torch.zeros(shape, device=device)
        return self.momentum_buffers[key]
    
    def clear_momentum_buffers(self):
        self.momentum_buffers.clear()


    def freeze_lr_head(self):
        """Friert den LR Head ein, um 'Lazy Disease' zu verhindern."""
        print("🔒 Freezing LR Head (Manual Safe Mode)")
        for p in self.lr_head.parameters():
            p.requires_grad = False


class SVDConsolidator:
    """SVD-basierte Wissenskonsolidierung (Korrigiert für Infinite Context)."""
    
    def __init__(self, config: DeepOptimizerConfig):
        self.config = config
        self.step = 0
    
    def should_consolidate_fast_to_medium(self) -> bool:
        return self.step > 0 and self.step % self.config.consolidation_interval == 0
    
    def should_consolidate_medium_to_slow(self) -> bool:
        return self.step > 0 and self.step % self.config.slow_consolidation_interval == 0
    
    @torch.no_grad()
    def consolidate_weights(
        self,
        source_params: List[torch.Tensor],
        target_params: List[torch.Tensor],
        transfer_rate: float = 0.05,
    ):
        for src, tgt in zip(source_params, target_params):
            if src.shape != tgt.shape:
                continue
            
            try:
                if src.dim() == 1:
                    tgt.add_(transfer_rate * src)
                    src.mul_(0.9)
                else:
                    U, S, Vh = torch.linalg.svd(src, full_matrices=False)
                    k = max(1, int(len(S) * self.config.svd_rank_ratio))
                    compressed = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
                    tgt.add_(transfer_rate * compressed)
                    src.mul_(0.9)
            except Exception:
                tgt.add_(transfer_rate * src * 0.1)
                src.mul_(0.9)


class DeepOptimizerManager:
    """Verwaltet den Deep Optimizer."""
    
    def __init__(
        self,
        config: DeepOptimizerConfig = None,
        device: torch.device = None,
    ):
        self.config = config or DeepOptimizerConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.optimizer_net = DeepOptimizerNetwork(self.config).to(self.device)
        
        self.meta_optimizer = torch.optim.AdamW(
            self.optimizer_net.parameters(),
            lr=self.config.meta_lr,
            weight_decay=0.01,
        )
        
        self.consolidator = SVDConsolidator(self.config)
        self.layer_states: Dict[int, Dict] = {}
        self.pending_updates: Dict[str, torch.Tensor] = {}
        
        self.global_step = 0
        self.stats = {
            "meta_loss": [],
            "surprise_avg": [],
            "lr_avg": [],
            "momentum_avg": [],
            "consolidations": 0,
        }
        self._max_history = 100
    
    def init_layer_state(self, layer_idx: int, batch_size: int = 1):
        self.layer_states[layer_idx] = {
            "fast_hidden": self.optimizer_net.init_hidden(batch_size, self.device),
            "medium_hidden": self.optimizer_net.init_hidden(batch_size, self.device),
            "slow_hidden": self.optimizer_net.init_hidden(batch_size, self.device),
        }
    
    def _detach_all_states(self):
        for layer_idx in self.layer_states:
            state = self.layer_states[layer_idx]
            for key in ["fast_hidden", "medium_hidden", "slow_hidden"]:
                h, c = state[key]
                state[key] = (h.detach(), c.detach())



    def force_lr_output(self, target_lr: float = 0.5):
        """
        Setzt die Gewichte des LR Heads so, dass der Output = target_lr ist.
        Sigmoid(x) = target_lr  =>  x = log(target_lr / (1 - target_lr))
        """
        print(f"🔧 Forcing Deep Opt LR Head to {target_lr}...")
        # Sigmoid inverse Berechnung
        bias_input = math.log(target_lr / (1.0 - target_lr))
        
        # Wir gehen durch das Modul "lr_head"
        # (Es ist ein Sequential Block: Linear -> GELU -> Linear -> Sigmoid)
        # Wir initialisieren das Ganze sehr klein, damit der Bias dominiert
        for m in self.optimizer_net.lr_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001) # Winzige Gewichtungen
                with torch.no_grad():
                    m.bias.fill_(bias_input) # Bias steuert das Ergebnis
        
        # Und dann frieren wir es sofort ein
        self.optimizer_net.freeze_lr_head()


    # In class DeepOptimizerManager:

    def compute_differentiable_update(
        self,
        hope_layers: List,
        detach_grads: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Berechnet Updates MIT Gradient-Flow für Meta-Learning.
        
        Returns:
            deltas: Dict mit differenzierbaren Update-Tensoren
            infos: Dict mit Logging-Infos (detached)
        """
        deltas = {}
        lr_tensors = {}  # Für Meta-Loss
        
        for i, layer in enumerate(hope_layers):
            # Hole Gradienten (detach vom Hauptgraph, aber behalte Werte)
            if layer.proj_down.weight.grad is not None:
                grad_down = layer.proj_down.weight.grad
                if detach_grads:
                    grad_down = grad_down.detach()
                
                # Feature Extraction (muss differenzierbar sein!)
                features = self._extract_features_differentiable(grad_down)
                
                # LSTM Forward (differenzierbar!)
                if i not in self.layer_states:
                    self.init_layer_state(i)
                
                state = self.layer_states[i]
                level_tensor = torch.tensor([0], device=self.device)  # Fast level
                
                lr, momentum, surprise, new_hidden = self.optimizer_net(
                    features.unsqueeze(0),
                    state["fast_hidden"],
                    level_tensor,
                )
                state["fast_hidden"] = (new_hidden[0].detach(), new_hidden[1].detach())
                
                # Differenzierbares Delta
                # lr ist ein Tensor mit Gradient-Verbindung zum LSTM!
                delta = -lr.squeeze() * grad_down
                
                # Clipping (differenzierbar)
                delta_norm = delta.norm() + 1e-8
                scale = torch.clamp(self.config.max_update_norm / delta_norm, max=1.0)
                delta = delta * scale
                
                deltas[f"{i}_proj_down"] = delta
                lr_tensors[f"{i}_proj_down"] = lr.squeeze()
            
            if layer.proj_up.weight.grad is not None:
                grad_up = layer.proj_up.weight.grad
                if detach_grads:
                    grad_up = grad_up.detach()
                
                features = self._extract_features_differentiable(grad_up)
                level_tensor = torch.tensor([0], device=self.device)
                
                lr, momentum, surprise, new_hidden = self.optimizer_net(
                    features.unsqueeze(0),
                    state["fast_hidden"],
                    level_tensor,
                )
                
                delta = -lr.squeeze() * grad_up
                delta_norm = delta.norm() + 1e-8
                scale = torch.clamp(self.config.max_update_norm / delta_norm, max=1.0)
                delta = delta * scale
                
                deltas[f"{i}_proj_up"] = delta
                lr_tensors[f"{i}_proj_up"] = lr.squeeze()
        
        return deltas, lr_tensors


    def _extract_features_differentiable(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Feature-Extraktion OHNE torch.no_grad().
        Einfachere Version für Gradient-Flow.
        """
        flat = grad.flatten().float()
        
        # Clipping für Stabilität
        flat = torch.clamp(flat, -10.0, 10.0)
        
        # Differenzierbare Statistiken
        mean = flat.mean()
        std = flat.std() + 1e-8
        norm = flat.norm() / (flat.numel() ** 0.5 + 1e-8)
        abs_mean = flat.abs().mean()
        
        # Pseudo-Features (konstant, aber Dimension muss stimmen)
        # Die ersten 4 sind differenzierbar, die letzten 4 sind Platzhalter
        features = torch.stack([
            mean,
            std,
            abs_mean,
            norm,
            torch.zeros_like(mean),  # Placeholder
            torch.zeros_like(mean),
            torch.zeros_like(mean),
            torch.zeros_like(mean),
        ])
        
        return torch.clamp(features, -5.0, 5.0)


    def apply_deltas_differentiable(
        self,
        hope_layers: List,
        deltas: Dict[str, torch.Tensor],
    ):
        """
        Wendet Updates an - IN-PLACE OHNE Referenzierung!
        """
        for i, layer in enumerate(hope_layers):
            key_down = f"{i}_proj_down"
            if key_down in deltas:
                # WICHTIG: .data.add_() statt Ersetzung
                # Wir ändern nur die Daten, das Parameter-Objekt bleibt gleich
                layer.proj_down.weight.data.add_(deltas[key_down])
            
            key_up = f"{i}_proj_up"
            if key_up in deltas:
                layer.proj_up.weight.data.add_(deltas[key_up])
        
    def compute_update_for_param(
        self,
        layer_idx: int,
        param_name: str,
        grad: torch.Tensor,
        level: str,
    ) -> Tuple[torch.Tensor, Dict]:
        if layer_idx not in self.layer_states:
            self.init_layer_state(layer_idx)
        
        state = self.layer_states[layer_idx]
        level_idx_map = {"fast": 0, "medium": 1, "slow": 2}
        level_idx = level_idx_map[level]
        hidden_key = f"{level}_hidden"
        
        features = self.optimizer_net.extract_grad_features(grad)
        features = features.unsqueeze(0)
        
        level_tensor = torch.tensor([level_idx], device=self.device)
        
        learned_lr, learned_momentum, surprise, new_hidden = self.optimizer_net(
            features, state[hidden_key], level_tensor
        )
        
        state[hidden_key] = new_hidden
        
        lr_val = learned_lr.item()
        momentum_val = learned_momentum.item()
        surprise_val = surprise.item()
        
        if surprise_val < self.config.surprise_threshold:
            lr_val *= 0.1
        
        momentum_key = f"{layer_idx}_{param_name}_{level}"
        momentum = self.optimizer_net.get_or_init_momentum(
            momentum_key, grad.shape, grad.device
        )
        
        with torch.no_grad():
            momentum.mul_(momentum_val).add_((1 - momentum_val) * grad)
        
        delta = -lr_val * momentum
        
        delta_norm = delta.norm()
        if delta_norm > self.config.max_update_norm:
            delta = delta * (self.config.max_update_norm / delta_norm)
        
        info = {
            "lr": lr_val,
            "momentum": momentum_val,
            "surprise": surprise_val,
            "delta_norm": delta.norm().item(),
        }
        
        return delta, info
    
    def compute_all_updates(self, hope_layers: List) -> Dict[str, torch.Tensor]:
        self.pending_updates.clear()
        all_info = {"lr": [], "momentum": [], "surprise": []}
        
        for i, layer in enumerate(hope_layers):
            if hasattr(layer, 'proj_down') and layer.proj_down.weight.grad is not None:
                delta, info = self.compute_update_for_param(
                    i, "proj_down", layer.proj_down.weight.grad, "fast"
                )
                self.pending_updates[f"{i}_proj_down"] = delta
                for k in all_info:
                    all_info[k].append(info[k])
            
            if hasattr(layer, 'proj_up') and layer.proj_up.weight.grad is not None:
                delta, info = self.compute_update_for_param(
                    i, "proj_up", layer.proj_up.weight.grad, "fast"
                )
                self.pending_updates[f"{i}_proj_up"] = delta
                for k in all_info:
                    all_info[k].append(info[k])
        
        for k in all_info:
            if all_info[k]:
                avg = sum(all_info[k]) / len(all_info[k])
                self.stats[f"{k}_avg"].append(avg)
        
        return self.pending_updates
    
    def compute_meta_loss(self, hope_layers: List, pending_updates: Dict[str, torch.Tensor]) -> torch.Tensor:
        meta_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for i, layer in enumerate(hope_layers):
            key = f"{i}_proj_down"
            if key in pending_updates and layer.proj_down.weight.grad is not None:
                grad = layer.proj_down.weight.grad
                delta = pending_updates[key]
                dot_product = torch.sum(grad * delta)
                meta_loss = meta_loss + dot_product
            
            key = f"{i}_proj_up"
            if key in pending_updates and layer.proj_up.weight.grad is not None:
                grad = layer.proj_up.weight.grad
                delta = pending_updates[key]
                dot_product = torch.sum(grad * delta)
                meta_loss = meta_loss + dot_product
        
        reg_loss = 0.0
        for delta in pending_updates.values():
            reg_loss += delta.norm() ** 2
        reg_loss = self.config.update_reg_weight * reg_loss
        
        meta_loss = meta_loss + reg_loss
        return meta_loss
    
    def apply_pending_updates(self, hope_layers: List):
        with torch.no_grad():
            for i, layer in enumerate(hope_layers):
                key = f"{i}_proj_down"
                if key in self.pending_updates:
                    layer.proj_down.weight.add_(self.pending_updates[key])
                
                key = f"{i}_proj_up"
                if key in self.pending_updates:
                    layer.proj_up.weight.add_(self.pending_updates[key])
        self.pending_updates.clear()
    
    def step(self, hope_layers: List, main_loss: torch.Tensor, do_meta_update: bool = True):
        self.global_step += 1
        self.consolidator.step = self.global_step
        
        pending = self.compute_all_updates(hope_layers)
        
        if do_meta_update and self.global_step % 10 == 0:
            try:
                meta_loss = self.compute_meta_loss(hope_layers, pending)
                self.meta_optimizer.zero_grad()
                meta_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.optimizer_net.parameters(), self.config.grad_clip)
                self.meta_optimizer.step()
                self.stats["meta_loss"].append(meta_loss.item())
            except RuntimeError as e:
                print(f"⚠️ Meta-Update Error: {e}")
            self._detach_all_states()
        
        self.apply_pending_updates(hope_layers)
        
        if self.consolidator.should_consolidate_fast_to_medium():
            self._do_consolidation(hope_layers, "fast", "medium")
        
        if self.consolidator.should_consolidate_medium_to_slow():
            self._do_consolidation(hope_layers, "medium", "slow")
        
        self._trim_stats()
    
    def _do_consolidation(self, hope_layers: List, source: str, target: str):
        print(f"   🔄 Consolidating: {source} → {target}")
        source_net = f"update_{source}"
        target_net = f"update_{target}"
        
        for layer in hope_layers:
            if hasattr(layer, source_net) and hasattr(layer, target_net):
                src_params = list(getattr(layer, source_net).parameters())
                tgt_params = list(getattr(layer, target_net).parameters())
                self.consolidator.consolidate_weights(src_params, tgt_params, transfer_rate=0.05)
        
        self.stats["consolidations"] += 1
    
    def _trim_stats(self):
        for key in ["meta_loss", "surprise_avg", "lr_avg", "momentum_avg"]:
            if key in self.stats and len(self.stats[key]) > self._max_history:
                self.stats[key] = self.stats[key][-self._max_history:]
    
    def get_stats(self) -> Dict:
        def safe_avg(lst):
            return sum(lst) / max(1, len(lst)) if lst else 0.0
        return {
            "global_step": self.global_step,
            "meta_loss_avg": safe_avg(self.stats.get("meta_loss", [])),
            "surprise_avg": safe_avg(self.stats.get("surprise_avg", [])),
            "lr_avg": safe_avg(self.stats.get("lr_avg", [])),
            "momentum_avg": safe_avg(self.stats.get("momentum_avg", [])),
            "consolidations": self.stats["consolidations"],
        }
    
    def reset(self):
        self.layer_states.clear()
        self.pending_updates.clear()
        self.optimizer_net.clear_momentum_buffers()
    
    def save(self, path: str):
        torch.save({
            "optimizer_net": self.optimizer_net.state_dict(),
            "meta_optimizer": self.meta_optimizer.state_dict(),
            "config": self.config,
            "global_step": self.global_step,
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.optimizer_net.load_state_dict(checkpoint["optimizer_net"])
        self.meta_optimizer.load_state_dict(checkpoint["meta_optimizer"])
        self.global_step = checkpoint["global_step"]