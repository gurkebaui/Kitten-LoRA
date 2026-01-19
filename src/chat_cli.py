#!/usr/bin/env python3
# src/chat_cli.py
"""
Kitten-LoRA CLI Chat mit Deep Optimizer Support.
"""

import sys
from pathlib import Path
from typing import Optional

import torch
from kitten_model import HOPEModel
from kitten_lora import HOPEConfig


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


class KittenCLI:
    """Interaktive CLI für Kitten-LoRA."""
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-0.6B",
        weights_path: Optional[str] = None,
        config: Optional[HOPEConfig] = None,
    ):
        self.config = config or HOPEConfig(
            r_fast=4,
            r_medium=16,
            r_slow=32,
            hidden_dim=32,
        )
        
        print(f"\n{Colors.HEADER}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}🐱 KITTEN-LoRA Chat{Colors.RESET}")
        print(f"{Colors.HEADER}{'='*60}{Colors.RESET}\n")
        
        self.model = HOPEModel(
            model_id=model_id,
            config=self.config,
            cache_dir="./cache",
        )
        
        # Gewichte laden
        self._load_weights(weights_path)
        
        self.history = []
        self.temperature = 0.7
        self.max_tokens = 256
        
        self.commands = {
            "/help": self.cmd_help,
            "/clear": self.cmd_clear,
            "/memory": self.cmd_memory,
            "/temp": self.cmd_temperature,
            "/quit": self.cmd_quit,
            "/exit": self.cmd_quit,
        }
    
    def _load_weights(self, weights_path: Optional[str]):
        """Lädt trainierte Gewichte."""
        paths_to_try = []
        
        if weights_path:
            paths_to_try.append(Path(weights_path))
        
        paths_to_try.extend([
            Path("models/kitten_deep/best"),
            Path("models/kitten_hope/best"),
            Path("models/kitten_deep/final"),
        ])
        
        for path in paths_to_try:
            if path.exists() and (path / "hope_lora.pt").exists():
                if self.model.load_hope_weights(str(path)):
                    print(f"{Colors.GREEN}✅ Gewichte geladen: {path}{Colors.RESET}")
                    return
        
        print(f"{Colors.YELLOW}⚠️ Keine trainierten Gewichte - Vanilla HOPE{Colors.RESET}")
    
    def cmd_help(self, args: str = ""):
        print(f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════╗
║                    COMMANDS                                ║
╠══════════════════════════════════════════════════════════╣
║  /help          Show this help                             ║
║  /clear         Reset memory                               ║
║  /memory        Show memory state                          ║
║  /temp [0-2]    Set temperature (current: {self.temperature:.1f})              ║
║  /quit          Exit                                       ║
╚══════════════════════════════════════════════════════════╝{Colors.RESET}
""")
    
    def cmd_clear(self, args: str = ""):
        self.model.reset_memory(1)
        self.history = []
        print(f"{Colors.GREEN}🔄 Memory cleared!{Colors.RESET}\n")
    
    def cmd_memory(self, args: str = ""):
        stats = self.model.get_memory_stats()
        print(f"""
{Colors.CYAN}╔═══════════════════════════════════════╗
║         MEMORY STATE                  ║
╠═══════════════════════════════════════╣
║  Steps:  {stats['total_steps']:>28} ║
║  Fast:   {stats['fast_norm_avg']:>28.4f} ║
║  Medium: {stats['medium_norm_avg']:>28.4f} ║
║  Slow:   {stats['slow_norm_avg']:>28.4f} ║
╚═══════════════════════════════════════╝{Colors.RESET}
""")
    
    def cmd_temperature(self, args: str):
        try:
            temp = float(args.strip())
            if 0 <= temp <= 2:
                self.temperature = temp
                print(f"{Colors.GREEN}✓ Temperature: {temp}{Colors.RESET}")
            else:
                print(f"{Colors.RED}Must be 0-2{Colors.RESET}")
        except ValueError:
            print(f"{Colors.YELLOW}Current: {self.temperature}{Colors.RESET}")
    
    def cmd_quit(self, args: str = ""):
        print(f"\n{Colors.CYAN}👋 Bye! 🐱{Colors.RESET}\n")
        sys.exit(0)
    
    def run(self):
        self.cmd_help()
        print(f"{Colors.DIM}Memory persists until /clear{Colors.RESET}\n")
        
        self.model.reset_memory(1)
        
        while True:
            try:
                user_input = input(f"{Colors.GREEN}You:{Colors.RESET} ").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    cmd = parts[0].lower()
                    args = parts[1] if len(parts) > 1 else ""
                    
                    if cmd in self.commands:
                        self.commands[cmd](args)
                        continue
                    else:
                        print(f"{Colors.RED}Unknown command{Colors.RESET}")
                        continue
                
                prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
                
                print(f"\n{Colors.BLUE}Kitten:{Colors.RESET} ", end='', flush=True)
                
                full_response = ""
                try:
                    for token in self.model.generate_streaming(
                        prompt,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        reset_memory=False,
                    ):
                        print(token, end='', flush=True)
                        full_response += token
                except KeyboardInterrupt:
                    print(f"\n{Colors.YELLOW}[Interrupted]{Colors.RESET}")
                
                print()
                
                self.history.append({"user": user_input, "assistant": full_response})
                
                stats = self.model.get_memory_stats()
                print(f"{Colors.DIM}[Mem: F={stats['fast_norm_avg']:.2f} M={stats['medium_norm_avg']:.2f} S={stats['slow_norm_avg']:.2f}]{Colors.RESET}\n")
                
            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}(/quit to exit){Colors.RESET}")
            except EOFError:
                self.cmd_quit()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Kitten-LoRA Chat")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args()
    
    cli = KittenCLI(
        model_id=args.model,
        weights_path=args.weights,
    )
    cli.run()


if __name__ == "__main__":
    main()