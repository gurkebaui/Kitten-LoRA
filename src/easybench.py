# src/benchmark_lite.py
"""
Kitten-LoRA Lite Benchmark - Angepasst für Deep Optimizer.
"""

from numpy import rint
import torch
import json
import time
import random
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from tqdm import tqdm

from kitten_model import HOPEModel
from kitten_lora import HOPEConfig

smol: bool = False  # Setze auf True für das kleine Modell (0.6B), False für 1.7B



@dataclass
class BenchmarkResult:
    test_name: str
    total_tokens: int
    beyond_context: bool
    accuracy: float
    details: Dict


class KittenBenchmarkLite:
    """Leichtes Benchmark für kleine Modelle (0.6B)."""

    
    
    def __init__(
        self,
        model: HOPEModel,
        context_window: int = 32768,
        output_dir: str = "benchmarks/results",
    ):
        self.model = model
        self.context_window = context_window
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
    
    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text))
    
    def _generate_filler(self, target_tokens: int) -> str:
        sentences = [
            "The weather is nice today. ",
            "Birds are singing outside. ",
            "The sun is shining bright. ",
            "Flowers bloom in spring. ",
            "Time passes quickly. ",
        ]
        filler = ""
        while self.count_tokens(filler) < target_tokens:
            filler += random.choice(sentences)
        return filler
    
    def _process_text(self, text: str):
        """Verarbeitet Text durch das Modell."""
        inputs = self.model.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=min(self.count_tokens(text) + 10, 4096),
        ).to(next(self.model.model.parameters()).device)
        
        with torch.no_grad():
            _ = self.model.model(**inputs)
    
    def test_number_memory(self, target_tokens: int = 35000, num_trials: int = 3) -> BenchmarkResult:
        """Einfachster Test: Merke dir eine Zahl."""
        print("\n" + "="*50)
        print("🔢 TEST 1: Simple Number Memory")
        print(f"   Target: {target_tokens:,} tokens ({target_tokens/self.context_window:.1f}x context)")
        print("="*50)
        
        correct = 0
        
        for trial in range(num_trials):
            secret = random.randint(1000, 9999)
            print(f"\n   Trial {trial+1}/{num_trials}: Secret = {secret}")
            
            self.model.reset_memory(1)
            
            intro = f"Remember this number: {secret}. This is very important. The number is {secret}."
            self._process_text(intro)
            
            current_tokens = self.count_tokens(intro)
            
            pbar = tqdm(total=target_tokens, initial=current_tokens, desc="   Processing")
            while current_tokens < target_tokens:
                filler = self._generate_filler(2000)
                self._process_text(filler)
                added = self.count_tokens(filler)
                current_tokens += added
                pbar.update(added)
            pbar.close()
            
            response = self.model.generate(
                "What number did I ask you to remember?",
                max_new_tokens=20,
                temperature=0,
                reset_memory=False,
            )
            
            found = str(secret) in response
            if found:
                correct += 1
                print(f"   ✅ Found: {secret}")
            else:
                print(f"   ❌ Expected {secret}, got: '{response[:50]}...'")
        
        return BenchmarkResult(
            test_name="number_memory",
            total_tokens=target_tokens,
            beyond_context=target_tokens > self.context_window,
            accuracy=correct / num_trials,
            details={"correct": correct, "trials": num_trials},
        )
    
    def test_color_memory(self, target_tokens: int = 35000, num_trials: int = 3) -> BenchmarkResult:
        """Merke dir eine Farbe."""
        print("\n" + "="*50)
        print("🎨 TEST 2: Color Memory")
        print("="*50)
        
        colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink"]
        correct = 0
        
        for trial in range(num_trials):
            color = random.choice(colors)
            print(f"\n   Trial {trial+1}/{num_trials}: Color = {color}")
            
            self.model.reset_memory(1)
            
            intro = f"My favorite color is {color}. I really love {color}. Remember: {color}."
            self._process_text(intro)
            
            current_tokens = self.count_tokens(intro)
            
            while current_tokens < target_tokens:
                filler = self._generate_filler(2000)
                self._process_text(filler)
                current_tokens += self.count_tokens(filler)
            
            response = self.model.generate(
                "What is my favorite color?",
                max_new_tokens=20,
                temperature=0,
                reset_memory=False,
            )
            
            found = color.lower() in response.lower()
            if found:
                correct += 1
                print(f"   ✅ Found: {color}")
            else:
                print(f"   ❌ Expected {color}, got: '{response[:50]}...'")
        
        return BenchmarkResult(
            test_name="color_memory",
            total_tokens=target_tokens,
            beyond_context=target_tokens > self.context_window,
            accuracy=correct / num_trials,
            details={"correct": correct, "trials": num_trials},
        )
    
    def test_name_memory(self, target_tokens: int = 35000, num_trials: int = 3) -> BenchmarkResult:
        """Merke dir einen Namen."""
        print("\n" + "="*50)
        print("👤 TEST 3: Name Memory")
        print("="*50)
        
        names = ["Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona", "George"]
        correct = 0
        
        for trial in range(num_trials):
            name = random.choice(names)
            print(f"\n   Trial {trial+1}/{num_trials}: Name = {name}")
            
            self.model.reset_memory(1)
            
            intro = f"My name is {name}. Hello, I am {name}. Please remember my name: {name}."
            self._process_text(intro)
            
            current_tokens = self.count_tokens(intro)
            
            while current_tokens < target_tokens:
                filler = self._generate_filler(2000)
                self._process_text(filler)
                current_tokens += self.count_tokens(filler)
            
            response = self.model.generate(
                "What is my name?",
                max_new_tokens=20,
                temperature=0,
                reset_memory=False,
            )
            
            found = name.lower() in response.lower()
            if found:
                correct += 1
                print(f"   ✅ Found: {name}")
            else:
                print(f"   ❌ Expected {name}, got: '{response[:50]}...'")
        
        return BenchmarkResult(
            test_name="name_memory",
            total_tokens=target_tokens,
            beyond_context=target_tokens > self.context_window,
            accuracy=correct / num_trials,
            details={"correct": correct, "trials": num_trials},
        )
    
    def test_multi_fact(self, target_tokens: int = 40000, num_facts: int = 3) -> BenchmarkResult:
        """Merke dir mehrere Fakten."""
        print("\n" + "="*50)
        print("📝 TEST 4: Multi-Fact Memory")
        print("="*50)
        
        facts = {
            "number": random.randint(100, 999),
            "color": random.choice(["red", "blue", "green"]),
            "animal": random.choice(["cat", "dog", "bird"]),
        }
        
        print(f"\n   Facts to remember: {facts}")
        
        self.model.reset_memory(1)
        
        intro = f"""Here are three things to remember:
1. The special number is {facts['number']}.
2. The chosen color is {facts['color']}.
3. The favorite animal is {facts['animal']}.
Please remember all three facts."""
        
        self._process_text(intro)
        current_tokens = self.count_tokens(intro)
        
        pbar = tqdm(total=target_tokens, initial=current_tokens, desc="   Processing")
        while current_tokens < target_tokens:
            filler = self._generate_filler(2000)
            self._process_text(filler)
            added = self.count_tokens(filler)
            current_tokens += added
            pbar.update(added)
        pbar.close()
        
        correct = 0
        
        r1 = self.model.generate("What was the special number?", max_new_tokens=20, temperature=0, reset_memory=False)
        if str(facts['number']) in r1:
            correct += 1
            print(f"   ✅ Number: {facts['number']}")
        else:
            print(f"   ❌ Number: expected {facts['number']}, got '{r1[:30]}...'")
        
        r2 = self.model.generate("What was the chosen color?", max_new_tokens=20, temperature=0, reset_memory=False)
        if facts['color'].lower() in r2.lower():
            correct += 1
            print(f"   ✅ Color: {facts['color']}")
        else:
            print(f"   ❌ Color: expected {facts['color']}, got '{r2[:30]}...'")
        
        r3 = self.model.generate("What was the favorite animal?", max_new_tokens=20, temperature=0, reset_memory=False)
        if facts['animal'].lower() in r3.lower():
            correct += 1
            print(f"   ✅ Animal: {facts['animal']}")
        else:
            print(f"   ❌ Animal: expected {facts['animal']}, got '{r3[:30]}...'")
        
        return BenchmarkResult(
            test_name="multi_fact",
            total_tokens=target_tokens,
            beyond_context=target_tokens > self.context_window,
            accuracy=correct / num_facts,
            details={"correct": correct, "total": num_facts, "facts": facts},
        )
    
    def run_all(self, quick: bool = False) -> Dict:
        """Führt alle Tests aus."""
        print("\n" + "="*60)
        print("🐱 KITTEN-LoRA LITE BENCHMARK")
        print("="*60)
        
        # Memory Stats zu Beginn
        stats = self.model.get_memory_stats()
        print(f"\n📊 Initial Memory: F={stats['fast_norm_avg']:.2f} M={stats['medium_norm_avg']:.2f} S={stats['slow_norm_avg']:.2f}")
        
        if quick:
            tokens = 35000
            trials = 1
        else:
            tokens = 36000
            trials = 2
        
        results = []
        results.append(self.test_number_memory(target_tokens=tokens, num_trials=trials))
        results.append(self.test_color_memory(target_tokens=tokens, num_trials=trials))
        results.append(self.test_name_memory(target_tokens=tokens, num_trials=trials))
        results.append(self.test_multi_fact(target_tokens=tokens + 5000))
        
        # Zusammenfassung
        print("\n" + "="*60)
        print("📊 BENCHMARK SUMMARY")
        print("="*60)
        
        summary = {}
        total_accuracy = 0
        
        for r in results:
            print(f"\n   {r.test_name}:")
            print(f"      Tokens: {r.total_tokens:,}")
            print(f"      Accuracy: {r.accuracy*100:.0f}%")
            summary[r.test_name] = r.accuracy
            total_accuracy += r.accuracy
        
        avg_accuracy = total_accuracy / len(results)
        summary["average"] = avg_accuracy
        
        print(f"\n   📈 AVERAGE: {avg_accuracy*100:.1f}%")
        
        # Final Memory Stats
        stats = self.model.get_memory_stats()
        print(f"\n📊 Final Memory: F={stats['fast_norm_avg']:.2f} M={stats['medium_norm_avg']:.2f} S={stats['slow_norm_avg']:.2f}")
        
        # Speichern
        output_file = self.output_dir / f"kitten_lite_{int(time.time())}.json"
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n   💾 Saved: {output_file}")
        
        return summary


def main():
    print("="*60)
    print("🐱 KITTEN-LoRA LITE BENCHMARK")
    print("="*60)
    
    # ════════════════════════════════════════════════════════════════════
    # PFAD-FIX: Pfad logisch berechnen
    # ════════════════════════════════════════════════════════════════════
    
    # 1. Finde das Wurzelverzeichnis des Skripts (egal ob es aus src/ läuft)
    try:
        base_dir = Path(__file__).resolve().parent.resolve().parent
        print(f"🔍 Projekt-Root erkannt: {base_dir}")
    except NameError:
        base_dir = Path.cwd()
        print(f"🔍 Fallback auf CWD: {base_dir}")

    # 2. Bilde den Pfad zu den Gewichten absolut korrekt
    weights_dir = base_dir / "models" / "kitten_simple" / "best" if smol else base_dir / "models" / "kitten_simple_big" / "best"
    print(f"🔍 Suche Gewichte in: {weights_dir}")
    
    # ══════════════════════════════════════════════════════════════════════
    # MODELL INIT
    # ══════════════════════════════════════════════════════════════════════
    config = HOPEConfig(
        r_fast=8 if smol else 16,
        r_medium=32 if smol else 64,
        r_slow=64 if smol else 128,  # AUF 64 GEÄNDERT!
        chunk_medium=16 if smol else 32,
        chunk_slow=64 if smol else 128,
        hidden_dim=64 if smol else 128,
        surprise_threshold=-1.0,
        memory_decay=0.9995,
        use_newton_schulz=False,
    )
    
    # Cache Pfad ebenfalls korrigieren
    cache_dir = base_dir / "cache"
    
    model = HOPEModel(
        model_id="Qwen/Qwen3-0.6B" if smol else "Qwen/Qwen3-1.7B",
        config=config,
        cache_dir=str(cache_dir),
    )
    
    # 3. Pfad-Ausgabe für Debugging
    if not weights_dir.exists():
        print(f"❌ FEHLER: Pfad existiert nicht: {weights_dir}")
        print(f"   Bitte überprüfen, ob die Dateien wirklich dort liegen.")
        return
    else:
        print(f"✅ Pfad gefunden. Lade Modell...")

    # 4. Lade die Gewichte (wie in kitten_model.py implementiert)
    if model.load_hope_weights(str(weights_dir)):
        print(f"✅ Gewichte geladen.")
    else:
        print(f"⚠️ Konnte Gewichte nicht laden (siehe oben im Fehler).")
        # Optional: Hier könnte man mit 'best' und ohne HOPE weitermachen, wenn man will.
        # Für diesen Test lassen wir es hierbei.

    print(f"   Device: {next(model.model.parameters()).device}")
    
    # ══════════════════════════════════════════════════════════════════════
    # BENCHMARK
    # ══════════════════════════════════════════════════════════════════════
    benchmark = KittenBenchmarkLite(model)
    
    # Wir nutzen die Tests, die über 32k Tokens gehen (Infinite Context Test)
    summary = {} # Initialize summary here
    results = benchmark.run_all(quick=True)
    
    # Zusammenfassung
     # ══════════════════════════════════════════════════════════════════
    # BENCHMARK SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("📊 BENCHMARK SUMMARY")
    print("="*60)

    # Hier die Reparatur
    for name, stats in summary.items():
        print(f"\n   {name}:")
        print(f"      Tokens: {stats.get('total_tokens', 0):,}")  # Fallback
        print(f"      Accuracy: {stats.get('accuracy', 0.0) * 100.0:.0f}%")
    
    # Memory-Block (wird jetzt erreicht)
    stats = model.get_memory_stats()
    print(f"\n📊 Final Memory Stats:")
    print(f"   F: {stats['fast_norm_avg']:.2f}")
    print(f"   M: {stats['medium_norm_avg']:.2f}")
    print(f"   {stats['total_steps']:,}")

    # Speichern
    output_file = base_dir / "benchmarks" / f"kitten_lite_{int(time.time())}.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("✅ BENCHMARK COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()