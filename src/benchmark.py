# src/benchmark_extended.py
"""
Extended Benchmark für HOPE-LoRA: Testet ÜBER das Kontextfenster hinaus.
Qwen3-0.6B hat 32,768 Token Kontextfenster.
"""

import torch
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from kitten_model import HOPEModel, HOPEConfig


@dataclass
class ExtendedBenchmarkResult:
    task_name: str
    context_tokens: int
    beyond_context_window: bool
    accuracy: float
    memory_stats: Dict
    latency_ms: float


class ExtendedContextBenchmark:
    """
    Benchmark der explizit ÜBER das 32k Kontextfenster hinausgeht.
    
    Strategie:
    1. Füttere das Modell mit Chunks die einzeln < 32k sind
    2. Aber kumulativ >> 32k sind
    3. Das Memory sollte die Info behalten
    """
    
    def __init__(
        self,
        model: HOPEModel,
        output_dir: str = "benchmarks/results",
        context_window: int = 32768,  # Qwen3 Kontextfenster
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.context_window = context_window
        self.results: List[ExtendedBenchmarkResult] = []
    
    def count_tokens(self, text: str) -> int:
        """Zählt Tokens in einem Text."""
        return len(self.model.tokenizer.encode(text))
    
    # ═══════════════════════════════════════════════════════
    # Test 1: Streaming Facts Beyond Context
    # ═══════════════════════════════════════════════════════
    
    def run_streaming_facts(
        self,
        num_facts: int = 100,
        tokens_between_facts: int = 500,
    ) -> ExtendedBenchmarkResult:
        """
        Streamt viele Fakten durch das Modell, weit über das Kontextfenster hinaus.
        
        Idee:
        - Definiere 100 Fakten (Person → Farbe, Tier)
        - Füttere sie einzeln, mit Filler dazwischen
        - Am Ende: Frage nach einem FRÜHEN Fakt
        
        Total tokens: ~100 * 500 = 50,000 (> 32k!)
        """
        print("\n" + "="*60)
        print("🌊 STREAMING FACTS BEYOND CONTEXT WINDOW")
        print("="*60)
        
        # Generiere Fakten
        names = [f"Person_{i}" for i in range(num_facts)]
        colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan", "magenta", "brown"]
        animals = ["dog", "cat", "bird", "fish", "rabbit", "hamster", "turtle", "snake", "lizard", "frog"]
        
        facts = []
        for i, name in enumerate(names):
            fact = {
                "name": name,
                "color": colors[i % len(colors)],
                "animal": animals[i % len(animals)],
                "number": random.randint(100, 999),
            }
            facts.append(fact)
        
        # Filler-Generator
        filler_templates = [
            "The weather continues to be pleasant. ",
            "Scientists made new discoveries today. ",
            "The stock market showed mixed results. ",
            "Local events attracted many visitors. ",
            "Technology advances rapidly. ",
        ]
        
        def generate_filler(target_tokens: int) -> str:
            filler = ""
            while self.count_tokens(filler) < target_tokens:
                filler += random.choice(filler_templates)
            return filler
        
        # Reset Memory EINMAL am Anfang
        self.model.reset_memory(1)
        
        total_tokens = 0
        start_time = time.time()
        
        print(f"\n📝 Streaming {num_facts} facts with ~{tokens_between_facts} tokens between each...")
        print(f"   Expected total: ~{num_facts * tokens_between_facts:,} tokens")
        print(f"   Context window: {self.context_window:,} tokens")
        print(f"   Beyond context: {num_facts * tokens_between_facts > self.context_window}")
        
        # Streame alle Fakten durch
        for i, fact in enumerate(facts):
            # Fakt als Statement
            fact_text = f"{fact['name']}'s favorite color is {fact['color']}, they have a {fact['animal']}, and their lucky number is {fact['number']}. "
            
            # Filler danach
            filler = generate_filler(tokens_between_facts)
            
            chunk = fact_text + filler
            chunk_tokens = self.count_tokens(chunk)
            total_tokens += chunk_tokens
            
            # Durch das Modell jagen (ohne Generation, nur Forward für Memory-Update)
            inputs = self.model.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=min(chunk_tokens + 100, self.context_window),
            ).to(next(self.model.model.parameters()).device)
            
            # Forward pass (Memory wird intern aktualisiert)
            with torch.no_grad():
                _ = self.model.model(**inputs)
            
            # Progress
            if (i + 1) % 10 == 0:
                stats = self.model.get_memory_stats()
                print(f"   [{i+1}/{num_facts}] Total tokens: {total_tokens:,} | "
                      f"Memory: F={stats['fast_norm_avg']:.2f} M={stats['medium_norm_avg']:.2f} S={stats['slow_norm_avg']:.2f}")
        
        processing_time = time.time() - start_time
        
        print(f"\n✅ Processed {total_tokens:,} tokens in {processing_time:.1f}s")
        print(f"   This is {total_tokens / self.context_window:.1f}x the context window!")
        
        # ═══════════════════════════════════════════════════════
        # Jetzt: Teste Erinnerung an FRÜHE Fakten
        # ═══════════════════════════════════════════════════════
        print("\n🧪 Testing recall of EARLY facts (should be outside context window)...")
        
        # Teste die ersten 5 Fakten (die am weitesten "zurück" liegen)
        test_facts = facts[:5]
        correct_color = 0
        correct_animal = 0
        correct_number = 0
        
        for fact in test_facts:
            prompt = f"What is {fact['name']}'s favorite color, pet, and lucky number?"
            
            # WICHTIG: reset_memory=False um das akkumulierte Memory zu behalten!
            response = self.model.generate(
                prompt,
                max_new_tokens=64,
                temperature=0,
                reset_memory=False,
            )
            
            print(f"\n   Q: {prompt}")
            print(f"   A: {response[:150]}...")
            print(f"   Expected: color={fact['color']}, animal={fact['animal']}, number={fact['number']}")
            
            if fact['color'].lower() in response.lower():
                correct_color += 1
                print(f"   ✅ Color correct")
            else:
                print(f"   ❌ Color wrong")
                
            if fact['animal'].lower() in response.lower():
                correct_animal += 1
                print(f"   ✅ Animal correct")
            else:
                print(f"   ❌ Animal wrong")
                
            if str(fact['number']) in response:
                correct_number += 1
                print(f"   ✅ Number correct")
            else:
                print(f"   ❌ Number wrong")
        
        total_correct = correct_color + correct_animal + correct_number
        total_possible = len(test_facts) * 3
        accuracy = total_correct / total_possible
        
        stats = self.model.get_memory_stats()
        
        result = ExtendedBenchmarkResult(
            task_name="streaming_facts_beyond_context",
            context_tokens=total_tokens,
            beyond_context_window=total_tokens > self.context_window,
            accuracy=accuracy,
            memory_stats=stats,
            latency_ms=processing_time * 1000,
        )
        
        print(f"\n📊 Results:")
        print(f"   Color: {correct_color}/{len(test_facts)}")
        print(f"   Animal: {correct_animal}/{len(test_facts)}")
        print(f"   Number: {correct_number}/{len(test_facts)}")
        print(f"   Overall: {accuracy*100:.1f}%")
        
        return result
    
    # ═══════════════════════════════════════════════════════
    # Test 2: Passkey at Extreme Distance
    # ═══════════════════════════════════════════════════════
    
    def run_passkey_extreme(
        self,
        target_tokens: int = 50000,  # Weit über 32k
    ) -> ExtendedBenchmarkResult:
        """
        Versteckt einen Passkey am Anfang, dann kommt SEHR viel Filler.
        """
        print("\n" + "="*60)
        print(f"🔑 PASSKEY AT EXTREME DISTANCE ({target_tokens:,} tokens)")
        print("="*60)
        
        passkey = f"ULTRAKEY{random.randint(10000, 99999)}"
        
        # Passkey-Statement
        passkey_text = f"IMPORTANT: The secret passkey is {passkey}. Remember this passkey. "
        
        # Generiere Filler in Chunks
        filler_sentences = [
            "The weather today is quite pleasant with mild temperatures. ",
            "Scientists have discovered new species in the deep ocean. ",
            "The stock market showed mixed results in early trading. ",
            "Local authorities announced new infrastructure projects. ",
            "Researchers are studying the effects of climate change. ",
            "The annual festival attracted thousands of visitors. ",
            "Technology companies reported strong quarterly earnings. ",
        ]
        
        # Reset Memory
        self.model.reset_memory(1)
        
        # Erst den Passkey durchs Modell
        print(f"\n1️⃣ Processing passkey: {passkey}")
        inputs = self.model.tokenizer(
            passkey_text,
            return_tensors="pt",
        ).to(next(self.model.model.parameters()).device)
        
        with torch.no_grad():
            _ = self.model.model(**inputs)
        
        total_tokens = self.count_tokens(passkey_text)
        
        # Jetzt viel Filler in Chunks
        print(f"\n2️⃣ Processing filler to reach {target_tokens:,} tokens...")
        
        chunk_size = 2000  # Tokens pro Chunk
        
        while total_tokens < target_tokens:
            # Generiere einen Chunk
            chunk = ""
            while self.count_tokens(chunk) < chunk_size:
                chunk += random.choice(filler_sentences)
            
            chunk_tokens = self.count_tokens(chunk)
            
            inputs = self.model.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=chunk_size + 100,
            ).to(next(self.model.model.parameters()).device)
            
            with torch.no_grad():
                _ = self.model.model(**inputs)
            
            total_tokens += chunk_tokens
            
            if total_tokens % 10000 < chunk_size:
                stats = self.model.get_memory_stats()
                print(f"   Processed: {total_tokens:,} tokens | "
                      f"Memory: F={stats['fast_norm_avg']:.2f} M={stats['medium_norm_avg']:.2f} S={stats['slow_norm_avg']:.2f}")
        
        print(f"\n✅ Total tokens processed: {total_tokens:,}")
        print(f"   This is {total_tokens / self.context_window:.1f}x the context window!")
        
        # Jetzt nach dem Passkey fragen
        print(f"\n3️⃣ Testing passkey recall...")
        
        prompt = "What is the secret passkey that was mentioned at the very beginning?"
        
        response = self.model.generate(
            prompt,
            max_new_tokens=32,
            temperature=0,
            reset_memory=False,
        )
        
        print(f"   Expected: {passkey}")
        print(f"   Got: {response}")
        
        correct = passkey in response
        
        result = ExtendedBenchmarkResult(
            task_name="passkey_extreme_distance",
            context_tokens=total_tokens,
            beyond_context_window=True,
            accuracy=1.0 if correct else 0.0,
            memory_stats=self.model.get_memory_stats(),
            latency_ms=0,
        )
        
        print(f"\n{'✅ CORRECT!' if correct else '❌ FAILED'}")
        
        return result
    
    # ═══════════════════════════════════════════════════════
    # Test 3: Multi-Turn Beyond Context (FIXED)
    # ═══════════════════════════════════════════════════════
    
    def run_multiturn_beyond_context(
        self,
        num_turns: int = 20,
        tokens_per_turn: int = 2000,
    ) -> ExtendedBenchmarkResult:
        """
        Simuliert eine sehr lange Konversation.
        Gesamte Tokens: num_turns * tokens_per_turn = 40,000 (> 32k)
        """
        print("\n" + "="*60)
        print(f"💬 MULTI-TURN BEYOND CONTEXT ({num_turns} turns × {tokens_per_turn} tokens)")
        print("="*60)
        
        # Secret am Anfang
        secret_number = random.randint(1000, 9999)
        secret_color = random.choice(["crimson", "azure", "emerald", "golden", "violet"])
        
        print(f"\n📌 Secrets to remember:")
        print(f"   Number: {secret_number}")
        print(f"   Color: {secret_color}")
        
        # Reset Memory EINMAL
        self.model.reset_memory(1)
        
        # Turn 1: Secrets einführen
        intro = f"""I want to tell you two secrets. 
First, my lucky number is {secret_number}. 
Second, my favorite color is {secret_color}. 
Please remember both of these facts."""
        
        print(f"\n[Turn 1] Introducing secrets...")
        inputs = self.model.tokenizer(intro, return_tensors="pt").to(next(self.model.model.parameters()).device)
        with torch.no_grad():
            _ = self.model.model(**inputs)
        
        total_tokens = self.count_tokens(intro)
        
        # Viele Zwischen-Turns mit Filler
        topics = [
            "Tell me about the history of ancient Rome.",
            "Explain how computers work.",
            "What are the benefits of exercise?",
            "Describe the solar system.",
            "Tell me about famous scientists.",
            "Explain climate change.",
            "What is artificial intelligence?",
            "Describe the human brain.",
            "Tell me about ocean life.",
            "Explain quantum physics basics.",
        ]
        
        for turn in range(2, num_turns):
            topic = topics[(turn - 2) % len(topics)]
            
            # Generiere eine lange Antwort
            filler = f"Let me tell you about: {topic}. " + " ".join([
                "This is a fascinating subject with many aspects to consider."
                for _ in range(tokens_per_turn // 20)
            ])
            
            inputs = self.model.tokenizer(
                filler,
                return_tensors="pt",
                truncation=True,
                max_length=tokens_per_turn + 100,
            ).to(next(self.model.model.parameters()).device)
            
            with torch.no_grad():
                _ = self.model.model(**inputs)
            
            total_tokens += self.count_tokens(filler)
            
            if turn % 5 == 0:
                stats = self.model.get_memory_stats()
                print(f"   [Turn {turn}/{num_turns}] Total: {total_tokens:,} tokens | "
                      f"Memory: F={stats['fast_norm_avg']:.2f} M={stats['medium_norm_avg']:.2f} S={stats['slow_norm_avg']:.2f}")
        
        print(f"\n✅ Processed {total_tokens:,} tokens ({total_tokens/self.context_window:.1f}x context window)")
        
        # Jetzt nach den Secrets fragen
        print("\n🧪 Testing recall of secrets from Turn 1...")
        
        # Test Number
        response_number = self.model.generate(
            "What was my lucky number that I told you at the very beginning?",
            max_new_tokens=32,
            temperature=0,
            reset_memory=False,
        )
        
        # Test Color
        response_color = self.model.generate(
            "What was my favorite color that I mentioned at the start?",
            max_new_tokens=32,
            temperature=0,
            reset_memory=False,
        )
        
        print(f"\n   Number - Expected: {secret_number}, Got: {response_number[:100]}")
        print(f"   Color - Expected: {secret_color}, Got: {response_color[:100]}")
        
        correct_number = str(secret_number) in response_number
        correct_color = secret_color.lower() in response_color.lower()
        
        accuracy = (int(correct_number) + int(correct_color)) / 2
        
        result = ExtendedBenchmarkResult(
            task_name="multiturn_beyond_context",
            context_tokens=total_tokens,
            beyond_context_window=True,
            accuracy=accuracy,
            memory_stats=self.model.get_memory_stats(),
            latency_ms=0,
        )
        
        print(f"\n   Number: {'✅' if correct_number else '❌'}")
        print(f"   Color: {'✅' if correct_color else '❌'}")
        print(f"   Overall: {accuracy*100:.0f}%")
        
        return result
    
    # ═══════════════════════════════════════════════════════
    # Baseline: Test OHNE HOPE (normales Modell)
    # ═══════════════════════════════════════════════════════
    
    def run_baseline_comparison(self) -> Dict:
        """
        Vergleicht HOPE-LoRA mit dem Base-Modell (theoretisch).
        """
        print("\n" + "="*60)
        print("📊 BASELINE COMPARISON")
        print("="*60)
        
        print("""
        HINWEIS: Ein fairer Vergleich würde erfordern:
        1. Das gleiche Modell OHNE HOPE-LoRA zu laden
        2. Die gleichen Tests durchzuführen
        
        Erwartung:
        - Base Modell: Versagt bei >32k Tokens (Info außerhalb des Fensters)
        - HOPE-LoRA: Sollte Info im Memory behalten
        
        Aktueller Status:
        - HOPE Memory akkumuliert (sichtbar an steigenden Norms)
        - Aber die Update-Netze sind UNTRAINIERT
        - Echte Verbesserung erfordert Training der Update-Netze
        """)
        
        return {"note": "Baseline comparison requires separate model loading"}
    
    # ═══════════════════════════════════════════════════════
    # Run All
    # ═══════════════════════════════════════════════════════
    
    def run_all(self) -> Dict:
        """Führt alle Extended Benchmarks aus."""
        results = []
        
        # Test 1: Streaming Facts
        result1 = self.run_streaming_facts(num_facts=50, tokens_between_facts=800)
        results.append(result1)
        
        # Test 2: Passkey Extreme
        result2 = self.run_passkey_extreme(target_tokens=40000)
        results.append(result2)
        
        # Test 3: Multi-Turn
        result3 = self.run_multiturn_beyond_context(num_turns=15, tokens_per_turn=3000)
        results.append(result3)
        
        # Baseline Info
        self.run_baseline_comparison()
        
        # Zusammenfassung
        print("\n" + "="*60)
        print("📈 EXTENDED BENCHMARK SUMMARY")
        print("="*60)
        
        summary = {
            "streaming_facts": {
                "accuracy": result1.accuracy,
                "tokens": result1.context_tokens,
                "beyond_context": result1.beyond_context_window,
            },
            "passkey_extreme": {
                "accuracy": result2.accuracy,
                "tokens": result2.context_tokens,
                "beyond_context": result2.beyond_context_window,
            },
            "multiturn": {
                "accuracy": result3.accuracy,
                "tokens": result3.context_tokens,
                "beyond_context": result3.beyond_context_window,
            },
        }
        
        print(f"\n🌊 Streaming Facts ({result1.context_tokens:,} tokens): {result1.accuracy*100:.1f}%")
        print(f"🔑 Passkey Extreme ({result2.context_tokens:,} tokens): {result2.accuracy*100:.1f}%")
        print(f"💬 Multi-Turn ({result3.context_tokens:,} tokens): {result3.accuracy*100:.1f}%")
        
        avg_accuracy = (result1.accuracy + result2.accuracy + result3.accuracy) / 3
        print(f"\n📊 Average (beyond {self.context_window:,} context): {avg_accuracy*100:.1f}%")
        
        # Speichern
        results_file = self.output_dir / f"extended_benchmark_{int(time.time())}.json"
        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Gespeichert: {results_file}")
        
        return summary


def main():
    """Extended Benchmark ausführen."""
    print("="*60)
    print("🚀 HOPE-LoRA EXTENDED CONTEXT BENCHMARK")
    print("   Testing BEYOND the 32k context window!")
    print("="*60)
    
    config = HOPEConfig(
        r_fast=4,
        r_medium=16,
        r_slow=64,
        chunk_medium=8,
        chunk_slow=64,
    )
    
    model = HOPEModel(
        model_id="Qwen/Qwen3-0.6B",
        config=config,
        cache_dir="./cache",
    )
    
    benchmark = ExtendedContextBenchmark(
        model,
        context_window=32768,
    )
    
    results = benchmark.run_all()
    
    print("\n" + "="*60)
    print("✅ EXTENDED BENCHMARK COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()