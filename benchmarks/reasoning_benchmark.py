"""
Reasoning Benchmark for NovaRCV
===============================
Tests logical reasoning, math, planning, and problem-solving capabilities.
"""
from typing import Dict, List, Callable, Any


class ReasoningTestCase:
    def __init__(self, name: str, prompt: str, category: str, 
                 validator: Callable[[str], bool], difficulty: str = "easy"):
        self.name = name
        self.prompt = prompt
        self.category = category
        self.validator = validator
        self.difficulty = difficulty


def get_benchmarks() -> List[ReasoningTestCase]:
    tests = []
    
    # ---- MATH REASONING ----
    
    def validate_math_calculation(response: str) -> bool:
        """Check if response contains correct math."""
        # Look for the answer
        if '3.14' in response or 'π' in response or 'pi' in response:
            return True
        if '22/7' in response:
            return True
        return False
    
    tests.append(ReasoningTestCase(
        name="pi_approximation",
        prompt="What is the approximate value of π (pi) to 3 decimal places? Show your calculation.",
        category="math",
        validator=validate_math_calculation,
        difficulty="easy",
    ))
    
    def validate_prime_factorization(response: str) -> bool:
        """Check if response identifies correct prime factors of 84."""
        if '2' in response and '3' in response and '7' in response:
            # Check for 2*2*3*7 or 2^2*3*7 pattern
            if ('2*2*3*7' in response or '2²×3×7' in response or 
                '2^2' in response and '3' in response and '7' in response):
                return True
        return False
    
    tests.append(ReasoningTestCase(
        name="prime_factorization",
        prompt="Find the prime factorization of 84. Show each step.",
        category="math",
        validator=validate_prime_factorization,
        difficulty="easy",
    ))
    
    # ---- LOGICAL REASONING ----
    
    def validate_syllogism(response: str) -> bool:
        """Check if response correctly solves the syllogism."""
        lowered = response.lower()
        if 'socrates' in lowered and 'mortal' in lowered:
            return True
        return False
    
    tests.append(ReasoningTestCase(
        name="syllogism_socrates",
        prompt="All men are mortal. Socrates is a man. Therefore, what conclusion follows?",
        category="logical_reasoning",
        validator=validate_syllogism,
        difficulty="easy",
    ))
    
    def validate_water_jug(response: str) -> bool:
        """Check if response describes correct 4-gallon solution."""
        lowered = response.lower()
        if '4' in response and ('gallon' in lowered or 'litre' in lowered or 'liter' in lowered):
            if 'fill' in lowered and ('pour' in lowered or 'empty' in lowered):
                return True
        return False
    
    tests.append(ReasoningTestCase(
        name="water_jug_problem",
        prompt="You have a 3-gallon jug and a 5-gallon jug. How do you measure exactly 4 gallons? Explain step by step.",
        category="logical_reasoning",
        validator=validate_water_jug,
        difficulty="medium",
    ))
    
    # ---- PLANNING ----
    
    def validate_travel_planning(response: str) -> bool:
        """Check if response has structured planning."""
        lowered = response.lower()
        has_steps = ('step' in lowered or 'first' in lowered or '1.' in lowered)
        has_transport = ('flight' in lowered or 'train' in lowered or 'bus' in lowered or 'car' in lowered)
        has_accommodation = ('hotel' in lowered or 'stay' in lowered or 'airbnb' in lowered)
        return has_steps and has_transport and has_accommodation
    
    tests.append(ReasoningTestCase(
        name="travel_itinerary",
        prompt="Plan a 3-day trip to Paris. Include transportation, accommodation, and daily activities.",
        category="planning",
        validator=validate_travel_planning,
        difficulty="medium",
    ))
    
    # ---- COMMON SENSE ----
    
    def validate_common_sense(response: str) -> bool:
        """Check if response correctly identifies what's wrong."""
        lowered = response.lower()
        if 'dry' in lowered and ('wet' in lowered or 'water' in lowered or 'umbrella' in lowered):
            return True
        return False
    
    tests.append(ReasoningTestCase(
        name="common_sense_paradox",
        prompt="If you're holding an umbrella while standing in the rain, why are you still getting wet? What's wrong with this scenario?",
        category="common_sense",
        validator=validate_common_sense,
        difficulty="easy",
    ))
    
    # ---- CAUSE AND EFFECT ----
    
    def validate_causation(response: str) -> bool:
        """Check if response correctly identifies causation vs correlation."""
        lowered = response.lower()
        if 'correlation' in lowered and 'causation' in lowered:
            return True
        return False
    
    tests.append(ReasoningTestCase(
        name="correlation_vs_causation",
        prompt="Ice cream sales increase when shark attacks increase. Does this mean ice cream causes shark attacks? Explain.",
        category="cause_effect",
        validator=validate_causation,
        difficulty="medium",
    ))
    
    # ---- ANALOGICAL REASONING ----
    
    def validate_analogy(response: str) -> bool:
        """Check if response correctly completes the analogy."""
        lowered = response.lower()
        if 'finger' in lowered and ('hand' in lowered or 'palm' in lowered):
            return True
        if 'leaf' in lowered and ('tree' in lowered or 'branch' in lowered):
            return True
        return False
    
    tests.append(ReasoningTestCase(
        name="analogy_completion",
        prompt="Complete the analogy: 'Toe is to foot as _______ is to hand.' Explain your reasoning.",
        category="analogical_reasoning",
        validator=validate_analogy,
        difficulty="easy",
    ))
    
    # ---- COUNTERFACTUAL REASONING ----
    
    def validate_counterfactual(response: str) -> bool:
        """Check if response engages with counterfactual thinking."""
        lowered = response.lower()
        if ('would' in lowered and ('different' in lowered or 'change' in lowered or 'instead' in lowered)):
            return True
        return False
    
    tests.append(ReasoningTestCase(
        name="counterfactual_thinking",
        prompt="If the internet had never been invented, how would people communicate and access information today?",
        category="counterfactual",
        validator=validate_counterfactual,
        difficulty="medium",
    ))
    
    return tests


def run_benchmark(model, tokenizer=None, verbose: bool = True) -> Dict[str, Any]:
    tests = get_benchmarks()
    results = []
    passed = 0
    failed = 0
    
    print("=" * 70)
    print("REASONING BENCHMARK")
    print("=" * 70)
    
    for i, test in enumerate(tests):
        if verbose:
            print(f"\n[{i+1}/{len(tests)}] {test.name} ({test.difficulty})")
        
        try:
            if tokenizer and hasattr(model, 'generate'):
                input_ids = tokenizer.encode(test.prompt, return_tensors='pt')
                generated = model.generate(input_ids, max_new_tokens=150)
                response = tokenizer.decode(generated[0], skip_special_tokens=True)
            elif hasattr(model, 'generate'):
                response = model.generate(test.prompt, max_length=150)
            else:
                response = model(test.prompt)
            
            success = test.validator(response)
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            success = False
            response = ""
        
        if success:
            passed += 1
            status = "✅ PASS"
        else:
            failed += 1
            status = "❌ FAIL"
        
        results.append({
            'name': test.name, 'category': test.category,
            'difficulty': test.difficulty, 'passed': success,
        })
        
        if verbose:
            print(f"  Result: {status}")
    
    total = len(tests)
    accuracy = (passed / total) * 100 if total > 0 else 0
    
    print(f"\n{'=' * 70}")
    print(f"REASONING BENCHMARK RESULTS")
    print(f"{'=' * 70}")
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    return {
        'benchmark': 'reasoning',
        'total': total, 'passed': passed, 'failed': failed,
        'accuracy': accuracy, 'results': results,
    }


if __name__ == '__main__':
    run_benchmark(None, verbose=True)