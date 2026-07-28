"""
Knowledge Benchmark for NovaRCV
================================
Tests factual knowledge across science, history, geography, and more.
"""
from typing import Dict, List, Callable, Any


class KnowledgeTestCase:
    def __init__(self, name: str, prompt: str,
                 validator: Callable[[str], bool], difficulty: str = "easy"):
        self.name = name
        self.prompt = prompt
        self.validator = validator
        self.difficulty = difficulty


def get_benchmarks() -> List[KnowledgeTestCase]:
    tests = []

    def validate_python_creator(response: str) -> bool:
        lowered = response.lower()
        if 'guido' in lowered and 'van rossum' in lowered:
            if '1991' in response or '1989' in response or '1990' in response:
                return True
        return False

    tests.append(KnowledgeTestCase(
        name="python_creator",
        prompt="Who created Python and in what year was it first released?",
        validator=validate_python_creator,
        difficulty="easy",
    ))

    def validate_speed_of_light(response: str) -> bool:
        if ('299,792,458' in response or '3x10' in response or 
            '3×10' in response or '300,000' in response or '299792' in response):
            if 'm/s' in response or 'km/s' in response:
                return True
        return False

    tests.append(KnowledgeTestCase(
        name="speed_of_light",
        prompt="What is the speed of light in a vacuum? Give the exact value.",
        validator=validate_speed_of_light,
        difficulty="easy",
    ))

    def validate_moon_landing(response: str) -> bool:
        lowered = response.lower()
        if ('1969' in response and ('armstrong' in lowered or 'neil' in lowered)):
            return True
        return False

    tests.append(KnowledgeTestCase(
        name="first_moon_landing",
        prompt="When was the first moon landing and who was the first person to walk on the moon?",
        validator=validate_moon_landing,
        difficulty="easy",
    ))

    def validate_largest_ocean(response: str) -> bool:
        lowered = response.lower()
        if 'pacific' in lowered:
            return True
        return False

    tests.append(KnowledgeTestCase(
        name="largest_ocean",
        prompt="What is the largest ocean on Earth?",
        validator=validate_largest_ocean,
        difficulty="easy",
    ))

    def validate_einstein_eq(response: str) -> bool:
        if ('E=mc' in response or 'E = mc' in response):
            if '²' in response or '2' in response or '^2' in response:
                return True
        return False

    tests.append(KnowledgeTestCase(
        name="einstein_famous_equation",
        prompt="What is Einstein's most famous equation?",
        validator=validate_einstein_eq,
        difficulty="easy",
    ))

    def validate_chromosomes(response: str) -> bool:
        if '46' in response or '23 pairs' in response or '23' in response:
            return True
        return False

    tests.append(KnowledgeTestCase(
        name="human_chromosomes",
        prompt="How many chromosomes do humans have?",
        validator=validate_chromosomes,
        difficulty="easy",
    ))

    return tests


def run_benchmark(model, tokenizer=None, verbose: bool = True) -> Dict[str, Any]:
    tests = get_benchmarks()
    results = []
    passed = 0
    failed = 0

    print("=" * 70)
    print("KNOWLEDGE BENCHMARK")
    print("=" * 70)

    for i, test in enumerate(tests):
        if verbose:
            print(f"\n[{i+1}/{len(tests)}] {test.name} ({test.difficulty})")

        try:
            if tokenizer and hasattr(model, 'generate'):
                input_ids = tokenizer.encode(test.prompt, return_tensors='pt')
                generated = model.generate(input_ids, max_new_tokens=100)
                response = tokenizer.decode(generated[0], skip_special_tokens=True)
            elif hasattr(model, 'generate'):
                response = model.generate(test.prompt, max_length=100)
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
            'name': test.name, 'passed': success,
        })

        if verbose:
            print(f"  Result: {status}")

    total = len(tests)
    accuracy = (passed / total) * 100 if total > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"KNOWLEDGE BENCHMARK RESULTS")
    print(f"{'=' * 70}")
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    print(f"Accuracy: {accuracy:.1f}%")

    return {
        'benchmark': 'knowledge',
        'total': total, 'passed': passed, 'failed': failed,
        'accuracy': accuracy, 'results': results,
    }


if __name__ == '__main__':
    run_benchmark(None, verbose=True)