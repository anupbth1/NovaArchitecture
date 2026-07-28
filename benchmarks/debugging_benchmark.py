"""
Debugging Benchmark for NovaRCV
================================
Tests ability to find and fix bugs in code snippets.
"""
from typing import Dict, List, Callable, Any


class DebuggingTestCase:
    def __init__(self, name: str, buggy_code: str, prompt: str,
                 category: str, validator: Callable[[str], bool],
                 difficulty: str = "easy"):
        self.name = name
        self.buggy_code = buggy_code
        self.prompt = prompt
        self.category = category
        self.validator = validator
        self.difficulty = difficulty


def get_benchmarks() -> List[DebuggingTestCase]:
    tests = []

    def validate_off_by_one(response: str) -> bool:
        """Bug: range(1,10) should be range(10)."""
        lowered = response.lower()
        if 'range(1' in response or 'range(1, 10)' in response:
            if ('off by one' in lowered or 'starts at 1' in lowered or 
                'start at 0' in lowered or 'should be range(10)' in lowered):
                return True
        # Check if the bug is correctly identified
        if 'off' in lowered and 'one' in lowered and 'range' in lowered:
            return True
        return False
    
    tests.append(DebuggingTestCase(
        name="off_by_one_error",
        buggy_code="""def print_numbers():
    for i in range(1, 10):
        print(i)
print_numbers()  # Expected: 0,1,2,...,9 but gets 1,2,...,9""",
        prompt="Find the bug in this code. The function should print numbers 0 through 9 but it's missing 0.",
        category="bug_finding",
        validator=validate_off_by_one,
        difficulty="easy",
    ))

    def validate_mutating_list(response: str) -> bool:
        """Bug: removing elements while iterating."""
        lowered = response.lower()
        if ('mutating' in lowered or 'modifying' in lowered or 
            'removing' in lowered or 'change' in lowered):
            if ('iterating' in lowered or 'loop' in lowered or 'list' in lowered):
                return True
        return False
    
    tests.append(DebuggingTestCase(
        name="mutating_list_while_iterating",
        buggy_code="""numbers = [1, 2, 3, 4, 5]
for n in numbers:
    if n % 2 == 0:
        numbers.remove(n)
print(numbers)  # Unexpected behavior""",
        prompt="This code tries to remove even numbers from a list but produces unexpected results. What's the bug?",
        category="bug_finding",
        validator=validate_mutating_list,
        difficulty="medium",
    ))

    def validate_integer_division(response: str) -> bool:
        """Bug: 5/2 gives 2.5 in Python 3, not 2."""
        lowered = response.lower()
        if ('integer division' in lowered or 'floor division' in lowered or
            '//' in lowered or 'float' in lowered):
            return True
        return False
    
    tests.append(DebuggingTestCase(
        name="integer_division",
        buggy_code="""def average(a, b):
    return (a + b) / 2
print(average(5, 3))  # Expected: 4, but gets 4.0""",
        prompt="This code should return an integer but returns a float. The bug is that / performs float division instead of integer division. Suggest the fix.",
        category="bug_fixing",
        validator=validate_integer_division,
        difficulty="easy",
    ))

    def validate_unclosed_file(response: str) -> bool:
        """Bug: file not closed properly."""
        lowered = response.lower()
        if ('close' in lowered and ('file' in lowered or 'resource' in lowered or 'with' in lowered)):
            if ('context manager' in lowered or 'with statement' in lowered or 
                'not closed' in lowered or 'leak' in lowered):
                return True
        return False
    
    tests.append(DebuggingTestCase(
        name="unclosed_file",
        buggy_code="""def read_file(filename):
    f = open(filename, 'r')
    data = f.read()
    return data  # File is never closed!""",
        prompt="This function reads a file but has a resource management bug. What's wrong and how would you fix it?",
        category="bug_finding",
        validator=validate_unclosed_file,
        difficulty="easy",
    ))

    def validate_variable_shadowing(response: str) -> bool:
        """Bug: variable shadows built-in."""
        lowered = response.lower()
        if ('shadow' in lowered or 'built-in' in lowered or 'override' in lowered):
            return True
        if 'list' in lowered and ('variable' in lowered or 'function' in lowered or 'name' in lowered):
            return True
        return False
    
    tests.append(DebuggingTestCase(
        name="variable_shadowing",
        buggy_code="""list = [1, 2, 3]
my_list = list  # This shadows the built-in list type
print(list(range(5)))  # ERROR!""",
        prompt="This code produces an error. What's the bug and how would you fix it?",
        category="bug_finding",
        validator=validate_variable_shadowing,
        difficulty="medium",
    ))

    def validate_infinite_recursion(response: str) -> bool:
        """Bug: no base case in recursion."""
        lowered = response.lower()
        if ('base case' in lowered or 'termination' in lowered or 'stop' in lowered):
            if ('recursion' in lowered or 'recursive' in lowered or 'infinite' in lowered):
                return True
        return False
    
    tests.append(DebuggingTestCase(
        name="infinite_recursion",
        buggy_code="""def factorial(n):
    return n * factorial(n - 1)  # Missing base case!
print(factorial(5))""",
        prompt="This recursive function causes a RecursionError. What's missing and how would you fix it?",
        category="bug_fixing",
        validator=validate_infinite_recursion,
        difficulty="easy",
    ))

    return tests


def run_benchmark(model, tokenizer=None, verbose: bool = True) -> Dict[str, Any]:
    tests = get_benchmarks()
    results = []
    passed = 0
    failed = 0
    
    print("=" * 70)
    print("DEBUGGING BENCHMARK")
    print("=" * 70)
    
    for i, test in enumerate(tests):
        if verbose:
            print(f"\n[{i+1}/{len(tests)}] {test.name} ({test.difficulty})")
            print(f"  Buggy code: {test.buggy_code[:60]}...")
        
        try:
            if tokenizer and hasattr(model, 'generate'):
                input_ids = tokenizer.encode(test.prompt, return_tensors='pt')
                generated = model.generate(input_ids, max_new_tokens=200)
                response = tokenizer.decode(generated[0], skip_special_tokens=True)
            elif hasattr(model, 'generate'):
                response = model.generate(test.prompt, max_length=200)
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
    print(f"DEBUGGING BENCHMARK RESULTS")
    print(f"{'=' * 70}")
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    return {
        'benchmark': 'debugging',
        'total': total, 'passed': passed, 'failed': failed,
        'accuracy': accuracy, 'results': results,
    }


if __name__ == '__main__':
    run_benchmark(None, verbose=True)