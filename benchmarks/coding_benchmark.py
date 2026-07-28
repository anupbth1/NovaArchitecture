"""
Coding Benchmark for NovaRCV
============================
Tests code generation, completion, bug fixing capabilities.

Each test case provides a prompt and expected validation criteria.
The benchmark runs the model and checks output against criteria.
"""
import ast
import sys
from typing import Dict, List, Callable, Any, Optional


class CodingTestCase:
    """A single coding test case with validation."""
    
    def __init__(
        self,
        name: str,
        prompt: str,
        category: str,
        validator: Callable[[str], bool],
        difficulty: str = "easy",
    ):
        self.name = name
        self.prompt = prompt
        self.category = category
        self.validator = validator
        self.difficulty = difficulty


def get_benchmarks() -> List[CodingTestCase]:
    """Return all coding benchmark test cases."""
    tests = []
    
    # ---- CODE GENERATION TESTS ----
    
    def validate_fibonacci(code: str) -> bool:
        """Check if code correctly computes Fibonacci."""
        try:
            exec_globals = {}
            exec(code, exec_globals)
            # Try common function names
            for name in ['fibonacci', 'fib', 'generate_fibonacci']:
                if name in exec_globals and callable(exec_globals[name]):
                    result = exec_globals[name](10)
                    if result == 55:
                        return True
            return False
        except Exception:
            return False
    
    tests.append(CodingTestCase(
        name="fibonacci_generation",
        prompt="Write a Python function to calculate the nth Fibonacci number. Return 55 for n=10.",
        category="code_generation",
        validator=validate_fibonacci,
        difficulty="easy",
    ))
    
    def validate_fizzbuzz(code: str) -> bool:
        """Check if code implements FizzBuzz correctly."""
        try:
            exec_globals = {}
            exec(code, exec_globals)
            for name in ['fizzbuzz', 'fizz_buzz', 'solve']:
                if name in exec_globals:
                    result = exec_globals[name](15)
                    expected = [1,2,'Fizz',4,'Buzz','Fizz',7,8,'Fizz','Buzz',11,'Fizz',13,14,'FizzBuzz']
                    if result == expected:
                        return True
            return False
        except Exception:
            return False
    
    tests.append(CodingTestCase(
        name="fizzbuzz_implementation",
        prompt="Write a Python function fizzbuzz(n) that returns a list for numbers 1 to n where multiples of 3 are 'Fizz', 5 are 'Buzz', both are 'FizzBuzz'.",
        category="code_generation",
        validator=validate_fizzbuzz,
        difficulty="easy",
    ))
    
    def validate_binary_search(code: str) -> bool:
        """Check binary search implementation."""
        try:
            exec_globals = {}
            exec(code, exec_globals)
            for name in ['binary_search', 'binsearch', 'search']:
                if name in exec_globals and callable(exec_globals[name]):
                    arr = [1, 3, 5, 7, 9, 11, 13]
                    idx = exec_globals[name](arr, 7)
                    if idx == 3:
                        return True
                    # Could be 0-indexed
                    if idx == 3:
                        return True
            return False
        except Exception:
            return False
    
    tests.append(CodingTestCase(
        name="binary_search_implementation",
        prompt="Write a Python function binary_search(arr, target) that returns the index of target in a sorted array. Test with arr=[1,3,5,7,9,11,13], target=7 should return 3.",
        category="code_generation",
        validator=validate_binary_search,
        difficulty="medium",
    ))
    
    # ---- CODE COMPLETION TESTS ----
    
    def validate_async_callable(code: str) -> bool:
        """Check if code contains proper async function with await."""
        try:
            tree = ast.parse(code)
            has_async = any(
                isinstance(node, ast.AsyncFunctionDef)
                for node in ast.walk(tree)
            )
            has_await = 'await' in code
            return has_async and has_await
        except SyntaxError:
            return False
    
    tests.append(CodingTestCase(
        name="async_fetch_implementation",
        prompt="Complete the async function that fetches data from an API using aiohttp. The function should be async def fetch_data(url): and use await.",
        category="code_completion",
        validator=validate_async_callable,
        difficulty="medium",
    ))
    
    def validate_decorator(code: str) -> bool:
        """Check decorator pattern."""
        try:
            tree = ast.parse(code)
            has_decorator = any(
                isinstance(node, ast.FunctionDef) and node.decorator_list
                for node in ast.walk(tree)
            )
            has_wraps = 'wraps' in code or 'functools.wraps' in code
            has_inner = any(
                isinstance(node, ast.FunctionDef) and node.name == 'wrapper'
                for node in ast.walk(tree)
            )
            return has_decorator and has_wraps and has_inner
        except SyntaxError:
            return False
    
    tests.append(CodingTestCase(
        name="decorator_pattern",
        prompt="Write a Python decorator `timer` that measures function execution time using functools.wraps. Include an inner `wrapper` function.",
        category="code_completion",
        validator=validate_decorator,
        difficulty="medium",
    ))
    
    # ---- ALGORITHM TESTS ----
    
    def validate_two_sum(code: str) -> bool:
        """Check two-sum solution."""
        try:
            exec_globals = {}
            exec(code, exec_globals)
            for name in ['two_sum', 'twoSum', 'find_indices']:
                if name in exec_globals and callable(exec_globals[name]):
                    result = exec_globals[name]([2, 7, 11, 15], 9)
                    if sorted(result) == [0, 1]:
                        return True
            return False
        except Exception:
            return False
    
    tests.append(CodingTestCase(
        name="two_sum_problem",
        prompt="Write a Python function two_sum(nums, target) that returns indices of two numbers that add up to target. Example: two_sum([2,7,11,15], 9) returns [0,1].",
        category="algorithm",
        validator=validate_two_sum,
        difficulty="medium",
    ))
    
    def validate_merge_sort(code: str) -> bool:
        """Check merge sort implementation."""
        try:
            exec_globals = {}
            exec(code, exec_globals)
            for name in ['merge_sort', 'mergesort', 'sort']:
                if name in exec_globals and callable(exec_globals[name]):
                    result = exec_globals[name]([3, 1, 4, 1, 5, 9, 2, 6])
                    if result == [1, 1, 2, 3, 4, 5, 6, 9]:
                        return True
            return False
        except Exception:
            return False
    
    tests.append(CodingTestCase(
        name="merge_sort_implementation",
        prompt="Write a Python function merge_sort(arr) that implements the merge sort algorithm. Test: merge_sort([3,1,4,1,5,9,2,6]) should return [1,1,2,3,4,5,6,9].",
        category="algorithm",
        validator=validate_merge_sort,
        difficulty="hard",
    ))
    
    # ---- API / WEB TESTS ----
    
    def validate_fastapi(code: str) -> bool:
        """Check FastAPI endpoint structure."""
        try:
            contains_app = 'FastAPI()' in code or 'FastAPI(' in code
            contains_decorator = '@app.get' in code or '@app.post' in code or '@app.api' in code
            contains_async_def = 'async def' in code
            return contains_app and contains_decorator and contains_async_def
        except Exception:
            return False
    
    tests.append(CodingTestCase(
        name="fastapi_hello_world",
        prompt="Write a FastAPI application with a GET endpoint at /hello that returns {'message': 'Hello World'}.",
        category="api_development",
        validator=validate_fastapi,
        difficulty="easy",
    ))
    
    def validate_pandas_groupby(code: str) -> bool:
        """Check pandas groupby pattern."""
        try:
            contains_groupby = '.groupby(' in code
            contains_mean = '.mean()' in code or "agg('mean')" in code
            contains_pd = 'pd.' in code or 'pandas' in code
            return contains_groupby and contains_mean and contains_pd
        except Exception:
            return False
    
    tests.append(CodingTestCase(
        name="dataframe_groupby_mean",
        prompt="Using pandas, write code to group a DataFrame 'df' by 'category' column and compute the mean of 'value' column.",
        category="data_science",
        validator=validate_pandas_groupby,
        difficulty="easy",
    ))
    
    return tests


def run_benchmark(model, tokenizer=None, verbose: bool = True) -> Dict[str, Any]:
    """
    Run the coding benchmark.
    
    Args:
        model: The NovaRCV model (or any text generation model)
        tokenizer: Tokenizer for encoding/decoding
        verbose: Print detailed results
        
    Returns:
        Dict with results
    """
    tests = get_benchmarks()
    results = []
    passed = 0
    failed = 0
    
    print("=" * 70)
    print("CODING BENCHMARK")
    print("=" * 70)
    
    for i, test in enumerate(tests):
        if verbose:
            print(f"\n[{i+1}/{len(tests)}] {test.name} ({test.difficulty})")
            print(f"  Prompt: {test.prompt[:80]}...")
        
        # Generate code from model
        try:
            if tokenizer and hasattr(model, 'generate'):
                input_ids = tokenizer.encode(test.prompt, return_tensors='pt')
                generated = model.generate(input_ids, max_new_tokens=200)
                response = tokenizer.decode(generated[0], skip_special_tokens=True)
            elif hasattr(model, 'generate'):
                # Direct use
                response = model.generate(test.prompt, max_length=200)
            else:
                response = model(test.prompt)
            
            # Extract code block if present
            code = response
            if '```python' in response:
                code = response.split('```python')[1].split('```')[0]
            elif '```' in response:
                code = response.split('```')[1].split('```')[0]
            
            # Validate
            success = test.validator(code)
            
        except Exception as e:
            if verbose:
                print(f"  ERROR during evaluation: {e}")
            success = False
            code = ""
        
        if success:
            passed += 1
            status = "✅ PASS"
        else:
            failed += 1
            status = "❌ FAIL"
        
        results.append({
            'name': test.name,
            'category': test.category,
            'difficulty': test.difficulty,
            'passed': success,
            'prompt': test.prompt,
            'response': code if success else None,
        })
        
        if verbose:
            print(f"  Result: {status}")
            if not success and len(code) > 0:
                print(f"  Got: {code[:100]}...")
    
    total = len(tests)
    accuracy = (passed / total) * 100 if total > 0 else 0
    
    print(f"\n{'=' * 70}")
    print(f"CODING BENCHMARK RESULTS")
    print(f"{'=' * 70}")
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    # Breakdown by category
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = {'pass': 0, 'total': 0}
        categories[cat]['total'] += 1
        if r['passed']:
            categories[cat]['pass'] += 1
    
    print(f"\nBreakdown by category:")
    for cat, stats in sorted(categories.items()):
        cat_acc = (stats['pass'] / stats['total']) * 100
        print(f"  {cat}: {stats['pass']}/{stats['total']} ({cat_acc:.1f}%)")
    
    return {
        'benchmark': 'coding',
        'total': total,
        'passed': passed,
        'failed': failed,
        'accuracy': accuracy,
        'results': results,
    }


if __name__ == '__main__':
    run_benchmark(None, verbose=True)