"""eval_eim.py - does the system REALLY get better?  Fixed problems, same settings, results kept per version.

    python eval_eim.py --check                  validate the problem set itself (no model needed)
    python eval_eim.py --label v3               run every problem with the real model, append the result to eval_history.jsonl
    python eval_eim.py --compare                compare the last two runs (or: --compare v2 v3)
    python eval_eim.py --selftest               tests of this script (no model, no network)

Options:  --iterations 4  --candidates 3  --problems my_problems.json  --with-memory

Fair-comparison rules built in
  * the learned memory and the repair log are OFF by default (--with-memory turns them on). Otherwise the engine would
    learn from the very problems it is graded on and the number would only measure memorisation.
  * the result cache is disabled, so repeated runs really run.
  * the hidden-test split is the engine's own: a problem counts as solved only if the visible AND hidden tests pass.
  * with few problems, one flipped problem is a big swing; the comparison prints a 95% interval and says so.

Own problems: a JSON list of {"id", "task", "tests": [at least 5 asserts], "reference": "<correct code>",
"initial_code": "<optional buggy code>"}. `--check` proves the reference passes and the buggy code does not.
On a free ZeroGPU Space the GPU is only available inside a Gradio request, so run this on a local/Colab GPU
(or a paid Space); each problem is its own GPU call (`gpu` from app.py) so no single call hits the duration limit.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from typing import Callable, Sequence

HISTORY = os.environ.get("EIM_EVAL_HISTORY", "eval_history.jsonl")

PROBLEMS: list[dict] = [
    {"id": "palindrome",
     "task": "Write is_palindrome(s) -> bool. Ignore case and every non-alphanumeric character.",
     "tests": ["assert is_palindrome('A man, a plan, a canal: Panama') is True", "assert is_palindrome('race a car') is False",
               "assert is_palindrome('') is True", "assert is_palindrome('No lemon, no melon') is True",
               "assert is_palindrome('ab') is False", "assert is_palindrome('12321') is True"],
     "reference": "def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]"},
    {"id": "fizzbuzz",
     "task": "Write fizzbuzz(n) -> list of strings for 1..n: 'Fizz' for multiples of 3, 'Buzz' for 5, 'FizzBuzz' for both, else the number as text.",
     "tests": ["assert fizzbuzz(0) == []", "assert fizzbuzz(1) == ['1']", "assert fizzbuzz(3) == ['1', '2', 'Fizz']",
               "assert fizzbuzz(5)[-1] == 'Buzz'", "assert fizzbuzz(15)[-1] == 'FizzBuzz'", "assert len(fizzbuzz(30)) == 30"],
     "reference": "def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n        out.append('FizzBuzz' if i % 15 == 0 else 'Fizz' if i % 3 == 0 else 'Buzz' if i % 5 == 0 else str(i))\n    return out"},
    {"id": "two_sum",
     "task": "Write two_sum(nums, target) -> [i, j] with i < j and nums[i] + nums[j] == target. Exactly one pair exists.",
     "tests": ["assert two_sum([2, 7, 11, 15], 9) == [0, 1]", "assert two_sum([3, 2, 4], 6) == [1, 2]",
               "assert two_sum([3, 3], 6) == [0, 1]", "assert two_sum([-1, -2, -3, -4, -5], -8) == [2, 4]",
               "assert two_sum([0, 4, 3, 0], 0) == [0, 3]", "assert two_sum([1, 5, 9, 14], 23) == [2, 3]"],
     "reference": "def two_sum(nums, target):\n    seen = {}\n    for j, x in enumerate(nums):\n        if target - x in seen:\n            return [seen[target - x], j]\n        seen[x] = j"},
    {"id": "flatten",
     "task": "Write flatten(items) that flattens arbitrarily nested lists into one flat list, keeping order. Strings are not lists.",
     "tests": ["assert flatten([]) == []", "assert flatten([1, [2, [3, [4]]]]) == [1, 2, 3, 4]", "assert flatten([[1, 2], [3], []]) == [1, 2, 3]",
               "assert flatten(['ab', ['cd']]) == ['ab', 'cd']", "assert flatten([[[[]]]]) == []", "assert flatten([1, 2, 3]) == [1, 2, 3]"],
     "reference": "def flatten(items):\n    out = []\n    for x in items:\n        if isinstance(x, list):\n            out.extend(flatten(x))\n        else:\n            out.append(x)\n    return out"},
    {"id": "roman",
     "task": "Write roman_to_int(s) converting a Roman numeral (I V X L C D M, with subtractive pairs like IV, IX, XL, XC, CD, CM) to an integer.",
     "tests": ["assert roman_to_int('III') == 3", "assert roman_to_int('IV') == 4", "assert roman_to_int('IX') == 9",
               "assert roman_to_int('LVIII') == 58", "assert roman_to_int('MCMXCIV') == 1994", "assert roman_to_int('CDXLIV') == 444"],
     "reference": "def roman_to_int(s):\n    v = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n    total = 0\n    for i, c in enumerate(s):\n        if i + 1 < len(s) and v[c] < v[s[i + 1]]:\n            total -= v[c]\n        else:\n            total += v[c]\n    return total"},
    {"id": "merge_intervals",
     "task": "Write merge_intervals(intervals) -> sorted list of [start, end] lists where overlapping or touching intervals are merged.",
     "tests": ["assert merge_intervals([]) == []", "assert merge_intervals([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]",
               "assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]", "assert merge_intervals([[5, 6], [1, 2]]) == [[1, 2], [5, 6]]",
               "assert merge_intervals([[1, 10], [2, 3]]) == [[1, 10]]", "assert merge_intervals([[2, 2]]) == [[2, 2]]"],
     "reference": "def merge_intervals(intervals):\n    out = []\n    for a, b in sorted(intervals):\n        if out and a <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], b)\n        else:\n            out.append([a, b])\n    return out"},
    {"id": "word_count",
     "task": "Write word_count(text) -> dict mapping each lowercase word to its count. Words are runs of letters; punctuation and digits separate words.",
     "tests": ["assert word_count('') == {}", "assert word_count('a a b') == {'a': 2, 'b': 1}", "assert word_count('Hello, hello!') == {'hello': 2}",
               "assert word_count('it is, is it?') == {'it': 2, 'is': 2}", "assert word_count('x1y') == {'x': 1, 'y': 1}",
               "assert word_count('...') == {}"],
     "reference": "import re\n\ndef word_count(text):\n    out = {}\n    for w in re.findall(r'[a-zA-Z]+', text):\n        w = w.lower()\n        out[w] = out.get(w, 0) + 1\n    return out"},
    {"id": "brackets",
     "task": "Write is_balanced(s) -> bool: True when every (), [] and {} in s is closed in the right order. Other characters are ignored.",
     "tests": ["assert is_balanced('') is True", "assert is_balanced('([]{})') is True", "assert is_balanced('(]') is False",
               "assert is_balanced('((') is False", "assert is_balanced('a(b)c') is True", "assert is_balanced('}{') is False"],
     "reference": "def is_balanced(s):\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n    for c in s:\n        if c in '([{':\n            stack.append(c)\n        elif c in pairs:\n            if not stack or stack.pop() != pairs[c]:\n                return False\n    return not stack"},
    {"id": "binary_search_repair",
     "task": "Fix binary_search(arr, x): arr is sorted ascending; return the index of x or -1 if absent.",
     "initial_code": "def binary_search(arr, x):\n    lo, hi = 0, len(arr) - 1\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == x:\n            return mid\n        if arr[mid] < x:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
     "tests": ["assert binary_search([], 1) == -1", "assert binary_search([5], 5) == 0", "assert binary_search([1, 3, 5, 7], 7) == 3",
               "assert binary_search([1, 3, 5, 7], 1) == 0", "assert binary_search([1, 3, 5, 7], 4) == -1", "assert binary_search([2, 4], 4) == 1"],
     "reference": "def binary_search(arr, x):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == x:\n            return mid\n        if arr[mid] < x:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1"},
    {"id": "is_anagram",
     "task": "Write is_anagram(a, b) -> bool. Ignore spaces, punctuation, and case; compare letter/digit counts.",
     "tests": ["assert is_anagram('listen', 'silent') is True", "assert is_anagram('Dormitory', 'dirty room') is True", "assert is_anagram('aabb', 'abab') is True", "assert is_anagram('rat', 'car') is False", "assert is_anagram('', '') is True", "assert is_anagram('a', '') is False"],
     "reference": "from collections import Counter\ndef is_anagram(a, b):\n    clean = lambda s: ''.join(c.lower() for c in s if c.isalnum())\n    return Counter(clean(a)) == Counter(clean(b))"},
    {"id": "gcd",
     "task": "Write gcd(a, b) -> non-negative greatest common divisor, including zero and negative inputs.",
     "tests": ["assert gcd(54, 24) == 6", "assert gcd(0, 5) == 5", "assert gcd(5, 0) == 5", "assert gcd(0, 0) == 0", "assert gcd(-12, 18) == 6", "assert gcd(17, 13) == 1"],
     "reference": "def gcd(a, b):\n    a, b = abs(a), abs(b)\n    while b:\n        a, b = b, a % b\n    return a"},
    {"id": "unique_preserve",
     "task": "Write unique_preserve(items) returning first occurrences only, preserving order, for hashable items.",
     "tests": ["assert unique_preserve([]) == []", "assert unique_preserve([1, 2, 1, 3, 2]) == [1, 2, 3]", "assert unique_preserve(['a', 'a', 'b']) == ['a', 'b']", "assert unique_preserve([0, 1, 0]) == [0, 1]", "assert unique_preserve([3]) == [3]", "assert unique_preserve([2, 1, 2, 1]) == [2, 1]"],
     "reference": "def unique_preserve(items):\n    seen, out = set(), []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            out.append(item)\n    return out"},
    {"id": "chunked",
     "task": "Write chunked(items, size) returning consecutive lists of at most size; raise ValueError if size <= 0.",
     "tests": ["assert chunked([], 2) == []", "assert chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]", "assert chunked([1, 2], 2) == [[1, 2]]", "assert chunked([1], 8) == [[1]]", "assert chunked([1, 2, 3], 1) == [[1], [2], [3]]", "try:\n    chunked([1], 0)\nexcept ValueError:\n    pass\nelse:\n    assert False"],
     "reference": "def chunked(items, size):\n    if size <= 0:\n        raise ValueError('size must be positive')\n    return [items[i:i + size] for i in range(0, len(items), size)]"},
    {"id": "max_subarray_sum",
     "task": "Write max_subarray_sum(nums) returning the largest sum of a non-empty contiguous subarray; input is non-empty.",
     "tests": ["assert max_subarray_sum([1, -2, 3, 4, -1]) == 7", "assert max_subarray_sum([-5, -2, -9]) == -2", "assert max_subarray_sum([4]) == 4", "assert max_subarray_sum([0, 0]) == 0", "assert max_subarray_sum([2, -1, 2, 3, -8, 4]) == 6", "assert max_subarray_sum([-1, 2, 3, -2]) == 5"],
     "reference": "def max_subarray_sum(nums):\n    current = best = nums[0]\n    for x in nums[1:]:\n        current = max(x, current + x)\n        best = max(best, current)\n    return best"},
    {"id": "transpose",
     "task": "Write transpose(matrix) for a rectangular list of lists; an empty matrix returns [].",
     "tests": ["assert transpose([]) == []", "assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]", "assert transpose([[1], [2]]) == [[1, 2]]", "assert transpose([[1, 2]]) == [[1], [2]]", "assert transpose([[7]]) == [[7]]", "assert transpose([[], []]) == []"],
     "reference": "def transpose(matrix):\n    return [list(row) for row in zip(*matrix)] if matrix else []"},
    {"id": "is_prime",
     "task": "Write is_prime(n) -> bool for integer n, returning False for n < 2.",
     "tests": ["assert is_prime(-7) is False", "assert is_prime(0) is False", "assert is_prime(1) is False", "assert is_prime(2) is True", "assert is_prime(97) is True", "assert is_prime(99) is False"],
     "reference": "def is_prime(n):\n    if n < 2: return False\n    if n % 2 == 0: return n == 2\n    d = 3\n    while d * d <= n:\n        if n % d == 0: return False\n        d += 2\n    return True"},
    {"id": "fibonacci",
     "task": "Write fibonacci(n) returning the nth Fibonacci number with F(0)=0 and F(1)=1; n is non-negative.",
     "tests": ["assert fibonacci(0) == 0", "assert fibonacci(1) == 1", "assert fibonacci(2) == 1", "assert fibonacci(10) == 55", "assert fibonacci(20) == 6765", "assert fibonacci(6) == 8"],
     "reference": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"},
    {"id": "first_unique_char",
     "task": "Write first_unique_char(s) returning the index of the first non-repeating character, or -1 if none.",
     "tests": ["assert first_unique_char('leetcode') == 0", "assert first_unique_char('loveleetcode') == 2", "assert first_unique_char('aabb') == -1", "assert first_unique_char('') == -1", "assert first_unique_char('z') == 0", "assert first_unique_char('aabcc') == 2"],
     "reference": "from collections import Counter\ndef first_unique_char(s):\n    counts = Counter(s)\n    return next((i for i, c in enumerate(s) if counts[c] == 1), -1)"},
    {"id": "rotate_list",
     "task": "Write rotate_list(items, k) returning a new list rotated right by k; handle empty lists and negative k.",
     "tests": ["assert rotate_list([], 3) == []", "assert rotate_list([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]", "assert rotate_list([1, 2, 3], 0) == [1, 2, 3]", "assert rotate_list([1, 2, 3], 3) == [1, 2, 3]", "assert rotate_list([1, 2, 3], -1) == [2, 3, 1]", "assert rotate_list([1], 100) == [1]"],
     "reference": "def rotate_list(items, k):\n    if not items: return []\n    k %= len(items)\n    return items[-k:] + items[:-k] if k else list(items)"},
    {"id": "flatten_dict_paths",
     "task": "Write flatten_dict(d) mapping nested dictionary leaves to dot-separated paths; empty nested dictionaries are omitted.",
     "tests": ["assert flatten_dict({}) == {}", "assert flatten_dict({'a': 1}) == {'a': 1}", "assert flatten_dict({'a': {'b': 2}}) == {'a.b': 2}", "assert flatten_dict({'a': {'b': 2}, 'c': 3}) == {'a.b': 2, 'c': 3}", "assert flatten_dict({'x': {}, 'y': {'z': {}}}) == {}", "assert flatten_dict({'a': {'b': {'c': 4}}}) == {'a.b.c': 4}"],
     "reference": "def flatten_dict(d):\n    out = {}\n    def visit(obj, prefix):\n        for key, value in obj.items():\n            path = f'{prefix}.{key}' if prefix else str(key)\n            if isinstance(value, dict): visit(value, path)\n            else: out[path] = value\n    visit(d, '')\n    return out"},
    {"id": "run_length_encode",
     "task": "Write run_length_encode(s) returning (character, count) pairs for each consecutive run.",
     "tests": ["assert run_length_encode('') == []", "assert run_length_encode('aaabbc') == [('a', 3), ('b', 2), ('c', 1)]", "assert run_length_encode('x') == [('x', 1)]", "assert run_length_encode('abab') == [('a', 1), ('b', 1), ('a', 1), ('b', 1)]", "assert run_length_encode('11122') == [('1', 3), ('2', 2)]", "assert run_length_encode('aaaa') == [('a', 4)]"],
     "reference": "def run_length_encode(s):\n    out = []\n    for c in s:\n        if out and out[-1][0] == c: out[-1] = (c, out[-1][1] + 1)\n        else: out.append((c, 1))\n    return out"},
    {"id": "clamp",
     "task": "Write clamp(value, low, high) limiting value to inclusive [low, high]. Assume low <= high.",
     "tests": ["assert clamp(5, 0, 10) == 5", "assert clamp(-2, 0, 10) == 0", "assert clamp(12, 0, 10) == 10", "assert clamp(0, 0, 10) == 0", "assert clamp(10, 0, 10) == 10", "assert clamp(3, 3, 3) == 3"],
     "reference": "def clamp(value, low, high):\n    return max(low, min(value, high))"},
    {"id": "is_subsequence",
     "task": "Write is_subsequence(s, t) -> bool: return whether s appears in t in order, not necessarily contiguously.",
     "tests": ["assert is_subsequence('', 'abc') is True", "assert is_subsequence('ace', 'abcde') is True", "assert is_subsequence('aec', 'abcde') is False", "assert is_subsequence('abc', 'abc') is True", "assert is_subsequence('long', 'short') is False", "assert is_subsequence('aab', 'aaab') is True"],
     "reference": "def is_subsequence(s, t):\n    it = iter(t)\n    return all(any(c == x for x in it) for c in s)"},
    {"id": "product_except_self",
     "task": "Write product_except_self(nums) returning products of all other elements without division; input length >= 1.",
     "tests": ["assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]", "assert product_except_self([0, 1, 2]) == [2, 0, 0]", "assert product_except_self([0, 0, 2]) == [0, 0, 0]", "assert product_except_self([5]) == [1]", "assert product_except_self([-1, 2, -3]) == [-6, 3, -2]", "assert product_except_self([2, 2]) == [2, 2]"],
     "reference": "def product_except_self(nums):\n    out = [1] * len(nums)\n    p = 1\n    for i, x in enumerate(nums): out[i] = p; p *= x\n    p = 1\n    for i in range(len(nums) - 1, -1, -1): out[i] *= p; p *= nums[i]\n    return out"},
    {"id": "longest_common_prefix",
     "task": "Write longest_common_prefix(strings) returning the longest prefix shared by every string; empty input returns ''.",
     "tests": ["assert longest_common_prefix([]) == ''", "assert longest_common_prefix(['flower', 'flow', 'flight']) == 'fl'", "assert longest_common_prefix(['dog', 'racecar', 'car']) == ''", "assert longest_common_prefix(['same', 'same']) == 'same'", "assert longest_common_prefix(['a']) == 'a'", "assert longest_common_prefix(['', 'abc']) == ''"],
     "reference": "def longest_common_prefix(strings):\n    if not strings: return ''\n    prefix = strings[0]\n    for s in strings[1:]:\n        while not s.startswith(prefix): prefix = prefix[:-1]\n        if not prefix: break\n    return prefix"}
]


# ---------------------------------------------------------------------------------------------------------------------
def check_problems(verifier, problems: Sequence[dict]) -> list[str]:
    """Problems with a wrong test (reference fails) or a fake 'repair' (buggy code already passes) poison every number."""
    errors = []
    for p in problems:
        if len(p["tests"]) < 5:
            errors.append(f"{p['id']}: needs at least 5 tests (hidden split)")
        ref = verifier.run(p["reference"], p["tests"])
        if ref.pass_rate < 1.0:
            errors.append(f"{p['id']}: the reference solution fails {ref.total - ref.passed}/{ref.total} of its own tests")
        if p.get("initial_code") and verifier.run(p["initial_code"], p["tests"]).pass_rate >= 1.0:
            errors.append(f"{p['id']}: the 'buggy' initial code already passes every test")
    return errors


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    denom = 1 + z * z / n
    return max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom)


def evaluate(engine, problems: Sequence[dict], iterations: int, candidates: int, label: str,
             wrap: Callable | None = None, meta: dict | None = None) -> dict:
    from eim_plus import Report

    def solve(problem: dict) -> tuple[bool, int, int, int]:
        report = None
        lm = getattr(engine, "lm", None)
        original_generate = getattr(lm, "generate", None) if lm is not None else None
        counter = [0]
        if original_generate is not None:
            def counted_generate(prompts, *args, **kwargs):
                try: counter[0] += len(prompts)
                except TypeError: counter[0] += 1
                return original_generate(prompts, *args, **kwargs)
            try: lm.generate = counted_generate
            except (AttributeError, TypeError): original_generate = None
        try:
            for event in engine.run(problem["task"], problem["tests"], problem.get("initial_code"), iterations, candidates,
                                    0.25, mutation=False):
                if isinstance(event, Report):
                    report = event
        finally:
            if original_generate is not None:
                try: lm.generate = original_generate
                except (AttributeError, TypeError): pass
        solved = (report is not None and report.best.result.pass_rate == 1.0
                  and (report.hidden is None or report.hidden.pass_rate == 1.0))
        return solved, (report.iterations if report else 0), (report.restarts if report else 0), counter[0]

    run_one = wrap(solve) if wrap else solve                 # on ZeroGPU: one GPU call per problem
    rows = []
    for problem in problems:
        started = time.perf_counter()
        try:
            solved, rounds, restarts, attempts = run_one(problem)
        except Exception as error:                           # a crash is a failure, not a reason to lose the whole run
            solved, rounds, restarts, attempts = False, 0, 0, 0
            print(f"  ! {problem['id']}: {type(error).__name__}: {error}")
        seconds = time.perf_counter() - started
        rows.append({"id": problem["id"], "solved": bool(solved), "seconds": round(seconds, 2),
                     "iterations": rounds, "restarts": restarts, "attempts": attempts})
        print(f"  {'OK  ' if solved else 'FAIL'} {problem['id']:<22} {seconds:6.1f}s  rounds={rounds} restarts={restarts} candidates={attempts}")
    n, ok = len(rows), sum(r["solved"] for r in rows)
    lo, hi = wilson(ok, n)
    record = {"label": label, "ts": int(time.time()), "n": n, "solved": ok, "rate": round(ok / n, 4) if n else 0.0,
              "ci95": [round(lo, 4), round(hi, 4)], "mean_seconds": round(sum(r["seconds"] for r in rows) / max(1, n), 2),
              "mean_attempts": round(sum(r["attempts"] for r in rows) / max(1, n), 2),
              "mean_restarts": round(sum(r["restarts"] for r in rows) / max(1, n), 2),
              "settings": {"iterations": iterations, "candidates": candidates, **(meta or {})}, "problems": rows}
    return record


def append_history(record: dict, path: str = HISTORY) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_history(path: str = HISTORY) -> list[dict]:
    out = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def compare(before: dict, after: dict) -> str:
    pts = (after["rate"] - before["rate"]) * 100
    rel = (after["rate"] / before["rate"] - 1) * 100 if before["rate"] else float("nan")
    d_time = (after["mean_seconds"] / before["mean_seconds"] - 1) * 100 if before["mean_seconds"] else float("nan")
    lines = [f"{before['label']}  ->  {after['label']}",
             f"success   {before['solved']}/{before['n']} ({before['rate']:.0%})  ->  {after['solved']}/{after['n']} ({after['rate']:.0%})"
             f"   = {pts:+.1f} points ({rel:+.1f}% relative)",
             f"time/task {before['mean_seconds']:.1f}s -> {after['mean_seconds']:.1f}s ({d_time:+.1f}%)",
             f"mean candidates/task {before.get('mean_attempts', 0):.1f} -> {after.get('mean_attempts', 0):.1f}; "
             f"mean restarts/task {before.get('mean_restarts', 0):.1f} -> {after.get('mean_restarts', 0):.1f}",
             f"95% interval  before {before['ci95'][0]:.0%}-{before['ci95'][1]:.0%}   after {after['ci95'][0]:.0%}-{after['ci95'][1]:.0%}"]
    prev = {p["id"]: p["solved"] for p in before["problems"]}
    now = {p["id"]: p["solved"] for p in after["problems"]}
    fixed = [i for i in now if now[i] and prev.get(i) is False]
    broke = [i for i in now if not now[i] and prev.get(i) is True]
    lines.append(f"newly solved: {', '.join(fixed) or '-'}   |   regressed: {', '.join(broke) or '-'}")
    if before["ci95"][1] >= after["ci95"][0] and after["ci95"][1] >= before["ci95"][0]:
        lines.append("note: the intervals overlap - with this many problems the difference may be noise. "
                     "Add problems (--problems) before drawing conclusions.")
    if before["settings"] != after["settings"]:
        lines.append(f"warning: settings differ ({before['settings']} vs {after['settings']}) - not a like-for-like comparison")
    if set(prev) != set(now):
        lines.append("warning: the two runs used different problem IDs - compare only after using the same benchmark set")
    return "\n".join(lines)


def _flag(name: str, default: str | None = None) -> str | None:
    if name in sys.argv:
        i = sys.argv.index(name)
        return sys.argv[i + 1] if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--") else default
    return None


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    problems = PROBLEMS
    custom = _flag("--problems")
    if custom:
        with open(custom, encoding="utf-8") as handle:
            problems = json.load(handle)

    if "--compare" in sys.argv:
        history = load_history()
        labels = [a for a in sys.argv[sys.argv.index("--compare") + 1:] if not a.startswith("--")]
        if len(labels) >= 2:
            pick = {r["label"]: r for r in history}
            if labels[0] not in pick or labels[1] not in pick:
                print("unknown label; known:", ", ".join(pick) or "(none)")
                return 1
            print(compare(pick[labels[0]], pick[labels[1]]))
        elif len(history) >= 2:
            print(compare(history[-2], history[-1]))
        else:
            print("need at least two runs in", HISTORY)
            return 1
        return 0

    from app import CFG, EUTV, gpu, get_lm
    from eim_plus import EIMPlus, SmartMemory, hardware_profile
    if "--check" in sys.argv:
        errors = check_problems(EUTV(CFG), problems)
        print("\n".join(errors) if errors else f"all {len(problems)} problems are sound")
        return 1 if errors else 0

    os.environ["EIM_RESULT_CACHE"] = "0"

    class _LM:
        def generate(self, prompts, temperature, max_new_tokens):
            return get_lm().generate(prompts, temperature, max_new_tokens)

    with_memory = "--with-memory" in sys.argv
    engine = EIMPlus(_LM(), EUTV(CFG), CFG, SmartMemory(path=CFG.memory_path if with_memory else ""),
                     repair_log="auto" if with_memory else None)
    label = _flag("--label", time.strftime("run-%Y%m%d-%H%M"))
    iterations, candidates = int(_flag("--iterations", "4")), int(_flag("--candidates", "3"))
    print(f"evaluating '{label}': {len(problems)} problems, iterations={iterations}, candidates={candidates}, "
          f"profile={hardware_profile().name}, memory={'on' if with_memory else 'off'}")
    record = evaluate(engine, problems, iterations, candidates, label, wrap=gpu,
                      meta={"model": CFG.model_name, "profile": hardware_profile().name, "memory": with_memory})
    append_history(record)
    history = load_history()
    print(f"\nsolved {record['solved']}/{record['n']} ({record['rate']:.0%}, 95% interval "
          f"{record['ci95'][0]:.0%}-{record['ci95'][1]:.0%}), mean {record['mean_seconds']:.1f}s per problem -> {HISTORY}")
    if len(history) >= 2:
        print("\n" + compare(history[-2], history[-1]))
    return 0


# ---------------------------------------------------------------------------------------------------------------------
def selftest() -> int:
    from app import Config, EUTV, _ScriptedLM
    from eim_plus import EIMPlus, Profile, SmartMemory
    failures = 0

    def check(name: str, ok: bool) -> None:
        nonlocal failures
        print(("PASS " if ok else "FAIL ") + name)
        failures += not ok

    fast = Config(per_test_seconds=2, wall_seconds=10.0, cpu_seconds=8, memory_path="")
    verifier = EUTV(fast)
    # Keep this smoke-test bounded; the full 25-task reference set has an independent offline check
    # in test_eval_benchmarks.py. Real process startup is intentionally exercised on a representative subset.
    smoke_problems = PROBLEMS[:3]
    check("representative references pass and buggy code fails in real child processes", check_problems(verifier, smoke_problems) == [])
    bad = [{"id": "x", "task": "t", "tests": ["assert f() == 1"] * 5, "reference": "def f():\n    return 2"}]
    check("a wrong reference is reported", len(check_problems(verifier, bad)) == 1)
    check("wilson interval brackets the rate", wilson(6, 8)[0] < 0.75 < wilson(6, 8)[1] and wilson(0, 0) == (0.0, 0.0))

    flat = Profile("test", 4, 8.0, False, 8, False, 0, 0)
    os.environ["EIM_RESULT_CACHE"] = "0"
    perfect = EIMPlus(_ScriptedLM([p["reference"] for p in smoke_problems]), verifier, fast, SmartMemory(path=""), profile=flat,
                      repair_log=None)
    good = evaluate(perfect, smoke_problems, 2, 1, "good")
    check("a scripted model that returns the references solves all smoke problems", good["solved"] == len(smoke_problems) and good["rate"] == 1.0)
    nothing = EIMPlus(_ScriptedLM(["def nothing():\n    return None"]), verifier, fast, SmartMemory(path=""), profile=flat,
                      repair_log=None)
    # These per-problem FAIL lines are expected for the deliberately bad fake model;
    # hide them in self-test output so they cannot be mistaken for failing self-tests.
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        poor = evaluate(nothing, smoke_problems[:2], 1, 1, "poor")
    check("a model that returns nonsense solves nothing", poor["solved"] == 0)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "history.jsonl")
        append_history(poor, path)
        append_history(good, path)
        loaded = load_history(path)
        check("history round-trips", [r["label"] for r in loaded] == ["poor", "good"])
    text = compare({**good, "label": "A", "rate": 0.5, "solved": 1, "ci95": [0.2, 0.8]}, {**good, "label": "B"})
    check("comparison reports points, time and overlap warning", "+50.0 points" in text and "overlap" in text)
    print(f"\n{'All checks passed.' if not failures else str(failures) + ' check(s) FAILED.'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
