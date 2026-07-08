"""Scan a Python file for odd-count } sequences inside f-string text (outside {{ }} pairs),
which would cause SyntaxError. Print all suspicious lines."""
import re

with open('backtest_strategies_v2.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

in_fstring = False
issues = []

for i, line in enumerate(lines, 1):
    # Very rough heuristic: look for lines with odd counts of consecutive }
    # that aren't inside Python dict/set literals
    raw = line.rstrip()
    # Find runs of 3+ closing braces
    for m in re.finditer(r'\}{3,}', raw):
        run = m.group()
        # Odd count of } in a run is potentially bad in f-string
        if len(run) % 2 == 1:
            issues.append((i, raw.strip(), len(run)))

if issues:
    print(f"Found {len(issues)} suspicious lines:")
    for lineno, text, count in issues:
        print(f"  Line {lineno} ({count} braces): {text[:100]}")
else:
    print("No odd-brace issues found!")
