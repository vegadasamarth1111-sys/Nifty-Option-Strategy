import re

with open('backtest_strategies.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove all non-latin1 characters from print statements only (safe approach: strip all non-ASCII from the whole file's string literals in print calls is complex)
# Simpler: replace all emoji/non-ASCII chars outside of HTML strings
# Strategy: replace emoji in Python source code (not in HTML template)
# We'll just strip all characters > 0x7F that appear in Python print() calls

def strip_non_ascii_in_prints(text):
    lines = text.split('\n')
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('print('):
            # Replace non-ASCII chars with simple ASCII equivalents
            new_line = ''
            for ch in line:
                if ord(ch) > 127:
                    new_line += '?'
                else:
                    new_line += ch
            result.append(new_line)
        else:
            result.append(line)
    return '\n'.join(result)

content = strip_non_ascii_in_prints(content)

with open('backtest_strategies.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed successfully')
