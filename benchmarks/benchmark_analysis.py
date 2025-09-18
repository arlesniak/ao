import re
import csv
import argparse
from pathlib import Path
import sys
import builtins
import inspect
_original_print = builtins.print

# State to hold computed column widths once the header is seen
_table_state = {
    'active': False,
    'widths': None,
    'fields': None,
}

def _compute_and_print_table_header(header_line: str, rows):
    fields = header_line.split('\t')
    widths = [len(f) for f in fields]
    # rows are dicts with keys matching the header fields
    for r in rows or []:
        for i, f in enumerate(fields):
            val = str(r.get(f, ''))
            if len(val) > widths[i]:
                widths[i] = len(val)

    _table_state['active'] = True
    _table_state['widths'] = widths
    _table_state['fields'] = fields

    header_parts = [fields[i].ljust(widths[i]) for i in range(len(fields))]
    _original_print('  '.join(header_parts))
    _original_print('  '.join('-' * widths[i] for i in range(len(fields))))

def _print_table_row(line: str):
    values = line.split('\t')
    parts = []
    for i, v in enumerate(values):
        w = _table_state['widths'][i] if i < len(_table_state['widths']) else len(v)
        parts.append(v.ljust(w))
    _original_print('  '.join(parts))

def print(*args, **kwargs):
    # Intercept only the specific tab-separated header/rows printed by the script.
    try:
        if args and isinstance(args[0], str) and '\t' in args[0]:
            s = args[0]
            # Try to obtain 'rows' from the caller so we can compute column widths for the header
            caller_frame = inspect.currentframe().f_back
            caller_rows = None
            if caller_frame is not None:
                caller_rows = caller_frame.f_locals.get('rows') or caller_frame.f_globals.get('rows')

            # Heuristic: header line contains tabs and matches number of columns (7 -> 6 tabs)
            if not _table_state['active'] and s.count('\t') >= 1:
                # treat the first tabbed line as header
                _compute_and_print_table_header(s, caller_rows)
                return
            if _table_state['active']:
                _print_table_row(s)
                return
    except Exception:
        # Fall back to normal print on any error
        pass

    _original_print(*args, **kwargs)

# Replace built-in print in this module so subsequent prints in the script use the table-aware printer.
builtins.print = print
#!/usr/bin/env python3
"""
Parse benchmarks/benchmark_low_bit_adam_bmg.txt and extract entries for
calls to benchmarks/benchmark_low_bit_adam.py. Outputs a CSV/table with:
    model, batch_size, optim, optim_cpu_offload, compile, speed, max_memory

Usage:
    python benchmark_analysis.py --input benchmarks/benchmark_low_bit_adam_bmg.txt --output parsed.csv
"""

CMD_TARGET = "benchmarks/benchmark_low_bit_adam.py"

def normalize_text(text: str) -> str:
    # Join lines continued with backslash and remove extra newlines so commands
    # that are split across lines become single lines.
    text = re.sub(r'\\\s*\n', ' ', text)
    # Also replace interior newlines that are preceded by whitespace (likely wraps)
    # with a single space to avoid splitting commands accidentally.
    return text

def find_command_chunks(text: str):
    # Find each command line that invokes the target script, and capture the
    # following log chunk up to the next such command or EOF.
    cmd_re = re.compile(
        r'(^|\n)(?P<cmd>python\b[^\n]*' + re.escape(CMD_TARGET) + r'[^\n]*)',
        flags=re.IGNORECASE
    )
    starts = []
    for m in cmd_re.finditer(text):
        starts.append((m.start('cmd'), m.end('cmd'), m.group('cmd').strip()))
    chunks = []
    for i, (s, e, cmd) in enumerate(starts):
        end_idx = starts[i+1][0] if i+1 < len(starts) else len(text)
        chunk = text[e:end_idx]
        chunks.append((cmd, chunk))
    return chunks

def extract_flag(cmd: str, name: str):
    # supports forms: --name value   or --name=value
    # value may be quoted with single or double quotes
    pat = re.compile(r'--' + re.escape(name) + r'(?:=|\s+)(?P<val>"[^"]*"|\'[^\']*\'|\S+)')
    m = pat.search(cmd)
    if not m:
        return ""
    val = m.group('val')
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    return val

def extract_speed(chunk: str):
    # Look for "Epoch ... 8.44it/s" style lines and capture the it/s token
    m = re.search(r'Epoch[^\n\r]*?((?:\d+(?:\.\d+)?)\s*it/s)', chunk, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # fallback: any standalone "xx.xx it/s"
    m2 = re.search(r'(\d+(?:\.\d+)?)\s*it/s', chunk, flags=re.IGNORECASE)
    if m2:
        return m2.group(0).strip()
    return ""

def extract_max_memory(chunk: str):
    # Look for "Max memory used: 1.53 GB" or similar
    m = re.search(r'Max memory used:\s*([0-9]+(?:\.[0-9]+)?\s*[GMK]B)', chunk, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""

def extract_compile_flag(cmd: str) -> str:
    # Detect presence of --compile. If given with a value that is false-like, return 'no'.
    m = re.search(r'(?<!\S)--compile(?:\b|=)', cmd)
    if not m:
        return "no"
    # check for forms like --compile=False or --compile false
    mval = re.search(r'--compile(?:=|\s+)(?P<val>\S+)', cmd)
    if mval:
        val = mval.group('val').strip().strip('"').strip("'").lower()
        if val in ('0', 'false', 'no', 'off'):
            return "no"
        return "yes"
    return "yes"

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', type=Path, default=Path('benchmark_low_bit_adam_bmg.txt'))
    p.add_argument('--output', '-o', type=Path, default=Path('benchmark_low_bit_adam_parsed.csv'))
    args = p.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    text = args.input.read_text(encoding='utf-8', errors='ignore')
    text = normalize_text(text)
    cmd_chunks = find_command_chunks(text)

    rows = []
    for cmd, chunk in cmd_chunks:
        model = extract_flag(cmd, 'model')
        batch = extract_flag(cmd, 'batch_size') or extract_flag(cmd, 'batch-size')
        optim_name = extract_flag(cmd, 'optim')
        optim_off = extract_flag(cmd, 'optim_cpu_offload')
        compile_present = extract_compile_flag(cmd)
        speed = extract_speed(chunk)
        max_mem = extract_max_memory(chunk)
        rows.append({
            'model': model,
            'batch_size': batch,
            'optim': optim_name,
            'optim_cpu_offload': optim_off,
            'compile': compile_present,
            'speed': speed,
            'max_memory': max_mem
        })

    # sort rows by model name (case-insensitive)
    rows.sort(key=lambda r: (r.get('model') or '').strip().lower())

    # write CSV
    fieldnames = ['model', 'batch_size', 'optim', 'optim_cpu_offload', 'compile', 'speed', 'max_memory']
    with args.output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # print quick table to stdout
    if rows:
        print(f"Wrote {len(rows)} rows to {args.output}")
        print("model\tbatch_size\toptim\toptim_cpu_offload\tcompile\tspeed\tmax_memory")
        for r in rows:
            print(f"{r['model']}\t{r['batch_size']}\t{r['optim']}\t{r['optim_cpu_offload']}\t{r['compile']}\t{r['speed']}\t{r['max_memory']}")
    else:
        print("No matching commands found.", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()