#!/usr/bin/env python3
"""Parse and summarize low-bit benchmark logs."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple


CMD_TARGET = "benchmarks/benchmark_low_bit_adam.py"
FIELD_NAMES = (
    "model",
    "batch_size",
    "optim",
    "optim_cpu_offload",
    "compile",
    "speed",
    "max_memory",
)


@dataclass
class BenchmarkEntry:
    """Structured representation of a single benchmark run."""

    model: str = ""
    batch_size: str = ""
    optim: str = ""
    optim_cpu_offload: str = ""
    compile: str = "no"
    speed: str = ""
    max_memory: str = ""

    def as_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in FIELD_NAMES}

    def as_row(self) -> List[str]:
        return [getattr(self, name) for name in FIELD_NAMES]


class LogParser:
    """Parse benchmark log files into structured entries."""

    _COMMAND_PATTERN = re.compile(
        r"(^|\n)(?P<cmd>python\b[^\n]*" + re.escape(CMD_TARGET) + r"[^\n]*)",
        flags=re.IGNORECASE,
    )

    def parse(self, text: str) -> List[BenchmarkEntry]:
        normalized = self._join_continued_lines(text)
        chunk_iter = self._iter_command_chunks(normalized)
        entries = [self._parse_chunk(cmd, chunk) for cmd, chunk in chunk_iter]
        entries.sort(key=lambda entry: entry.model.lower())
        return entries

    @staticmethod
    def _join_continued_lines(text: str) -> str:
        # Merge commands that use trailing "\" continuation characters.
        text = re.sub(r"\\\s*\n", " ", text)
        return text

    def _iter_command_chunks(self, text: str) -> Iterator[Tuple[str, str]]:
        matches = list(self._COMMAND_PATTERN.finditer(text))
        for index, match in enumerate(matches):
            start, end = match.start("cmd"), match.end("cmd")
            command = match.group("cmd").strip()
            next_start = matches[index + 1].start("cmd") if index + 1 < len(matches) else len(text)
            yield command, text[end:next_start]

    def _parse_chunk(self, command: str, chunk: str) -> BenchmarkEntry:
        return BenchmarkEntry(
            model=self._extract_flag(command, "model"),
            batch_size=self._extract_flag(command, "batch_size")
            or self._extract_flag(command, "batch-size"),
            optim=self._extract_flag(command, "optim"),
            optim_cpu_offload=self._extract_flag(command, "optim_cpu_offload"),
            compile=self._extract_compile_flag(command),
            speed=self._extract_speed(chunk),
            max_memory=self._extract_max_memory(chunk),
        )

    @staticmethod
    def _extract_flag(command: str, name: str) -> str:
        pattern = re.compile(
            r"--" + re.escape(name) + r"(?:=|\s+)(?P<value>\"[^\"]*\"|'[^']*'|\S+)"
        )
        match = pattern.search(command)
        if not match:
            return ""
        value = match.group("value")
        if value.startswith(("\"", "'")) and value.endswith(("\"", "'")):
            return value[1:-1]
        return value

    @staticmethod
    def _extract_speed(chunk: str) -> str:
        match = re.search(r"Epoch[^\n\r]*?((?:\d+(?:\.\d+)?)\s*it/s)", chunk, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        fallback = re.search(r"(\d+(?:\.\d+)?)\s*it/s", chunk, flags=re.IGNORECASE)
        return fallback.group(0).strip() if fallback else ""

    @staticmethod
    def _extract_max_memory(chunk: str) -> str:
        match = re.search(r"Max memory used:\s*([0-9]+(?:\.[0-9]+)?\s*[GMK]B)", chunk, flags=re.IGNORECASE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_compile_flag(command: str) -> str:
        if not re.search(r"(?<!\S)--compile(?:\b|=)", command):
            return "no"
        match = re.search(r"--compile(?:=|\s+)(?P<value>\S+)", command)
        if not match:
            return "yes"
        value = match.group("value").strip().strip("\"").strip("'").lower()
        return "no" if value in {"0", "false", "no", "off"} else "yes"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("benchmark_low_bit_adam_bmg.txt"),
        help="Benchmark log to analyze.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("benchmark_low_bit_adam_parsed.csv"),
        help="CSV file that will store parsed entries.",
    )
    return parser


def write_csv(entries: Sequence[BenchmarkEntry], destination: Path) -> None:
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, FIELD_NAMES)
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry.as_dict())


def format_table(entries: Sequence[BenchmarkEntry]) -> str:
    if not entries:
        return ""

    column_widths = [len(name) for name in FIELD_NAMES]
    for entry in entries:
        for index, value in enumerate(entry.as_row()):
            column_widths[index] = max(column_widths[index], len(value))

    def pad_row(values: Iterable[str]) -> str:
        return "  ".join(value.ljust(column_widths[idx]) for idx, value in enumerate(values))

    header = pad_row(FIELD_NAMES)
    separator = pad_row("-" * width for width in column_widths)
    rows = [pad_row(entry.as_row()) for entry in entries]
    return "\n".join([header, separator, *rows])


def main() -> None:
    arg_parser = build_arg_parser()
    args = arg_parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    log_text = args.input.read_text(encoding="utf-8", errors="ignore")
    entries = LogParser().parse(log_text)

    if not entries:
        print("No matching commands found.", file=sys.stderr)
        sys.exit(1)

    write_csv(entries, args.output)

    print(f"Wrote {len(entries)} rows to {args.output}")
    print(format_table(entries))


if __name__ == "__main__":
    main()