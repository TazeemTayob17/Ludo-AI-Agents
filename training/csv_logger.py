# Writes one row per episode to a CSV file, flushing immediately so progress is readable mid-run.

from __future__ import annotations

import csv
from pathlib import Path

# Opens the CSV file for writing and writes its header row.
class CsvLogger:

    # Creates (or overwrites) the CSV file with the given column names.
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()

    # Writes one row and flushes it to disk immediately.
    def log(self, row: dict) -> None:
        self._writer.writerow(row)
        self._file.flush()

    # Closes the underlying file.
    def close(self) -> None:
        self._file.close()
