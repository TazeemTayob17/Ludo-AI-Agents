# Tests for training/csv_logger.py.

import csv

from training.csv_logger import CsvLogger


# Checks the header and logged rows are readable back exactly as written.
def test_log_writes_header_and_rows(tmp_path):
    path = tmp_path / "log.csv"
    logger = CsvLogger(path, fieldnames=["episode", "reward"])
    logger.log({"episode": 1, "reward": 0.5})
    logger.log({"episode": 2, "reward": -1.0})
    logger.close()

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [
        {"episode": "1", "reward": "0.5"},
        {"episode": "2", "reward": "-1.0"},
    ]


# Checks the file is flushed after each row, not only when closed.
def test_log_flushes_immediately(tmp_path):
    path = tmp_path / "log.csv"
    logger = CsvLogger(path, fieldnames=["episode"])
    logger.log({"episode": 1})
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"episode": "1"}]
    logger.close()
