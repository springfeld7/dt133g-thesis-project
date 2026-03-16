import csv

from transtructiver.reporting.summary_logger import write_summary, write_summary_totals


def test_write_summary_creates_file(tmp_path):
    log_file = tmp_path / "summary_log.csv"

    write_summary("row_1", True, errors=[], log_path=str(log_file))

    assert log_file.exists()
    with open(log_file, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows == [["row_1", "1", ""]]


def test_write_summary_totals_appends_total_row(tmp_path):
    log_file = tmp_path / "summary_log.csv"

    write_summary("row_1", False, errors=["mismatch"], log_path=str(log_file))
    write_summary_totals(
        parsed_ok=1,
        parse_skipped=0,
        verified_ok=0,
        verified_fail=1,
        log_path=str(log_file),
    )

    with open(log_file, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == ["row_1", "0", "mismatch"]
    assert rows[1][0] == "TOTAL"
    assert rows[1][1] == "0/1"
