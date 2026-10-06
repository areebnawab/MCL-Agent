"""Utilities for reading and normalizing the MCL Excel workbook."""

from pathlib import Path

import pandas as pd

MCL_HEADER_MARKERS = {
    "REGION", "IDX", "INITIAL USER NAME", "LEASE NUMBER", "LEASE AGE (YEAR)",
    "CURRENT USER (AFTER RE-ASSIGNMENTS)", "LEASEE/ COMPANY", "USER E-MAIL",
    "SHIPMENT DATE (LEASE START DATE)", "PRODUCT", "SO#",
}


def normalize_column_name(value: object) -> str:
    """Convert Excel header variations into stable uppercase column names."""
    return " ".join(str(value or "").replace("\n", " ").split()).strip().upper()


def _find_header_row(preview: pd.DataFrame) -> tuple[int, int] | None:
    """Find a likely MCL header row in a small, header-free worksheet preview."""
    for row_index, row in preview.iterrows():
        values = [normalize_column_name(value) for value in row.tolist()]
        recognized = sum(value in MCL_HEADER_MARKERS for value in values)
        # Several distinctive headers together make a title row an unlikely match.
        has_identifier = any(value in {"LEASE NUMBER", "IDX", "SO#"} for value in values)
        if recognized >= 3 and has_identifier:
            return int(row_index), recognized
    return None


def load_mcl_workbook(file_path: Path) -> pd.DataFrame:
    """Find an MCL-like header in any worksheet and load the rows beneath it."""
    if not file_path.exists():
        raise FileNotFoundError(f"MCL workbook not found: {file_path}")
    workbook = pd.ExcelFile(file_path)
    diagnostics = []
    candidates = []

    # Look through a few top rows on every tab to find the real table header.
    for sheet_name in workbook.sheet_names:
        preview = pd.read_excel(workbook, sheet_name=sheet_name, header=None, nrows=100, dtype=object)
        header = _find_header_row(preview)
        if header is None:
            diagnostics.append(f"{sheet_name}: no recognizable MCL header")
            continue
        header_row, score = header
        frame = pd.read_excel(workbook, sheet_name=sheet_name, header=header_row, dtype=object)
        frame.columns = [normalize_column_name(column) for column in frame.columns]
        frame = frame.loc[:, [column for column in frame.columns if column and not column.startswith("UNNAMED:")]]
        frame = frame.dropna(how="all").fillna("")
        if frame.empty:
            diagnostics.append(f"{sheet_name}: found a likely header on row {header_row + 1}, but no data rows below it")
            continue
        candidates.append((score, sheet_name, frame))
        diagnostics.append(f"{sheet_name}: found {len(frame)} data rows (header row {header_row + 1})")

    if not candidates:
        sheet_details = "; ".join(diagnostics) if diagnostics else "The workbook has no worksheets."
        expected = ", ".join(sorted(MCL_HEADER_MARKERS))
        raise ValueError(
            "Could not find a populated MCL table in this workbook. "
            f"Worksheet scan: {sheet_details}. Looked for at least three MCL headers, including a lease number, IDX, or SO# column. "
            f"Recognized headers include: {expected}."
        )

    # Prefer the worksheet with the most recognized MCL headers, keeping tab order for ties.
    return max(candidates, key=lambda candidate: candidate[0])[2]
