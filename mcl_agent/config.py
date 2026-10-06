"""Shared configuration and source-column names used by the agent."""

from pathlib import Path

DEFAULT_LEASE_FOLDER = Path.home() / "Desktop" / "Lease Documents"
SUPPORTED_LEASE_EXTENSIONS = {".pdf", ".docx", ".txt"}
# Assante operating context supplied by the user: each MCL line is one device;
# Erika Laincy is an authorized lease signatory, not necessarily the end user.
AUTHORIZED_ORGANIZATION_SIGNERS = {
    "ASSANTE WEALTH MANAGEMENT (CANADA) LTD.": {"ERIKA LAINCY"},
}
IDENTIFIER_COLUMNS = [
    "LEASE NUMBER", "SO#", "CURRENT SERIAL #", "ORIGINAL SERIAL #",
    "SERIAL # CHANGE 1", "SERIAL # CHANGE 2", "SERIAL # CHANGE 3",
    "SERIAL # CHANGE 4", "SERIAL # CHANGE 5", "SERIAL # CHANGE 6",
]
USER_CONTEXT_COLUMNS = [
    "REGION", "CURRENT USER (AFTER RE-ASSIGNMENTS)", "GOOD NAME",
    "LEASEE/ COMPANY", "LEASE STATUS", "USER STATUS", "COMMENTS", "PRODUCT",
    "LEASE AGE (YEAR)", "SHIPMENT DATE (LEASE START DATE)",
    "USER E-MAIL", "LEASE NUMBER", "SO#", "CURRENT SERIAL #",
]
