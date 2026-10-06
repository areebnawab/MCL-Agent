# MCL Renewal Assistant

A local Streamlit application that reads an MCL workbook, searches a Lease Documents folder, interprets a natural-language question, and creates a traceable Excel report. The workflow is implemented as explicit LangGraph nodes so each stage is easy to inspect and change.

## Start the application

Use Python 3.10 or newer. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
cp .env.example .env               # Optional; set an API key for richer language understanding
streamlit run app.py
```

The app opens in a browser on the same computer. Upload the MCL `.xlsx` file. Lease documents are optional: the default folder is `~/Desktop/Lease Documents`, and you can change the folder or upload files in the sidebar. Use **Upload lease PDFs** to add one or more PDFs directly. The app extracts selectable text page by page, shows the extracted text for review, and includes it in the agent's analysis. DOCX and TXT files can also be uploaded. If no lease documents are present, the MCL query still runs and the report flags records whose renewal dates cannot be determined.

## MLflow observability

Each analysis creates an MLflow run in the `MCL Renewal Agent` experiment and records typed `CHAIN` spans for query planning, lease extraction, record matching, and response generation. LangChain tracing is enabled so configured ChatOpenAI calls appear as typed LLM spans with provider-reported token usage. Span timing and success/error status are recorded by the MLflow span lifecycle. The root analysis span contains the full query, MCL row data, and lease-document text passed to the graph; stage spans contain their actual inputs and outputs, including extracted evidence, matched results, and the final answer. The generated workbook binary is not uploaded; the trace records its filename and byte size. The default tracking server is `http://127.0.0.1:5000`. Set `MLFLOW_TRACKING_URI` or `MLFLOW_EXPERIMENT_NAME` in the environment (or `.env`) to override the defaults. Open the tracking URI in a browser to review runs and traces.

Run parameters and metrics include aggregate row/document counts, duration, result counts, model configuration, and success/failure status. Trace payloads include query and source data, so configure the MLflow tracking server and its access controls for the sensitivity of those records. MLflow connection or logging failures do not stop the local analysis; the app shows a sanitized reason when telemetry or trace spans are unavailable and links to the configured MLflow UI.

## Ask a question

Examples:

- “Which Microsoft software licenses expire in the next 90 days?”
- “Show overdue Teams renewals for Acme in Ontario.”
- “Find software renewals in the next 6 months.”

## Assante matching rules

For the supplied Assante MCL data, each MCL line item represents one device; device totals count matching rows, not the `TOTAL COUNT` cell. A user can have separate hardware and software leases with different lease numbers, so a lease-number difference alone does not reject a license document. The agent checks the organization, end user, authorized signer, and product. The user has identified Erika Laincy as Assante's authorized lease signatory; a PDF listing her as contact supports the signer check, but does not establish that she is the end user or account owner. Product names still need to agree: Microsoft 365 Apps for Business is not automatically treated as a BP/Business Premium license.

The MCL header list provided contains `REGION` and `LEASEE/ COMPANY`, but no explicit office field or renewal-date field. `LEASE AGE (YEAR)` is treated as elapsed years, so an age of `1.0` means 12 elapsed months. The report uses an explicit end date from the lease document when available. Otherwise, it calculates the end date by adding the document's initial lease term to `SHIPMENT DATE (LEASE START DATE)`. It compares Lease Age with the elapsed time since that MCL start date and flags differences greater than two months for review. The report therefore shows region/company context and preserves the source date or term evidence it can extract.

## What the report contains

The downloaded workbook has a `Summary` tab and a `Renewal Results` tab. Results include available account, user, lease age/start date, software-license context, matched lease document, candidate end date, date basis, evidence snippet, and a review status. An explicit source renewal date is distinguished from a projected lease term end; projected dates are flagged to confirm that the software renewal follows the lease term. Joins use exact normalized lease number or sales order. Serial numbers are read from MCL but are not currently extracted from lease documents as join keys. Rows without exact document matches, usable term/end dates, or readable files are exposed as review items rather than silently assumed to be accurate.

## Privacy and extraction modes

Without `OPENAI_API_KEY`, the app runs locally and uses labeled date patterns and common renewal phrases, plus simple query rules. This fallback is intentionally conservative and may miss unusual lease layouts or complex questions. Setting `OPENAI_API_KEY` enables structured natural-language filter parsing and lease fact extraction through the configured OpenAI model; document text is then sent to that provider. Follow your organization's data handling policy before enabling it. `OPENAI_MODEL` can override the model name.

Scanned/image-only PDFs need OCR before text extraction can find dates. The app reports when an uploaded PDF has no selectable text; OCR is not included in this starter. Date extraction is not a substitute for validating an unusual contract clause. Reported dates and quoted excerpts should be checked against the source document during rollout, especially before using the workflow for client notifications.

## Project layout

```text
app.py                         Streamlit frontend and file selection
mcl_agent/ingestion.py         Excel reading and header normalization
mcl_agent/lease_reader.py      PDF, DOCX, and TXT text extraction
mcl_agent/query_parser.py      Natural-language filters, optional LLM
mcl_agent/matching.py          Evidence extraction, joins, result flags
mcl_agent/graph.py             LangGraph nodes and control flow
mcl_agent/exporter.py          Formatted Excel report generation
architecture.mmd               Editable Mermaid architecture diagram
```

See [architecture.mmd](architecture.mmd) for the standalone architecture diagram.
