# Official schemas; synthetic row fixtures

Two named releases are supported:

- Nsight Systems 2024.2, EXPORT_SCHEMA_VERSION 3.11.0:
  https://archive.docs.nvidia.com/nsight-systems/2024.2/UserGuide/index.html#sqlite-schema-reference
- Nsight Systems 2025.3, EXPORT_SCHEMA_VERSION 3.20.2:
  https://archive.docs.nvidia.com/nsight-systems/2025.3/UserGuide/index.html#sqlite-schema-reference

`schema-*.sql` copies the actual CREATE TABLE bodies for the nine supported
tables from those official manuals. Comments and column order are retained;
these are documented schemas, not measured report databases. The manuals are
retained with exact SHA256 identities in the author UPSTREAM-PINS.json.
`schema.go` is generated from these SQL bodies by the self-contained
`testdata/generate_schema.py` (`go generate`), then gofmt. It checks full declared column order/type/
nullability of the selected tables. Other tables/providers are not interpreted.

`rows-*.sql` contains explicitly SYNTHETIC host control rows: a kernel, memcpy,
memset, CUDA API, NVTX NCCL-named annotation and OS runtime event. No Nsight,
CUDA, CUPTI or NCCL capture was run. CPU tests create actual SQLite databases
from the official schema plus these synthetic rows and import them through a
test-only read-only adapter; no binary database is staged or preserved as a
hardware artifact. Constructed in-memory rows have the same explicit status.

Both manuals describe NCCL tracing as NVTX annotations. This subset retains
NVTX text, domain/range identity and raw JSON/binary payload fields but does not
guess undocumented NCCL payload semantics or classify a string as a completed
collective. A missing provider is unavailable, not a zero observation.

Documented export recipe: `nsys export --type sqlite --lazy=false report.nsys-rep`.
This is documentation only: the importer does not execute it. Export options
can normalize timestamps to UTC and shift them; the caller must retain those
actual options in Source.Clock/ShiftNS. Capture session UTC metadata is kept
separately and does not imply that rows were normalized. Correlation is confined
to one report SHA and serialized target/process identity, not timestamps or
another run/pass. Generic database subprocess/custody controls are the Reader
owner's responsibility, outside the NVIDIA table semantics package.

NVIDIA documentation copyright and license terms remain applicable:
https://archive.docs.nvidia.com/nsight-systems/2025.3/CopyrightAndLicenses/index.html
