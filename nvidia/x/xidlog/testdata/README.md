# Constructed log fixtures

All records here are constructed parser controls, not captures from a device.
Driver labels535/550/570/580 are caller provenance inputs and do not certify
branch-specific grammar. Linux kmsg framing follows the primary ABI document:
https://www.kernel.org/doc/Documentation/ABI/testing/dev-kmsg (read2026-10-07).
Each actual device Read is one record, potentially with metadata continuation
lines; EPIPE is reader loss, and c is a fragment hint. No journal export schema
or boot completeness is inferred. UUID association is a supplied log fact.
