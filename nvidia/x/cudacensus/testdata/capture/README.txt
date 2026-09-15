Trimmed capture fixture.

Source: $TMPDIR/gputrace-perfetto-20260831/dgx/go-cold.gpucapture,
an MLX CUDA inference run captured on the DGX on 2026-08-31.

Trimming, from the original events.jsonl.1745273.jsonl and
events.app.1745263.jsonl:

  events.trimmed.jsonl      the capture_meta and clock_sync records, then
                            every kernel record wholly inside the first
                            decode token span (132 launches, 10 distinct
                            symbols), plus the first memcpy and the first
                            memset in the run so those kinds are exercised.
  events.app.trimmed.jsonl  the prefill span, the first token span, and the
                            decode span.

Records are copied verbatim except for JSON key spacing. No timestamp, symbol,
grid, block or byte count was altered, so the window is a real token and the
per-token launch count in it is the real one.

The file names do not match the events.app.*.jsonl / events.*.jsonl globs by
accident: events.app.trimmed.jsonl is the app file and events.trimmed.jsonl is
the device file, exactly as ReadDir separates them.
