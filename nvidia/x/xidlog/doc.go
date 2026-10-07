// Package xidlog reads supplied NVRM XID and GPU-to-PCI observations.
// It does not open /dev/kmsg, run commands, classify health or infer a boot's
// completeness. Product, driver, boot and input identity come from the caller.
// Catalog applicability is product-specific; L4 and GB10 are not inferred from
// other products. XID154 text uses xidcatalog's bounded transition parser.
//
// Text input consists of newline-delimited supplied log lines. Kmsg input must
// preserve Linux /dev/kmsg's one-record-per-Read semantics, including continuation
// metadata. A concatenated dump should be split into records by its caller,
// then passed to ParseKmsg. Monotonic microseconds are not wall-clock time.
// Fragments are retained without joining interleaved messages. EPIPE becomes
// an explicit gap; no inferred lost-count or preceding event is fabricated.
//
// Reader bounds retained input and record populations. The caller owns blocking
// I/O, cancellation, open/close and any previous-boot journal/text retrieval.
package xidlog
