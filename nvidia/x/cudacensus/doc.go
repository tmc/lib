// Package cudacensus turns a GPU trace capture into a per-kernel census and,
// where the caller can supply a cost model, into roofline points.
//
// A capture directory holds newline-delimited JSON: device records
// (kernel, memcpy, memset, clock_sync) in events.*.jsonl and application
// spans in events.app.*.jsonl. ReadDir decodes both and puts the spans on
// the device clock, so a phase or token span can select the launches inside
// it.
//
// # Bytes are not in the capture
//
// A kernel record names the kernel and gives its grid, block, register count,
// static shared memory and device timestamps. It does not record any argument,
// any pointer, any tensor shape, or any byte count. Only memcpy and memset
// records carry bytes, and those are transfers, not kernels. Bytes moved by a
// kernel therefore cannot be derived from a capture alone: the caller must
// supply a Coster mapping a Launch to a Cost. Table and ParseTable provide one
// keyed by kernel symbol; testdata holds a worked example.
//
// # Nor is residency
//
// A Cost carries a Residency along with its Bytes, because a byte count only
// means something against the ceiling for the path those bytes travel: on a
// measured GB10 the same bytes attain 239.6 GB/s device-resident and
// 162.4 GB/s read through host page tables. A capture cannot supply this
// either. No record kind holds a pointer, an allocation size, or an
// allocation-to-launch mapping; the one allocator call that appears by name
// records neither what it returned nor how large it was. So residency, like
// bytes, is the model's to state, and stating it wrong is undetectable by
// arithmetic.
//
// Because the wrong ceiling is worse than no ceiling, an unstated residency is
// UnknownResidency and Points refuses it, returning an error naming every
// entry it would not judge rather than quietly defaulting to device-resident.
//
// Because a Coster rarely covers every kernel, Census marks each entry Modeled
// and Coverage reports the device time the model accounts for. Read the
// coverage before reading the bandwidth: a model covering a tenth of the
// device time says nothing about the workload.
//
// Basic use:
//
//	c, err := cudacensus.ReadDir("00-baseline.gpucapture")
//	span, _ := c.Span("decode")
//	entries := cudacensus.Census(c.Window(span.StartNs, span.EndNs), coster)
//	modeled, total := cudacensus.Coverage(entries)
package cudacensus
