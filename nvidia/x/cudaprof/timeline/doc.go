// Package timeline reconstructs the device timeline of a cudaprof capture.
//
// pprof samples are aggregates: they sum durations and drop the start and stop
// timestamps. The per-record detail lives in the capture journals attached to
// the capture. [Journals] returns them, refusing a capture whose journals are
// absent, omitted, or lossy, and [Decode] reads them into [Records], refusing a
// journal that does not decode to completion: a timeline built from a
// surviving prefix reads exactly like one built from the whole run.
//
// [Events] and [APIRecords] give the stream events and host API calls of a
// capture, from the pprof samples where they carry timestamps and from the
// journals otherwise. [Union] totals the time a set of intervals covers.
package timeline
