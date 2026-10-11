// Package tracecontract defines versioned source, track, receipt, and diagnostic
// finding metadata for cudaprof traces.
//
// Findings compute their admission from the evidence each finding actually requires:
// probe coverage, backend support, record loss by family, capture bounds,
// correlation outcomes, clock domains, and experimental intervention receipts.
// Official Perfetto TrackEvent annotations carry the metadata; display names
// and imported slice IDs do not identify evidence.
package tracecontract
