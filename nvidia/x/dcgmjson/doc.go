// Package dcgmjson reads bounded, single-iteration dcgmi diag -j output from
// the pinned DCGM 4.6.1 producer. It never runs diagnostics or loads DCGM.
//
// Status is the literal JSON status. This producer renders warning results as
// Fail and omits not-run results; neither distinction can be recovered from a
// missing row. Missing fields remain unavailable, not zero or passing results.
// Numeric warning error_id values use dcgm.ErrorCode, not DCGM return codes.
// Unknown fields and the original JSON are retained without interpretation.
package dcgmjson
