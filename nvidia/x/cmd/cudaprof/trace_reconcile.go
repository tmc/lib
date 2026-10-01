package main

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"

	"github.com/tmc/lib/nvidia/x/cudaprof/traceanalysis"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
)

// Reconciliation compares importer-visible raw records with the bounded
// export receipt, never with summary/hierarchy rows. Supplied receipts remain
// provenance claims; agreement does not establish native collector completeness.
func reconcileTraceImport(data []byte, processor string, limits traceProcessorLimits) (traceEvidence, error) {
	sql, err := traceanalysis.SQL("inventory")
	if err != nil {
		return traceEvidence{}, err
	}
	// Replace only the final named-query SELECT, retaining canonical role views.
	const final = "SELECT * FROM cudaprof_inventory_v1 ORDER BY role,clock,arm,journal,device;\n"
	if len(sql) < len(final) || sql[len(sql)-len(final):] != final {
		return traceEvidence{}, errors.New("trace reconciliation: canonical SQL changed")
	}
	sql = sql[:len(sql)-len(final)] + `SELECT 'receipt' AS section, EXTRACT_ARG(arg_set_id,'debug.cudaprof_input_sha256') AS identity, HEX(EXTRACT_ARG(arg_set_id,'debug.cudaprof_receipt_json')) AS value FROM slice WHERE EXTRACT_ARG(arg_set_id,'debug.cudaprof_schema')='cudaprof.receipt.v1'
UNION ALL SELECT 'raw_device',capture_sha256||':'||activity_kind,CAST(COUNT(*) AS TEXT) FROM cudaprof_raw_device_v1 GROUP BY capture_sha256,activity_kind;`
	backend, err := queryTraceProcessorLimits(data, processor, sql, limits)
	if err != nil {
		return traceEvidence{}, err
	}
	unavailable := traceEvidence{"unavailable", "no versioned export receipt; imported counts cannot be reconciled", "Trace Processor raw role rows"}
	if len(backend.Rows) == 0 || len(backend.Rows[0]) != 3 || backend.Rows[0][0] != "section" {
		return unavailable, errors.New("trace reconciliation: incompatible result schema")
	}
	receipts := map[string]tracecontract.Receipt{}
	counts := map[string]int64{}
	for _, row := range backend.Rows[1:] {
		if len(row) != 3 {
			return unavailable, errors.New("trace reconciliation: malformed row")
		}
		switch row[0] {
		case "receipt":
			var receipt tracecontract.Receipt
			encoded, err := hex.DecodeString(row[2])
			if err != nil {
				return unavailable, errors.New("trace reconciliation: invalid receipt encoding")
			}
			if err := json.Unmarshal(encoded, &receipt); err != nil {
				return unavailable, fmt.Errorf("trace reconciliation receipt: %w", err)
			}
			if _, err := receipt.JSON(); err != nil {
				return unavailable, err
			}
			if row[1] != receipt.InputSHA256 {
				return unavailable, errors.New("trace reconciliation: receipt source mismatch")
			}
			if _, exists := receipts[row[1]]; exists {
				return unavailable, errors.New("trace reconciliation: duplicate source receipt")
			}
			receipts[row[1]] = receipt
		case "raw_device":
			count, err := strconv.ParseInt(row[2], 10, 64)
			if err != nil || count < 0 {
				return unavailable, errors.New("trace reconciliation: invalid count")
			}
			counts[row[1]] = count
		default:
			return unavailable, errors.New("trace reconciliation: unknown row section")
		}
	}
	if len(receipts) == 0 {
		return unavailable, nil
	}
	for hash, receipt := range receipts {
		for _, kind := range []string{"kernel", "memcpy", "memset"} {
			key := hash + ":" + kind
			if counts[key] != receipt.RawDeviceRecords[kind] {
				return traceEvidence{"failed", fmt.Sprintf("%s imported %d %s records; export expected %d", hash, counts[key], kind, receipt.RawDeviceRecords[kind]), "export receipt and importer raw rows"}, nil
			}
			delete(counts, key)
		}
	}
	if len(counts) != 0 {
		return traceEvidence{"failed", "imported raw records lack a corresponding source receipt", "export receipt and importer raw rows"}, nil
	}
	return traceEvidence{"passed", "raw kernel/copy/memset counts agree with export receipts; native collector completeness remains unavailable", "export receipt and importer raw rows"}, nil
}
