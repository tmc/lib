package traceanalysis

import (
	_ "embed"
	"fmt"
)

//go:embed sql/cudaprof/views.sql
var sqlViews string

// SQL returns the canonical v1 PerfettoSQL for a named query. The caller owns
// local executable qualification, importer checks, bounded output and receipt
// validation. An empty SQL table is missing evidence, not a zero observation.
func SQL(name string) (string, error) {
	statements := map[string]string{
		"inventory":        "SELECT * FROM cudaprof_inventory_v1 ORDER BY role,clock,arm,journal,device",
		"qualifications":   "SELECT * FROM cudaprof_qualifications_v1",
		"delivery-cadence": "SELECT * FROM cudaprof_delivery_cadence_v1",
		"kernel-shapes":    "SELECT * FROM cudaprof_kernel_shapes_v1 ORDER BY capture_sha256,arm,journal,device,raw_symbol",
		"recorded-gaps":    "SELECT capture_sha256,arm,journal,device,clock, 'recorded_union' AS measurement,NULL AS start_ns,NULL AS end_ns,recorded_union_ns AS duration_ns FROM cudaprof_device_totals_v1 UNION ALL SELECT capture_sha256,arm,journal,device,clock,'recorded_gap',start_ns,end_ns,gap_ns FROM cudaprof_recorded_gaps_v1 ORDER BY capture_sha256,arm,journal,device,clock,measurement,start_ns",
		"unmatched":        "SELECT * FROM cudaprof_unmatched_v1",
		"correlations":     "SELECT * FROM cudaprof_correlations_v1",
	}
	statement, ok := statements[name]
	if !ok {
		return "", fmt.Errorf("trace SQL: unknown named query %q", name)
	}
	return sqlViews + "\n" + statement + ";\n", nil
}
