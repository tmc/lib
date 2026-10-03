package cudasass

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
)

// PeakGPRValue derives peak live GPR count from ParseLifeRangeCount output.
// Only the exact entry is admitted. A missing or empty map is unavailable;
// a present zero count remains a real derived zero. This is not an allocated
// register count or compiler spill proof.
func PeakGPRValue(pressure map[string]map[uint64]RegisterPressure, source cudaevidence.Source) (cudaevidence.Value, error) {
	if err := source.Identity.Validate(); err != nil {
		return cudaevidence.Value{}, err
	}
	if strings.TrimSpace(source.Producer) == "" || strings.TrimSpace(source.Locator) == "" {
		return cudaevidence.Value{}, fmt.Errorf("pressure evidence: missing producer or locator")
	}
	rows, ok := pressure[source.Identity.Entry]
	if !ok || len(rows) == 0 {
		return cudaevidence.New(cudaevidence.Record{Source: source, Reason: "nvdisasm life-range rows unavailable for exact entry"})
	}
	peak := 0
	for _, row := range rows {
		if row.GPR < 0 || row.Pred < 0 || row.UGPR < 0 || row.UPred < 0 {
			return cudaevidence.Value{}, fmt.Errorf("pressure evidence: negative register count")
		}
		peak = max(peak, row.GPR)
	}
	number := float64(peak)
	return cudaevidence.New(cudaevidence.Record{Class: cudaevidence.Derived, Number: &number, Unit: "registers", Source: source,
		Basis: "maximum GPR over parsed nvdisasm life-range rows for exact entry; static dataflow interpretation, not allocated registers or spill proof"})
}
