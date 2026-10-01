package main

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/tmc/lib/nvidia/x/cudaprof/isolation"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

func isolationExportGate(result isolation.Result) tracecontract.Gate {
	disposition := "unchecked"
	switch result.State {
	case "clear", "sampled clear", "sampled_clear":
		disposition = "passed"
	case "contaminated":
		disposition = "failed"
	}
	return tracecontract.Gate{Disposition: disposition, Reason: fmt.Sprintf("isolation %s: %v; %s", result.State, result.Reasons, result.Limits), Provenance: "bound sampled observer receipt; independent of collector completeness"}
}

func writeIsolationTrace(out io.Writer, result isolation.Result, identity traceExportIdentity, id, clock uint64) error {
	w := perfetto.NewWriter(out)
	w.TimestampClockID = clock
	track := tracecontract.Track{UUID: id, Name: "GPU isolation qualification", Role: "isolation", Clock: identity.sha256 + ":" + identity.arm + ":export-relative", Arm: identity.arm, CaptureSHA256: identity.sha256, Provenance: "bound observer evidence", Aggregation: "none"}
	if err := track.Write(w, identity.shift); err != nil {
		return err
	}
	// The receipt remains an instant: observer clock equivalence to CUPTI is
	// not established merely by binding the capture's host sampling window.
	data, err := json.Marshal(result)
	if err != nil {
		return err
	}
	return w.WriteFlowInstant(id, identity.shift, "GPU isolation: "+result.State, "qualification", []perfetto.DebugAnnotation{perfetto.NewStringAnnotation("isolation_state", result.State), perfetto.NewStringAnnotation("isolation_receipt_sha256", result.SHA256), perfetto.NewStringAnnotation("isolation_evidence", string(data)), perfetto.NewStringAnnotation("meaning", "sampled residency evidence does not quantify timing impact; observer intervals require an explicit CUPTI clock binding")}, nil, nil)
}
