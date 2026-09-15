package main

import "fmt"

// observation is everything the tool saw, and is written to the receipt
// whether the device qualifies or not. A refusal is evidence about the
// host and must be retained, not discarded.
type observation struct {
	Schema    string              `json:"schema"`
	UTC       string              `json:"utc"`
	Host      string              `json:"host"`
	SMIExit   int                 `json:"nvidia_smi_exit"`
	SMIError  string              `json:"nvidia_smi_error,omitempty"`
	GPUs      []map[string]string `json:"gpus"`
	CUDA      cudaProbe           `json:"cuda_probe"`
	Expect    expectation         `json:"expectation"`
	Qualifies bool                `json:"qualifies"`
	Refusals  []string            `json:"refusals"`
	Scope     string              `json:"scope"`
}

// cudaProbe is the functional half: what the driver itself answered.
// Listing a GPU is not health, so this is the gate.
type cudaProbe struct {
	InitOK      bool   `json:"init_ok"`
	InitError   string `json:"init_error,omitempty"`
	DeviceCount int    `json:"device_count"`
	Name        string `json:"name,omitempty"`
	Major       int    `json:"compute_major,omitempty"`
	Minor       int    `json:"compute_minor,omitempty"`
	TotalMem    uint64 `json:"total_mem_bytes,omitempty"`
	ProbeError  string `json:"probe_error,omitempty"`
}

// expectation holds the caller's explicitly supplied link expectation.
// Zero means "record it, do not gate on it": no healthy baseline for this
// machine's link was ever retained, and on an integrated part the link is
// not established as the path the weights travel. A threshold nobody
// measured is a guess, and a guess that fails a run is worse than no gate.
type expectation struct {
	LinkWidth int `json:"link_width,omitempty"`
	LinkGen   int `json:"link_gen,omitempty"`
}

// judge decides from the observations alone, so the decision is testable
// without a GPU. It returns the refusals in the order they were found;
// an empty slice means the device qualifies.
func judge(gpus []map[string]string, probe cudaProbe, smiExit int, smiErr string, expect expectation) []string {
	var refusals []string
	if smiErr != "" || smiExit != 0 {
		refusals = append(refusals, fmt.Sprintf("nvidia-smi failed: exit %d %s", smiExit, smiErr))
	}
	for i, row := range gpus {
		for _, field := range smiFields {
			if requiredFields[field] && isUnusable(row[field]) {
				refusals = append(refusals, fmt.Sprintf("gpu %d: %s is %q", i, field, row[field]))
			}
		}
		switch state, n := eccErrors(row); {
		case state == eccMalformed:
			refusals = append(refusals, fmt.Sprintf("gpu %d: uncorrected ECC errors is %q, neither a count nor an unsupported marker",
				i, row["ecc.errors.uncorrected.volatile.total"]))
		case state == eccCount && n > 0:
			refusals = append(refusals, fmt.Sprintf("gpu %d: %d uncorrected ECC errors", i, n))
		}
		if expect.LinkWidth > 0 {
			if n, ok := integer(row, "pcie.link.width.current"); !ok {
				refusals = append(refusals, fmt.Sprintf("gpu %d: link width unreadable but an expectation was given", i))
			} else if n < expect.LinkWidth {
				refusals = append(refusals, fmt.Sprintf("gpu %d: link width %d below expected %d", i, n, expect.LinkWidth))
			}
		}
		if expect.LinkGen > 0 {
			if n, ok := integer(row, "pcie.link.gen.current"); !ok {
				refusals = append(refusals, fmt.Sprintf("gpu %d: link gen unreadable but an expectation was given", i))
			} else if n < expect.LinkGen {
				refusals = append(refusals, fmt.Sprintf("gpu %d: link gen %d below expected %d", i, n, expect.LinkGen))
			}
		}
	}
	// The functional probe is the gate. nvidia-smi can list a device that
	// cuInit then refuses with NO_DEVICE, which is exactly the state this
	// tool exists to catch, so a listing never rescues a failed probe.
	switch {
	case !probe.InitOK:
		refusals = append(refusals, "cuda: driver did not initialize: "+probe.InitError)
	case probe.DeviceCount < 1:
		refusals = append(refusals, "cuda: driver reports no devices")
	case probe.ProbeError != "":
		refusals = append(refusals, "cuda: device did not answer: "+probe.ProbeError)
	case probe.Name == "":
		refusals = append(refusals, "cuda: device reported an empty name")
	case probe.TotalMem == 0:
		refusals = append(refusals, "cuda: device reported zero memory")
	}
	return refusals
}
