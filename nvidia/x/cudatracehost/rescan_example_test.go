package cudatracehost_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudatracehost"
	"time"
)

func ExampleThreadRescan() {
	r := cudatracehost.ThreadRescan{Status: "generation_verified_attachment", MissedCalls: "unknown", BlindInterval: "between_scans_unknown"}
	fmt.Println(r.Status, r.MissedCalls)
	// Output: generation_verified_attachment unknown
}

func ExampleConfig_threadRescan() {
	cfg := cudatracehost.Config{Backend: "bpf", ThreadRescanInterval: time.Second, MaxThreadRescans: 32}
	fmt.Println(cfg.ThreadRescanInterval, cfg.MaxThreadRescans)
	// Output: 1s 32
}
