package cudaprofagent

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
)

// A Count is how many instructions one program counter executed, as the
// hardware counted them: the function the counter belongs to, spelled the way
// the disassembly spells it, and the offset of the instruction inside it.
type Count struct {
	Function string
	Offset   uint64
	Value    int64
}

// Counts is what one measurement of one module produced. Module is the
// SHA-256 of the image the counts were measured against, hex encoded, which
// is what binds them to a module of the capture: the counter values say
// nothing about which of several builds of a kernel ran, and the image does.
type Counts struct {
	Module string
	Metric string
	Counts []Count
}

// WriteCounts publishes executed-instruction counts to the capture this
// process is running under, and reports whether there was a capture to
// publish to.
//
// The agent collects what CUPTI reports on its own: activity records and,
// with PC sampling enabled, where warps were when the hardware looked. An
// exact count of what each instruction executed comes from a different CUPTI
// API, which patches the kernel and runs on the calling thread's context, so
// it can only be driven by the process that owns that context. This is how
// the result reaches the capture, where it becomes a sample type next to the
// sampled one, over the same instructions.
//
// image is the module the counts were measured against, as it was loaded.
// Publishing the same name twice replaces it. Without a capture it writes
// nothing and returns false, so the call is safe to leave in a program that
// is usually run without one.
func WriteCounts(name string, image []byte, metric string, counts []Count) (bool, error) {
	if len(image) == 0 {
		return false, errors.New("counts name no module image")
	}
	sum := sha256.Sum256(image)
	data, err := json.Marshal(Counts{Module: hex.EncodeToString(sum[:]), Metric: metric, Counts: counts})
	if err != nil {
		return false, fmt.Errorf("publish counts %s: %w", name, err)
	}
	switch err := publish("counts", name, data); {
	case errors.Is(err, errNoCapture):
		return false, nil
	case err != nil:
		return false, fmt.Errorf("publish counts %s: %w", name, err)
	}
	return true, nil
}
