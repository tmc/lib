package main

import (
	"os"
	"runtime"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasafetensors"
)

// The retention memory measurement needs a real checkpoint, so it is opt-in.
//
// LLAMA_RETAIN_CHECKPOINT names the directory. LLAMA_RETAIN_MODE selects what
// the process loads: "retained" keeps the source dtype, "float32" widens the
// way the existing loader does. Each mode is measured in its own process,
// because a process that has held both representations cannot report the
// retained cost of either.
const (
	checkpointEnv = "LLAMA_RETAIN_CHECKPOINT"
	modeEnv       = "LLAMA_RETAIN_MODE"
)

// TestRetainedCheckpointHostMemory reports the host memory one loading mode
// costs on a real checkpoint.
//
// It measures the same loading scope in both modes: every tensor the
// checkpoint declares, read through the same reader, retained in a live map
// until after the measurement. The difference between the two runs is the
// representation, not the set of tensors.
func TestRetainedCheckpointHostMemory(t *testing.T) {
	dir := os.Getenv(checkpointEnv)
	if dir == "" {
		t.Skipf("set %s to a checkpoint directory to measure retention", checkpointEnv)
	}
	mode := os.Getenv(modeEnv)
	if mode != "retained" && mode != "float32" {
		t.Fatalf("set %s to \"retained\" or \"float32\", got %q", modeEnv, mode)
	}

	sw, err := cudasafetensors.OpenSharded(dir)
	if err != nil {
		t.Fatalf("open checkpoint: %v", err)
	}
	defer sw.Close()

	names := make([]string, 0, len(sw.Tensors))
	for name := range sw.Tensors {
		names = append(names, name)
	}

	// Baseline after the reader is open but before any payload is held, so the
	// reported cost is the weights and not the index.
	var before runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)

	var (
		elems     int
		payload   int
		retained  *RetainedCheckpoint
		widened   map[string][]float32
		dtypeSeen = map[string]int{}
	)
	switch mode {
	case "retained":
		retained, err = loadRetainedTensors(sw, names)
		if err != nil {
			t.Fatalf("load retained: %v", err)
		}
		for _, tensor := range retained.Tensors {
			elems += tensor.Elems()
			payload += tensor.Bytes()
			dtypeSeen[tensor.DType]++
		}
	case "float32":
		widened = make(map[string][]float32, len(names))
		for _, name := range names {
			raw, info, err := sw.TensorBytes(name)
			if err != nil {
				t.Fatalf("read %s: %v", name, err)
			}
			vals := decodeTensorToFloat32(raw, info.DType)
			widened[name] = vals
			elems += len(vals)
			payload += len(vals) * 4
			dtypeSeen[info.DType]++
		}
	}

	var after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&after)

	// Keep the payload alive across the measurement.
	runtime.KeepAlive(retained)
	runtime.KeepAlive(widened)

	t.Logf("checkpoint=%s mode=%s tensors=%d elements=%d", dir, mode, len(names), elems)
	t.Logf("dtypes=%v", dtypeSeen)
	t.Logf("declared payload bytes=%d (%.2f bytes/element)", payload, float64(payload)/float64(elems))
	t.Logf("heap in use after load=%d bytes (delta %d)", after.HeapAlloc, int64(after.HeapAlloc)-int64(before.HeapAlloc))
	t.Logf("cumulative heap allocated=%d bytes", after.TotalAlloc-before.TotalAlloc)
	t.Logf("peak heap from OS=%d bytes", after.HeapSys)

	if elems == 0 {
		t.Fatal("checkpoint declared no elements")
	}
	if mode == "retained" {
		// The accounting claim: half tensors are retained at two bytes per
		// element, with no widened copy alongside them.
		for name, tensor := range retained.Tensors {
			size, err := dtypeElemSize(tensor.DType)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			if got := tensor.Bytes(); got != tensor.Elems()*size {
				t.Fatalf("%s retains %d bytes for %d %s elements", name, got, tensor.Elems(), tensor.DType)
			}
		}
		if got := float64(payload) / float64(elems); got > 4 {
			t.Fatalf("retained %.2f bytes per element, which is not narrower than float32", got)
		}
	}
}
