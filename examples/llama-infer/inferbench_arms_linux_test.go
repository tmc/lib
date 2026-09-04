//go:build linux

package main

import "testing"

func TestCUDABenchmarkArmsChangeOnlyBatchWidth(t *testing.T) {
	eng := new(ModelEngine)
	if err := configureBenchmarkArm(eng, "llama-infer"); err != nil {
		t.Fatal(err)
	}
	packed := cudaBatchWidths(4, eng.benchmarkSerial)
	if err := configureBenchmarkArm(eng, "llama-infer-serial"); err != nil {
		t.Fatal(err)
	}
	serial := cudaBatchWidths(4, eng.benchmarkSerial)

	if len(packed) != 1 || packed[0] != 4 {
		t.Fatalf("packed widths = %v", packed)
	}
	if len(serial) != 4 {
		t.Fatalf("serial widths = %v", serial)
	}
	var packedWork, serialWork int
	for _, width := range packed {
		packedWork += width
	}
	for _, width := range serial {
		if width != 1 {
			t.Fatalf("serial width = %d, want 1", width)
		}
		serialWork += width
	}
	if packedWork != serialWork {
		t.Fatalf("work differs: packed %d, serial %d", packedWork, serialWork)
	}
}

func TestCUDABenchmarkArmRestoresPackedScheduling(t *testing.T) {
	eng := new(ModelEngine)
	if err := configureBenchmarkArm(eng, "llama-infer-serial"); err != nil {
		t.Fatal(err)
	}
	if err := configureBenchmarkArm(eng, "llama-infer"); err != nil {
		t.Fatal(err)
	}
	if eng.benchmarkSerial {
		t.Fatal("ordinary arm retained serial scheduling")
	}
}
