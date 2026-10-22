package main

import (
	"os"
	"strings"
	"testing"
)

// This is host admission of the immutable published support artifact, not
// a forward/device oracle. An absent file cannot count as a passing gate.
func TestDSparkPublishedSupportAdmission(t *testing.T) {
	path := os.Getenv("DS4_DSPARK_SUPPORT_TEST")
	if path == "" {
		t.Skip("set DS4_DSPARK_SUPPORT_TEST to the verified published support file")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	if info.Size() != 5989114272 {
		t.Fatalf("support size = %d", info.Size())
	}
	m, err := readDSparkModel(f, info.Size(), "7e319924541db3f7a163ed7e11d7532a70d48228ab59d36cb81e1d4511885360")
	if err != nil {
		t.Fatal(err)
	}
	var bytes uint64
	types := make(map[uint32]int)
	for _, tensor := range m.tensors {
		n, err := tensor.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		bytes += n
		types[tensor.TypeID]++
	}
	for name, tensor := range m.tensors {
		if strings.HasSuffix(name, ".ffn_gate_inp.weight") {
			t.Logf("router=%s type=%d dimensions=%v", name, tensor.TypeID, tensor.Dimensions)
		}
	}
	t.Logf("sha=%s bytes=%d stages=%d block=%d rank=%d noise=%d targets=%v tensors=%d tensor_bytes=%d types=%v", m.sha256, info.Size(), m.stages, m.block, m.rank, m.noise, m.targets, len(m.tensors), bytes, types)
}
