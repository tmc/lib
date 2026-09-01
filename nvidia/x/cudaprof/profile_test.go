package cudaprof_test

import (
	"bytes"
	"compress/gzip"
	"context"
	"debug/elf"
	"io"
	"runtime/pprof"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

func TestProfileSession(t *testing.T) {
	var buf bytes.Buffer
	session := cudaprof.NewSession(&buf)

	ctx := pprof.WithLabels(context.Background(), pprof.Labels(
		"model_layer", "transformer_block_0",
		"op", "gemm_forward",
	))

	session.RecordSample(ctx, "GemmKernel", "gemm.cu", 42, 0x7fff00001040, 100)
	session.RecordSample(ctx, "SoftmaxKernel", "softmax.cu", 18, 0x7fff00002080, 50)

	if err := session.Close(); err != nil {
		t.Fatalf("session.Close failed: %v", err)
	}

	if buf.Len() == 0 {
		t.Fatal("expected non-empty profile buffer")
	}

	// Decompress and verify
	gr, err := gzip.NewReader(&buf)
	if err != nil {
		t.Fatalf("gzip reader failed: %v", err)
	}
	defer gr.Close()

	decompressed, err := io.ReadAll(gr)
	if err != nil {
		t.Fatalf("decompression failed: %v", err)
	}

	if len(decompressed) == 0 {
		t.Fatal("expected non-empty decompressed profile payload")
	}
}

func TestSynthesizeELFBinary(t *testing.T) {
	syms := []cudaprof.Symbol{
		{Name: "KernelZ", Address: 0x7f003000, Size: 0x40},
		{Name: "KernelA", Address: 0x7f001000, Size: 0x20},
		{Name: "KernelM", Address: 0x7f002000, Size: 0x30},
	}

	data, err := cudaprof.SynthesizeELFBinary(syms)
	if err != nil {
		t.Fatalf("SynthesizeELFBinary failed: %v", err)
	}

	f, err := elf.NewFile(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("failed to parse synthesized ELF: %v", err)
	}
	defer f.Close()

	if f.FileHeader.Class != elf.ELFCLASS64 {
		t.Errorf("expected 64-bit ELF, got %v", f.FileHeader.Class)
	}

	symbols, err := f.Symbols()
	if err != nil {
		t.Fatalf("failed to read symbols: %v", err)
	}

	found := make(map[string]uint64)
	for _, sym := range symbols {
		if sym.Name != "" {
			found[sym.Name] = sym.Value
		}
	}

	for _, want := range syms {
		if addr, ok := found[want.Name]; !ok {
			t.Errorf("missing symbol %s", want.Name)
		} else if addr != want.Address {
			t.Errorf("symbol %s: expected address 0x%x, got 0x%x", want.Name, want.Address, addr)
		}
	}
}

func TestRecordTraceback(t *testing.T) {
	var buf bytes.Buffer
	session := cudaprof.NewSession(&buf)

	vals := cudaprof.SampleValues{
		Samples:       1000,
		DRAMReadBytes: 4096,
	}

	session.RecordTraceback(
		context.Background(),
		vals,
		"LinearGEMV",
		"forward.cu",
		90,
		"forward.ptx",
		26,
		0x7f001040,
	)

	if err := session.Close(); err != nil {
		t.Fatalf("session.Close failed: %v", err)
	}

	if buf.Len() == 0 {
		t.Fatal("expected non-empty profile buffer")
	}
}

