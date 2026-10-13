package cudainspect

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"encoding/hex"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
)

func registerELF(attributes []byte) []byte {
	b := metadataELF(true, 0, elf.EM_CUDA)
	copy(b[bytes.Index(b, []byte(".nv.shared.kernel")):], []byte(".nv.info.kernel\x00"))
	section := b[64+3*64:]
	binary.LittleEndian.PutUint32(section[4:], uint32(elf.SHT_PROGBITS))
	binary.LittleEndian.PutUint64(section[24:], uint64(len(b)))
	binary.LittleEndian.PutUint64(section[32:], uint64(len(attributes)))
	return append(b, attributes...)
}

func TestAllocatedRegistersExcludeLimits(t *testing.T) {
	attribute := func(tag byte, value uint32) []byte {
		b := []byte{4, tag, 4, 0, 0, 0, 0, 0}
		binary.LittleEndian.PutUint32(b[4:], value)
		return b
	}
	limit := attribute(cudaoccupancy.AttrMaxRegCount, 64)
	allocation := attribute(cudaoccupancy.AttrRegCount, 24)
	for _, tt := range []struct {
		name  string
		data  []byte
		known bool
		value uint32
	}{
		{"limit_only", limit, false, 0},
		{"allocation_only", allocation, true, 24},
		{"distinct_limit", append(append([]byte(nil), limit...), allocation...), true, 24},
		{"absent", nil, false, 0},
		{"ambiguous_zero", attribute(cudaoccupancy.AttrRegCount, 0), false, 0},
		{"unqualified_attribute", attribute(2, 64), false, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			i, _ := NewInspector(".")
			r, err := i.Inspect(registerELF(tt.data), "kernel")
			if err != nil {
				t.Fatal(err)
			}
			if r.RegCountKnown != tt.known || r.RegCount != tt.value || r.PeakGPRKnown || r.PeakGPR != 0 {
				t.Fatalf("register distinctions lost: %+v", r)
			}
		})
	}
}

func TestReportRecordedAllocatedRegisters(t *testing.T) {
	raw, err := os.ReadFile("../cudaoccupancy/testdata/kernels.cubin.hex")
	if err != nil {
		t.Fatal(err)
	}
	image, err := hex.DecodeString(strings.Join(strings.Fields(string(raw)), ""))
	if err != nil {
		t.Fatal(err)
	}
	i, _ := NewInspector(".")
	r, err := i.Inspect(image, "compute")
	if err != nil {
		t.Fatal(err)
	}
	if !r.RegCountKnown || r.RegCount != 8 || r.PeakGPRKnown {
		t.Fatalf("recorded allocation: %+v", r)
	}
}
