package cudarepro

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudasanitizer"
)

func fixture(t *testing.T) *Launch {
	t.Helper()
	raw := make([]byte, 64)
	copy(raw, []byte{0x7f, 'E', 'L', 'F', 2, 1, 1})
	binary.LittleEndian.PutUint16(raw[16:], 2)
	binary.LittleEndian.PutUint16(raw[18:], 190)
	binary.LittleEndian.PutUint32(raw[20:], 1)
	binary.LittleEndian.PutUint16(raw[52:], 64)
	binary.LittleEndian.PutUint16(raw[58:], 64)
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "out", IsPointer: true}, cudakernel.Param{Name: "alias", IsPointer: true})
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, raw, "kernel", schema, "sm_121")
	if err != nil {
		t.Fatal(err)
	}
	return &Launch{Version: 1, ID: "first", Artifact: a, Grid: [3]uint32{1, 1, 1}, Block: [3]uint32{1, 1, 1}, Arguments: []Argument{{Name: "out", Buffer: true, Address: 0x1000, Size: 16}, {Name: "alias", Buffer: true, Address: 0x1004, Size: 12}}, Allocations: []Allocation{{Address: 0x1000, Data: make([]byte, 16)}}}
}
func TestCaptureValidation(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*Launch)
	}{
		{"arity", func(l *Launch) { l.Arguments = l.Arguments[:1] }},
		{"width", func(l *Launch) { l.Artifact.Schema.Params[0].ByteWidth = 4 }},
		{"image", func(l *Launch) { l.Artifact.Raw[0]++ }},
		{"unmapped", func(l *Launch) { l.Arguments[0].Address = 0x2000 }},
		{"extent", func(l *Launch) { l.Arguments[0].Size = 17 }},
		{"overflow", func(l *Launch) { l.Allocations[0].Address = ^uint64(0) - 4 }},
		{"overlap", func(l *Launch) { l.Allocations = append(l.Allocations, Allocation{0x1001, []byte{1}}) }},
		{"dimensions", func(l *Launch) { l.Block[0] = 0 }},
		{"scalar", func(l *Launch) { l.Arguments = append(l.Arguments, Argument{Name: "count"}) }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			l := fixture(t)
			tt.change(l)
			if l.Validate() == nil {
				t.Fatal("accepted invalid capture")
			}
		})
	}
}
func TestRoundTrip(t *testing.T) {
	l := fixture(t)
	var b bytes.Buffer
	if err := l.Write(&b); err != nil {
		t.Fatal(err)
	}
	data := b.Bytes()
	got, err := Read(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	if got.Arguments[1].Address-got.Allocations[0].Address != 4 {
		t.Fatal("lost alias offset")
	}
	if _, err := Read(bytes.NewReader(data), int64(len(data)-1)); err == nil {
		t.Fatal("ignored limit")
	}
}
func TestAttributeAmbiguity(t *testing.T) {
	l := fixture(t)
	f := cudasanitizer.Finding{Headline: "Invalid __global__ write of size 4 bytes", Detail: []string{"Address 0x1010 is out of bounds", "and is 0 bytes after the nearest allocation at 0x1000 of size 16 bytes"}}
	got := Attribute(f, l)
	if len(got) != 2 || !got[0].OutsideSlice || !got[0].NearestAllocation {
		t.Fatalf("alias candidates: %+v", got)
	}
	other := *l
	other.ID = "second"
	got = Attribute(f, l, &other)
	if len(got) != 4 || got[2].LaunchID != "second" {
		t.Fatalf("reused address must stay ambiguous: %+v", got)
	}
	f.Detail = []string{"Address 0x9999 is out of bounds"}
	if len(Attribute(f, l)) != 0 {
		t.Fatal("guessed nearest allocation")
	}
	f.Detail = []string{"Address 0x1003 is out of bounds"}
	got = Attribute(f, l)
	if len(got) != 2 || got[0].OutsideSlice || !got[1].OutsideSlice {
		t.Fatalf("slice ranges: %+v", got)
	}
}
func ExampleCapture() {
	_, err := Capture(nil, nil, [3]uint32{1, 1, 1}, [3]uint32{1, 1, 1}, 0, 1024)
	fmt.Println(err)
	// Output: cudarepro: nil stream
}
func ExampleRead() {
	_, err := Read(bytes.NewBufferString("{}"), 64)
	fmt.Println(err)
	// Output: cudarepro: invalid capture version or identity
}
func ExampleLaunch_Validate() {
	var launch *Launch
	fmt.Println(launch.Validate())
	// Output: cudarepro: invalid capture version or identity
}
func ExampleLaunch_Write() {
	var launch *Launch
	fmt.Println(launch.Write(&bytes.Buffer{}))
	// Output: cudarepro: invalid capture version or identity
}
func ExampleLaunch_Replay() {
	var launch *Launch
	_, err := launch.Replay(nil, nil)
	fmt.Println(err)
	// Output: cudarepro: invalid capture version or identity
}
func ExampleAttribute() {
	fmt.Println(len(Attribute(cudasanitizer.Finding{Headline: "unrecognized report"})))
	// Output: 0
}

func TestAdjacentEmptySlice(t *testing.T) {
	l := fixture(t)
	l.Allocations = append(l.Allocations, Allocation{0x1010, make([]byte, 4)})
	if got := l.allocation(0x1010, 0); got != 1 {
		t.Fatalf("adjacent base resolved to allocation %d", got)
	}
	if got := l.allocation(0x1014, 0); got != -1 {
		t.Fatalf("one-past address resolved to allocation %d", got)
	}
}
