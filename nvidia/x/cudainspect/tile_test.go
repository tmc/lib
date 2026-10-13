package cudainspect

import (
	"bytes"
	"context"
	"debug/elf"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func tileContainer(payload []byte, duplicate bool) []byte {
	b := metadataELF(duplicate, 0, elf.EM_CUDA)
	// Reuse the synthetic text section as .nv.tileir; names only get shorter.
	index := bytes.Index(b, []byte(".text.kernel"))
	copy(b[index:], []byte(".nv.tileir\x00"))
	if duplicate {
		index = bytes.Index(b, []byte(".nv.shared.kernel"))
		copy(b[index:], []byte(".tilebc\x00"))
	}
	section := b[64+2*64:]
	binary.LittleEndian.PutUint64(section[24:], uint64(len(b)))
	binary.LittleEndian.PutUint64(section[32:], uint64(len(payload)))
	return append(b, payload...)
}

func TestTileBytecodeExtractionAndRoundTrip(t *testing.T) {
	golden, err := os.ReadFile("../tileir/testdata/bytecodegoldens/f32_array.golden.tileirbc")
	if err != nil {
		t.Fatal(err)
	}
	image := tileContainer(golden, false)
	r, err := ExtractTileIR(image)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(r.Bytecode, golden) || !r.RoundTripExact || r.Version == "" || r.Digest == "" || r.ContainerDigest == "" || r.Section != ".nv.tileir" {
		t.Fatalf("extraction: %+v", r)
	}
	for i := range image {
		image[i] = 0
	}
	if !bytes.Equal(r.Bytecode, golden) {
		t.Fatal("artifact aliases container")
	}
}

func TestTileExtractionRefusesMissingAmbiguousAndInvalid(t *testing.T) {
	for _, image := range [][]byte{[]byte("invalid"), metadataELF(false, 0, elf.EM_CUDA), tileContainer([]byte("bad"), true)} {
		if _, err := ExtractTileIR(image); err == nil {
			t.Fatal("accepted invalid or ambiguous container")
		}
	}
	payload := []byte("bad bytecode header")
	r, err := ExtractTileIR(tileContainer(payload, false))
	if err == nil || r == nil || r.DecodeError == "" || r.RoundTripExact || !bytes.Equal(r.Bytecode, payload) {
		t.Fatalf("lost invalid intermediate: %+v, %v", r, err)
	}
}

func ExampleExtractTileIR() {
	_, err := ExtractTileIR([]byte("not an ELF container"))
	fmt.Println(err != nil)
	// Output: true
}

func TestTileExtractionBeforeDisassembly(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX subprocess fixture")
	}
	golden, err := os.ReadFile("../tileir/testdata/bytecodegoldens/f32_array.golden.tileirbc")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "expected"), golden, 0600); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "tileirdisasm")
	body := `#!/bin/sh
test "$1" = -g || exit 9
cmp "$2" "$(dirname "$0")/expected" || exit 8
printf 'entry @fixture() { return }\n'
`
	if err := os.WriteFile(path, []byte(body), 0700); err != nil {
		t.Fatal(err)
	}
	i, _ := NewInspector(t.TempDir())
	r, err := i.InspectTileIR(context.Background(), tileContainer(golden, false), path)
	if err != nil || r.Module == nil || r.Artifact == nil || r.Output.Path != path || len(r.Output.Stdout) == 0 {
		t.Fatalf("pipeline=%+v, %v", r, err)
	}
	r, err = i.InspectTileIR(context.Background(), tileContainer([]byte("invalid header"), false), path)
	if err == nil || r.Artifact == nil || r.Output.Path != "" {
		t.Fatalf("invalid header reached tool: %+v %v", r, err)
	}
	r, err = i.InspectTileIR(context.Background(), tileContainer(golden, false), filepath.Join(dir, "missing"))
	if err == nil || r.Artifact == nil || len(r.Output.Args) == 0 {
		t.Fatalf("lost extracted bytes on tool failure: %+v %v", r, err)
	}
	entries, err := os.ReadDir(i.WorkDir)
	if err != nil || len(entries) != 0 {
		t.Fatal("scratch leaked")
	}
}

func ExampleInspector_InspectTileIR() {
	i, _ := NewInspector(".")
	_, err := i.InspectTileIR(context.Background(), []byte("invalid ELF"), "tileirdisasm")
	fmt.Println(err != nil)
	// Output: true
}
