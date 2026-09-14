package cudasass_test

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

const (
	tkinfoFixture = "testdata/tkinfo-ptxas-13.0-sm121.cubin.hex"
	// cubinFixture is a whole cubin owned by a sibling package. Reading it
	// here exercises the section walk on an image with other sections and
	// other notes in it, which the small fixture cannot.
	cubinFixture = "../cudaoccupancy/testdata/kernels.cubin.hex"
	mixedFixture = "testdata/tkinfo-mixed-13.0-13.1.cubin.hex"

	wantTool    = "ptxas"
	wantVersion = "Cuda compilation tools, release 13.0, V13.0.88"
)

func readHexFixture(t *testing.T, path string) []byte {
	t.Helper()
	text, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	b, err := hex.DecodeString(strings.Join(strings.Fields(string(text)), ""))
	if err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
	return b
}

// TestTKInfoFieldOrder checks the descriptor field order this package assumes
// against nvdisasm's own rendering of a .note.nv.tkinfo section. NVIDIA's tool
// authored that key order, so agreement is evidence and not self-agreement.
func TestTKInfoFieldOrder(t *testing.T) {
	raw, err := os.ReadFile("testdata/nvdisasm-13.0-sm121.json")
	if err != nil {
		t.Fatalf("read nvdisasm fixture: %v", err)
	}
	var top []json.RawMessage
	if err := json.Unmarshal(raw, &top); err != nil || len(top) == 0 {
		t.Fatalf("decode nvdisasm fixture: %v", err)
	}
	var meta map[string]json.RawMessage
	if err := json.Unmarshal(top[0], &meta); err != nil {
		t.Fatalf("decode nvdisasm metadata: %v", err)
	}
	note, ok := meta[".note.nv.tkinfo"]
	if !ok {
		t.Fatal("nvdisasm fixture has no .note.nv.tkinfo object")
	}

	// Recover the key order, which json.Unmarshal into a map discards.
	dec := json.NewDecoder(bytes.NewReader(note))
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		t.Fatalf("tkinfo object: token %v, err %v", tok, err)
	}
	var keys []string
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			t.Fatalf("tkinfo key: %v", err)
		}
		keys = append(keys, tok.(string))
		var discard json.RawMessage
		if err := dec.Decode(&discard); err != nil {
			t.Fatalf("tkinfo value for %q: %v", tok, err)
		}
	}

	want := []string{"nv_note_tkinfo", "tki_objFname", "tki_toolName", "tki_toolVersion", "tki_toolBranch", "tki_toolOptions"}
	if len(keys) != len(want) {
		t.Fatalf("nvdisasm renders %d tkinfo fields %v, this package reads a %d-field descriptor", len(keys), keys, len(want))
	}
	for i := range want {
		if keys[i] != want[i] {
			t.Errorf("tkinfo field %d is %q, this package assumes %q", i, keys[i], want[i])
		}
	}

	// The same fixture is the oracle for tki_toolBranch being untrustworthy:
	// here it holds a real branch, while the cubin fixture holds a
	// placeholder, so the field means different things in different images.
	var fields map[string]string
	if err := json.Unmarshal(note, &fields); err == nil {
		if got := fields["tki_toolBranch"]; got == "" || strings.Contains(got, "must define") {
			t.Errorf("nvdisasm fixture tki_toolBranch = %q, expected a real branch string in this image", got)
		}
	}
}

func TestReadToolchains(t *testing.T) {
	small := readHexFixture(t, tkinfoFixture)
	if small == nil {
		t.Fatalf("missing %s", tkinfoFixture)
	}
	// The expected strings must be present verbatim in the fixture bytes, so
	// the assertions below cannot pass on a descriptor this package invented.
	for _, s := range []string{wantTool, wantVersion} {
		if !bytes.Contains(small, []byte(s)) {
			t.Fatalf("fixture does not contain %q", s)
		}
	}

	tests := []struct {
		name  string
		path  string
		image []byte
		want  []cudasass.Toolchain
	}{
		{
			name:  "note only",
			path:  tkinfoFixture,
			image: small,
			want:  []cudasass.Toolchain{{Tool: wantTool, Version: wantVersion, Count: 1}},
		},
		{
			name:  "whole cubin",
			path:  cubinFixture,
			image: readHexFixture(t, cubinFixture),
			want:  []cudasass.Toolchain{{Tool: wantTool, Version: wantVersion, Count: 1}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.image == nil {
				t.Skipf("fixture %s is not present; this case did not run", tt.path)
			}
			got, err := cudasass.ReadToolchains(bytes.NewReader(tt.image))
			if err != nil {
				t.Fatalf("ReadToolchains: %v", err)
			}
			if len(got) != len(tt.want) {
				t.Fatalf("ReadToolchains = %v, want %v", got, tt.want)
			}
			for i := range tt.want {
				if got[i] != tt.want[i] {
					t.Errorf("stamp %d = %+v, want %+v", i, got[i], tt.want[i])
				}
			}
		})
	}
}

// TestReadToolchainsMixed checks that a second, different stamp is reported
// rather than collapsed into the first, and that identical stamps are counted.
func TestReadToolchainsMixed(t *testing.T) {
	mixed := readHexFixture(t, mixedFixture)
	if mixed == nil {
		t.Fatalf("missing %s", mixedFixture)
	}
	got, err := cudasass.ReadToolchains(bytes.NewReader(mixed))
	if err != nil {
		t.Fatalf("ReadToolchains: %v", err)
	}
	want := []cudasass.Toolchain{
		{Tool: wantTool, Version: wantVersion, Count: 1},
		{Tool: wantTool, Version: "Cuda compilation tools, release 13.1, V13.1.55", Count: 1},
	}
	if len(got) != len(want) {
		t.Fatalf("ReadToolchains = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("stamp %d = %+v, want %+v", i, got[i], want[i])
		}
	}

	// Editing the second stamp back to the first, a same-length replacement
	// that leaves every descriptor offset valid, must yield one stamp of two.
	same := bytes.Replace(mixed, []byte("release 13.1, V13.1.55"), []byte("release 13.0, V13.0.88"), 1)
	if bytes.Equal(same, mixed) {
		t.Fatal("second version string not found in fixture")
	}
	got, err = cudasass.ReadToolchains(bytes.NewReader(same))
	if err != nil {
		t.Fatalf("ReadToolchains: %v", err)
	}
	if len(got) != 1 || got[0].Count != 2 {
		t.Fatalf("two identical stamps report %v, want one stamp with Count 2", got)
	}
}

func TestReadToolchainsRejects(t *testing.T) {
	small := readHexFixture(t, tkinfoFixture)
	if small == nil {
		t.Fatalf("missing %s", tkinfoFixture)
	}
	descOff := bytes.Index(small, []byte("NVIDIA Corp\x00"))
	if descOff < 0 {
		t.Fatal("note name not found in fixture")
	}
	descOff += len("NVIDIA Corp\x00")

	tests := []struct {
		name string
		mut  func([]byte)
		want string
	}{
		{
			name: "unknown descriptor version",
			mut:  func(b []byte) { b[descOff] = 3 },
			want: "descriptor version 3",
		},
		{
			name: "tool name offset past blob",
			mut:  func(b []byte) { b[descOff+8] = 0xff },
			want: "past",
		},
		{
			// Point tki_toolName at the empty tki_objFname string.
			name: "tool name is empty",
			mut:  func(b []byte) { b[descOff+8] = 0 },
			want: "not a tool name",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := bytes.Clone(small)
			tt.mut(b)
			_, err := cudasass.ReadToolchains(bytes.NewReader(b))
			if err == nil {
				t.Fatal("ReadToolchains accepted a malformed descriptor")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error is %v, want it to mention %q", err, tt.want)
			}
		})
	}
}

// TestReadToolchainsUnstamped checks that an ELF with no CUDA note yields no
// stamps and no error, and that a non-ELF input is reported as such.
func TestReadToolchainsUnstamped(t *testing.T) {
	self, err := os.ReadFile(os.Args[0])
	if err != nil {
		t.Skipf("cannot read the test binary: %v", err)
	}
	got, err := cudasass.ReadToolchains(bytes.NewReader(self))
	if err != nil {
		// The test binary is Mach-O on darwin, so only the ELF hosts reach
		// the section walk. Everywhere else the ELF rejection is the check.
		if !strings.Contains(err.Error(), "cudasass: read elf") {
			t.Fatalf("ReadToolchains: %v", err)
		}
		return
	}
	if len(got) != 0 {
		t.Errorf("unstamped host binary reports %v", got)
	}
}

func TestReadToolchainsNotELF(t *testing.T) {
	_, err := cudasass.ReadToolchains(bytes.NewReader([]byte("not an elf at all")))
	if err == nil {
		t.Fatal("ReadToolchains accepted a non-ELF input")
	}
	if !strings.Contains(err.Error(), "cudasass: read elf") {
		t.Errorf("error is %v, want it wrapped with a package prefix", err)
	}
}

// TestReadToolchainsCUDA12 checks the diagnostic for a cubin whose e_version
// is the producing toolkit version, which CUDA 12 nvcc writes and [debug/elf]
// rejects. The fixture is the real image with that one header field edited; no
// CUDA 12 cubin was available on this host to use directly.
func TestReadToolchainsCUDA12(t *testing.T) {
	b := readHexFixture(t, tkinfoFixture)
	if b == nil {
		t.Fatalf("missing %s", tkinfoFixture)
	}
	binary.LittleEndian.PutUint32(b[20:24], 126)
	_, err := cudasass.ReadToolchains(bytes.NewReader(b))
	if err == nil {
		t.Fatal("ReadToolchains accepted an image debug/elf cannot read")
	}
	for _, want := range []string{"e_version is 126", "CUDA 12"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error is %v, want it to mention %q", err, want)
		}
	}
}
