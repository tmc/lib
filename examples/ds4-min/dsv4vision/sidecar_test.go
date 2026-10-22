package dsv4vision

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/tmc/lib/gguf"
)

type sidecarFixture struct {
	metadata map[string]gguf.Value
	tensors  []gguf.Tensor
}

func newSidecarFixture() sidecarFixture {
	f := sidecarFixture{metadata: map[string]gguf.Value{
		"general.architecture":                              {TypeID: gguf.TypeString, Value: "deepseek4-vision"},
		"general.source.revision":                           {TypeID: gguf.TypeString, Value: EncoderRevision},
		"deepseek4-vision.checkpoint_variant":               {TypeID: gguf.TypeString, Value: "vision-exp"},
		"deepseek4-vision.attention.layer_norm_rms_epsilon": {TypeID: gguf.TypeFloat32, Value: float32(1e-6)},
	}}
	for _, d := range sidecarDimensions {
		f.metadata["deepseek4-vision."+d.key] = gguf.Value{TypeID: gguf.TypeUint32, Value: uint32(d.value)}
	}
	off := uint64(0)
	for _, c := range sidecarContracts() {
		t := gguf.Tensor{Name: c.name, Dimensions: append([]uint64(nil), c.dims...), TypeID: c.typeID, Offset: off}
		f.tensors = append(f.tensors, t)
		n, _ := t.Bytes()
		off = (off + n + 31) &^ 31
	}
	return f
}

// fixtureReader avoids allocating a nearly-gigabyte tensor payload. Tensor
// data is deterministic zero-filled and header reads use real GGUF bytes.
type fixtureReader struct {
	header []byte
	size   int64
}

func (r fixtureReader) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 {
		return 0, fmt.Errorf("negative read")
	}
	if off >= r.size {
		return 0, io.EOF
	}
	n := len(p)
	if int64(n) > r.size-off {
		n = int(r.size - off)
	}
	clear(p[:n])
	if off < int64(len(r.header)) {
		copy(p[:n], r.header[off:])
	}
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

func (f sidecarFixture) reader() fixtureReader {
	var b bytes.Buffer
	write := func(v any) {
		if err := binary.Write(&b, binary.LittleEndian, v); err != nil {
			panic(err)
		}
	}
	str := func(s string) { write(uint64(len(s))); b.WriteString(s) }
	b.WriteString("GGUF")
	write(uint32(3))
	write(uint64(len(f.tensors)))
	write(uint64(len(f.metadata)))
	var keys []string
	for k := range f.metadata {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		v := f.metadata[k]
		str(k)
		write(v.TypeID)
		if v.TypeID == gguf.TypeString {
			str(v.Value.(string))
		} else {
			write(v.Value)
		}
	}
	maxEnd := uint64(0)
	for _, t := range f.tensors {
		str(t.Name)
		write(uint32(len(t.Dimensions)))
		for _, d := range t.Dimensions {
			write(d)
		}
		write(t.TypeID)
		write(t.Offset)
		n, err := t.Bytes()
		if err == nil && t.Offset+n > maxEnd {
			maxEnd = t.Offset + n
		}
	}
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	return fixtureReader{b.Bytes(), int64(b.Len()) + int64(maxEnd)}
}

func TestSidecar(t *testing.T) {
	f := newSidecarFixture()
	r := f.reader()
	s, err := ReadSidecar(r, r.size)
	if err != nil {
		t.Fatal(err)
	}
	if len(s.tensors) != 316 {
		t.Fatal(len(s.tensors))
	}
	d, err := s.Tensor("aligner.w1.weight")
	if err != nil {
		t.Fatal(err)
	}
	if d.TypeID != 30 || d.Bytes != 9216*4096*2 || d.Dimensions[0] != 9216 {
		t.Fatal(d)
	}
	d.Dimensions[0] = 1
	again, _ := s.Tensor(d.Name)
	if again.Dimensions[0] != 9216 {
		t.Fatal("caller changed bound dimensions")
	}
	reader, err := s.Reader("image_start")
	if err != nil {
		t.Fatal(err)
	}
	if reader.Size() != 4096*2 {
		t.Fatal(reader.Size())
	}
	_, offset, extent := reader.Outer()
	expect, _ := s.Tensor("image_start")
	if offset != int64(expect.Offset) || extent != int64(expect.Bytes) {
		t.Fatalf("reader extent %d/%d, want %d/%d", offset, extent, expect.Offset, expect.Bytes)
	}
	got, err := io.ReadAll(reader)
	if err != nil || len(got) != 4096*2 {
		t.Fatalf("read: %d %v", len(got), err)
	}
	if _, err := reader.ReadAt(make([]byte, 1), reader.Size()); err != io.EOF {
		t.Fatalf("read crossed tensor: %v", err)
	}
	if _, err := s.Tensor("not-a-tensor"); err == nil {
		t.Fatal("unknown tensor admitted")
	}
	if _, err := (*Sidecar)(nil).Reader("image_start"); err == nil {
		t.Fatal("nil sidecar admitted")
	}
}

func TestSidecarCodeMutants(t *testing.T) {
	if os.Getenv("DS4_VISION_SIDECAR_MUTATIONS") != "1" {
		t.Skip("set DS4_VISION_SIDECAR_MUTATIONS=1")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("mutants require gpu-lock")
	}
	for _, m := range []struct{ name, file, old, new, want string }{
		{"revision", "sidecar.go", "s != want.value", "(s != want.value && false)", "--- FAIL: TestSidecarRejects/revision"},
		{"tensor-type", "sidecar.go", "t.TypeID != c.typeID || ", "", "--- FAIL: TestSidecarRejects/tensor-type"},
		{"canonical-offset", "../../../gguf/gguf.go", "if t.Offset != expectedOffset {", "if false {", "--- FAIL: TestSidecarRejects/overlap"},
		{"range-end", "../../../gguf/gguf.go", "!ok || end > g.Size-g.DataOffset", "!ok", "--- FAIL: TestSidecarRejects/truncated-payload"},
		{"reader-offset", "sidecar.go", "int64(t.Offset), int64(t.Bytes)", "0, int64(t.Bytes)", "--- FAIL: TestSidecar"},
		{"reader-bound", "sidecar.go", "int64(t.Offset), int64(t.Bytes)", "int64(t.Offset), int64(t.Bytes)+1", "--- FAIL: TestSidecar"},
	} {
		t.Run(m.name, func(t *testing.T) {
			abs, err := filepath.Abs(m.file)
			if err != nil {
				t.Fatal(err)
			}
			b, err := os.ReadFile(abs)
			if err != nil {
				t.Fatal(err)
			}
			if strings.Count(string(b), m.old) != 1 {
				t.Fatalf("site count !=1: %s", m.old)
			}
			dir := t.TempDir()
			mut := filepath.Join(dir, filepath.Base(m.file))
			if err := os.WriteFile(mut, []byte(strings.Replace(string(b), m.old, m.new, 1)), 0600); err != nil {
				t.Fatal(err)
			}
			overlay, _ := json.Marshal(map[string]any{"Replace": map[string]string{abs: mut}})
			path := filepath.Join(dir, "overlay.json")
			if err := os.WriteFile(path, overlay, 0600); err != nil {
				t.Fatal(err)
			}
			cmd := exec.Command("go", "test", "-mod=readonly", "-overlay", path, "-count=1", "-run", "^(TestSidecar|TestSidecarRejects)$", ".")
			out, err := cmd.CombinedOutput()
			if err == nil || !bytes.Contains(out, []byte(m.want)) {
				t.Fatalf("mutant not rejected at expected ordinary gate: %v %s", err, out)
			}
			t.Logf("actual %s mutant rejected:\n%s", m.name, out)
		})
	}
}

func TestSidecarRejects(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*sidecarFixture)
	}{
		{"revision", func(f *sidecarFixture) {
			f.metadata["general.source.revision"] = gguf.Value{TypeID: gguf.TypeString, Value: "wrong"}
		}},
		{"v41", func(f *sidecarFixture) {
			f.metadata["deepseek4-vision.checkpoint_variant"] = gguf.Value{TypeID: gguf.TypeString, Value: "v4.1-flash"}
		}},
		{"architecture", func(f *sidecarFixture) {
			f.metadata["general.architecture"] = gguf.Value{TypeID: gguf.TypeString, Value: "glm53"}
		}},
		{"missing-metadata", func(f *sidecarFixture) { delete(f.metadata, "deepseek4-vision.patch_size") }},
		{"dimension-metadata", func(f *sidecarFixture) {
			f.metadata["deepseek4-vision.projection_length"] = gguf.Value{TypeID: gguf.TypeUint32, Value: uint32(5120)}
		}},
		{"metadata-integer-type", func(f *sidecarFixture) {
			f.metadata["deepseek4-vision.patch_size"] = gguf.Value{TypeID: gguf.TypeUint64, Value: uint64(14)}
		}},
		{"epsilon", func(f *sidecarFixture) {
			f.metadata["deepseek4-vision.attention.layer_norm_rms_epsilon"] = gguf.Value{TypeID: gguf.TypeFloat32, Value: float32(1e-5)}
		}},
		{"tensor-count", func(f *sidecarFixture) { f.tensors = f.tensors[:315] }},
		{"tensor-type", func(f *sidecarFixture) { f.tensors[0].TypeID = 1 }},
		{"tensor-dimension", func(f *sidecarFixture) { f.tensors[0].Dimensions[0] = 589 }},
		{"missing-tensor", func(f *sidecarFixture) { f.tensors[0].Name = "other" }},
		{"duplicate", func(f *sidecarFixture) { f.tensors[1].Name = f.tensors[0].Name }},
		{"overlap", func(f *sidecarFixture) { f.tensors[1].Offset = f.tensors[0].Offset }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newSidecarFixture()
			tc.change(&f)
			r := f.reader()
			if _, err := ReadSidecar(r, r.size); err == nil {
				t.Fatal("malformed sidecar admitted")
			}
		})
	}
	t.Run("truncated-payload", func(t *testing.T) {
		r := newSidecarFixture().reader()
		if _, err := ReadSidecar(r, r.size-1); err == nil {
			t.Fatal("truncated payload admitted")
		}
	})
	t.Run("truncated-header", func(t *testing.T) {
		r := newSidecarFixture().reader()
		if _, err := ReadSidecar(bytes.NewReader(r.header[:31]), r.size); err == nil {
			t.Fatal("truncated header admitted")
		}
	})
	if _, err := ReadSidecar(nil, 0); err == nil {
		t.Fatal("nil reader admitted")
	}
}

func ExampleReadSidecar() {
	// A production reader is an immutable open encoder GGUF file.
	r := newSidecarFixture().reader()
	s, err := ReadSidecar(r, r.size)
	tensor, _ := s.Tensor("image_start")
	fmt.Println(tensor.Dimensions, tensor.Bytes, err)
	// Output: [4096] 8192 <nil>
}

func ExampleSidecar_Tensor() {
	r := newSidecarFixture().reader()
	s, _ := ReadSidecar(r, r.size)
	t, _ := s.Tensor("vision.patch_embed.proj.weight")
	fmt.Println(t.Dimensions, t.TypeID)
	// Output: [588 1024] 30
}

func ExampleSidecar_Reader() {
	r := newSidecarFixture().reader()
	s, _ := ReadSidecar(r, r.size)
	tensor, _ := s.Reader("image_start")
	fmt.Println(tensor.Size())
	// Output: 8192
}

func TestRealEncoderHeader(t *testing.T) {
	path := os.Getenv("DS4_VISION_ENCODER_HEADER")
	if path == "" {
		t.Skip("set DS4_VISION_ENCODER_HEADER to pinned range artifact")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("header gate requires gpu-lock")
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
	if info.Size() != 1048576 {
		t.Fatalf("pinned encoder header size %d, want 1048576", info.Size())
	}
	b, err := io.ReadAll(io.LimitReader(f, 1048577))
	if err != nil {
		t.Fatal(err)
	}
	if len(b) != 1048576 || fmt.Sprintf("%x", sha256.Sum256(b)) != "2ab4057763b73d50d79fa773c74039cb18c814f2cc11895770fee830d4222483" {
		t.Fatal("pinned encoder header content mismatch")
	}
	s, err := ReadSidecar(bytes.NewReader(b), 932857760)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("actual encoder header binds %d tensors; payload content is not read or hash-verified", len(s.tensors))
}
