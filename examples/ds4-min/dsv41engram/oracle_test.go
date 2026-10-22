package dsv41engram

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"
)

// TestPinnedCOracle builds the unmodified Engram implementation from the exact
// 11e8cad Git object. Set DS4_ENGRAM_ORACLE to an upstream checkout to run V1.
// The fixture is host-only; it makes no model or GPU parity claim.
func TestPinnedCOracle(t *testing.T) {
	repo := os.Getenv("DS4_ENGRAM_ORACLE")
	if repo == "" {
		t.Skip("set DS4_ENGRAM_ORACLE for the pinned C gate")
	}
	dir := t.TempDir()
	const revision = "11e8cad4473716a41baf4d62e92ed9764a00fce6"
	for _, source := range []struct{ name, hash string }{
		{"ds4_engram.c", "02785ce48599fa4f514a530697efd0f4483d405c42367332748f8facb2c106cc"},
		{"ds4_engram.h", "f84a264e0fe199d23a6f0c56fbbd19e222adc7009eed13c185af6403f68f7f5c"},
	} {
		name := source.name
		cmd := exec.Command("git", "-C", repo, "show", revision+":"+name)
		data, err := cmd.Output()
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(data)) != source.hash {
			t.Fatalf("pinned source hash mismatch: %s", name)
		}
		if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	harness, err := os.ReadFile("testdata/oracle.c")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "oracle.c"), harness, 0600); err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(dir, "oracle")
	cmd := exec.Command("cc", "-std=c11", "-O2", "-pthread", "-I", dir, filepath.Join(dir, "oracle.c"), filepath.Join(dir, "ds4_engram.c"), "-lm", "-o", bin)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("compile pinned C: %v\n%s", err, out)
	}
	s := testSpec()
	tokens := make([]int32, 10000)
	mask := make([]byte, len(tokens))
	for i := range tokens {
		tokens[i] = int32((i*i + 31*i + 7) % len(s.TokenMap))
		mask[i] = 1
	}
	for _, masked := range []bool{false, true} {
		if masked {
			for i := range mask {
				if i%17 == 0 || i%89 < 4 {
					mask[i] = 0
				}
			}
		}
		for _, split := range []int{0, 1, 2, 3, 127, 2048, 9999, 10000} {
			l, err := NewLayout(s)
			if err != nil {
				t.Fatal(err)
			}
			var h History
			got := make([]uint32, len(tokens)*Layers*Columns)
			if err := l.Hash(&h, tokens[:split], mask[:split], got[:split*Layers*Columns]); err != nil {
				t.Fatal(err)
			}
			if err := l.Hash(&h, tokens[split:], mask[split:], got[split*Layers*Columns:]); err != nil {
				t.Fatal(err)
			}
			ok, tail, want := runC(t, bin, s, tokens, mask, split)
			if !ok || tail != h.tail || !reflect.DeepEqual(got, want) {
				t.Fatalf("masked=%v split=%d differs from pinned C", masked, split)
			}
		}
	}
	// Independent wrong-history, ignored-mask and wrong-multiplier controls must
	// differ from the oracle, rather than merely asserting that Go ran.
	_, _, want := runC(t, bin, s, tokens, mask, 10000)
	for _, kind := range []string{"history", "mask", "hash"} {
		l, _ := NewLayout(s)
		var h History
		got := make([]uint32, len(want))
		m := mask
		if kind == "mask" {
			m = bytes.Repeat([]byte{1}, len(mask))
		}
		if kind == "hash" {
			l.spec.Multipliers[0][0] += 2
		}
		for i := range tokens {
			if kind == "history" && h.initialized {
				h.tail[0], h.tail[2] = h.tail[2], h.tail[0]
			}
			if err := l.Hash(&h, tokens[i:i+1], m[i:i+1], got[i*Layers*Columns:(i+1)*Layers*Columns]); err != nil {
				t.Fatal(err)
			}
		}
		if reflect.DeepEqual(got, want) {
			t.Fatalf("%s mutation was invisible", kind)
		}
	}
	bad := append([]int32(nil), tokens...)
	bad[9999] = -1
	if ok, tail, _ := runC(t, bin, s, bad, mask, 10000); ok || tail != [3]int32{-1, -1, -1} {
		t.Fatal("C invalid token admission changed history")
	}
	for _, mutate := range []func(*LayoutSpec){func(s *LayoutSpec) { s.PadID = 101 }, func(s *LayoutSpec) { s.TokenMap[0] = 101 }, func(s *LayoutSpec) { s.Multipliers[1][2] = 2 }, func(s *LayoutSpec) { s.Rows[1]++ }} {
		invalid := testSpec()
		mutate(&invalid)
		_, err := NewLayout(invalid)
		ok, _, _ := runC(t, bin, invalid, tokens, mask, 10000)
		if err == nil || ok {
			t.Fatal("invalid layout admitted")
		}
	}
	// Exhaust the finite E4M3 byte space and E8M0 scales 0..144, comparing
	// decoded BF16 bits directly, including signed zero and subnormals.
	raw := make([]byte, 256*RowBytes)
	for r := range 256 {
		for j := range Dim {
			code := byte((r + j) % 256)
			if code&127 == 127 {
				code--
			}
			raw[r*RowBytes+j] = code
		}
		for j := range 8 {
			raw[r*RowBytes+Dim+j] = byte((r + j) % 145)
		}
	}
	path := filepath.Join(dir, "rows")
	if err := os.WriteFile(path, raw, 0600); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(bin, path).Output()
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 4+256*Dim*4 || binary.LittleEndian.Uint32(out) != 1 {
		t.Fatal("C row decode failed")
	}
	for r := range 256 {
		got := make([]float32, Dim)
		if err := DecodeRow(raw[r*RowBytes:(r+1)*RowBytes], got); err != nil {
			t.Fatal(err)
		}
		for j, v := range got {
			want := binary.LittleEndian.Uint32(out[4+(r*Dim+j)*4:])
			if math.Float32bits(v) != want {
				t.Fatalf("row %d col %d bits %08x != %08x", r, j, math.Float32bits(v), want)
			}
		}
	}
	t.Log("V1: 16 corpora x 10000 tokens x 48 row IDs; 3 test-input controls; 5 invalid admissions; 65536 decoded float bits")
}

// TestGoSourceMutations runs the ordinary pinned-C comparison with one actual
// production source change per Go overlay. Compilation errors or unrelated
// failures do not count as a detected numerical mutation.
func TestGoSourceMutations(t *testing.T) {
	if os.Getenv("DS4_ENGRAM_ORACLE") == "" {
		t.Skip("set DS4_ENGRAM_ORACLE for the pinned C mutation gate")
	}
	dir := t.TempDir()
	for _, m := range []struct{ name, file, old, replacement, failure string }{
		{"dead-propagation", "hash.go", "blocked = blocked || id == -1", "blocked = id == -1", "differs from pinned C"},
		{"xor-update", "hash.go", "hash ^= uint64(ids[j]) * l.spec.Multipliers[layer][j]", "hash += uint64(ids[j]) * l.spec.Multipliers[layer][j]", "differs from pinned C"},
		{"bf16-rounding", "table.go", "bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000", "bits = bits", "bits "},
	} {
		t.Run(m.name, func(t *testing.T) {
			path, err := filepath.Abs(m.file)
			if err != nil {
				t.Fatal(err)
			}
			source, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if bytes.Count(source, []byte(m.old)) != 1 {
				t.Fatal("mutation source site must occur exactly once")
			}
			mutant := filepath.Join(dir, m.name+".go")
			if err := os.WriteFile(mutant, bytes.Replace(source, []byte(m.old), []byte(m.replacement), 1), 0600); err != nil {
				t.Fatal(err)
			}
			overlay := filepath.Join(dir, m.name+".json")
			data, err := json.Marshal(struct{ Replace map[string]string }{map[string]string{path: mutant}})
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(overlay, data, 0600); err != nil {
				t.Fatal(err)
			}
			cmd := exec.Command("go", "test", "-overlay="+overlay, "-count=1", "-run=^TestPinnedCOracle$", "-v", ".")
			out, err := cmd.CombinedOutput()
			var exit *exec.ExitError
			if !errors.As(err, &exit) || exit.ExitCode() != 1 || !bytes.Contains(out, []byte("--- FAIL: TestPinnedCOracle")) || !bytes.Contains(out, []byte(m.failure)) {
				t.Fatalf("mutation was not rejected by ordinary C comparison: %v\n%s", err, out)
			}
			t.Logf("actual source overlay rejected: %s\n%s", m.name, out)
		})
	}
}

func runC(t *testing.T, bin string, s LayoutSpec, tokens []int32, mask []byte, split int) (bool, [3]int32, []uint32) {
	t.Helper()
	var in bytes.Buffer
	for _, v := range []any{uint32(len(s.TokenMap)), s.CompressedVocabSize, s.PadID, s.Rows, s.Multipliers, s.Primes, s.TokenMap, uint32(len(tokens)), uint32(split), tokens} {
		if err := binary.Write(&in, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	in.Write(mask)
	cmd := exec.Command(bin)
	cmd.Stdin = &in
	out, err := cmd.Output()
	if err != nil {
		t.Fatal(err)
	}
	r := bytes.NewReader(out)
	var status int32
	var tail [3]int32
	if err := binary.Read(r, binary.LittleEndian, &status); err != nil {
		t.Fatal(err)
	}
	if err := binary.Read(r, binary.LittleEndian, &tail); err != nil {
		t.Fatal(err)
	}
	if status == 0 {
		return false, tail, nil
	}
	rows := make([]uint32, len(tokens)*Layers*Columns)
	if err := binary.Read(r, binary.LittleEndian, rows); err != nil {
		t.Fatal(err)
	}
	if r.Len() != 0 {
		t.Fatal("unexpected C output")
	}
	return true, tail, rows
}
