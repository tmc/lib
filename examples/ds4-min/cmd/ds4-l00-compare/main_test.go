package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func fixture(t *testing.T) (string, packet, expected) {
	t.Helper()
	dir := t.TempDir()
	hash := strings.Repeat("a", 64)
	p := packet{Version: 1, Identity: identity{2047, 4096, hash, hash, hash, hash, "scalar-l00", 1, 0}, Stages: make(map[string]stage)}
	for name, shape := range required(p.Identity.Route) {
		data := make([]byte, shape.width*4)
		path := name + ".bin"
		if err := os.WriteFile(filepath.Join(dir, path), data, 0600); err != nil {
			t.Fatal(err)
		}
		p.Stages[name] = stage{path, fmt.Sprintf("%x", sha256.Sum256(data)), shape.kind, 1, shape.width}
	}
	path := filepath.Join(dir, "packet.json")
	want := seal(t, path, p)
	return path, p, want
}

func seal(t *testing.T, path string, p packet) expected {
	t.Helper()
	b, e := json.Marshal(p)
	if e != nil {
		t.Fatal(e)
	}
	if e = os.WriteFile(path, b, 0600); e != nil {
		t.Fatal(e)
	}
	return expected{fmt.Sprintf("%x", sha256.Sum256(b)), p.Identity}
}

func TestPacketMutations(t *testing.T) {
	for _, name := range []string{"ordinary", "missing-stage", "extra-stage", "extra-raw", "row2045", "position", "route", "history", "extent", "nonfinite", "data-sha", "missing-field", "duplicate", "trailing", "null-row"} {
		t.Run(name, func(t *testing.T) {
			path, p, want := fixture(t)
			switch name {
			case "missing-stage":
				delete(p.Stages, "hc_ffn_mix")
			case "extra-stage":
				p.Stages["unexpected"] = p.Stages["hc_ffn_mix"]
			case "extra-raw":
				if err := os.WriteFile(filepath.Join(filepath.Dir(path), "unexpected.f32le"), []byte{0, 0, 0, 0}, 0600); err != nil {
					t.Fatal(err)
				}
			case "row2045":
				p.Identity.Route = "batched-l00"
				p.Identity.Rows = 2048
				p.Identity.SelectedRow = 2045
				want.Identity = p.Identity
			case "position":
				p.Identity.Position = 2045
			case "route":
				p.Identity.Route = "unknown"
			case "history":
				p.Identity.History = strings.Repeat("b", 64)
			case "extent":
				s := p.Stages["hc_ffn_mix"]
				s.Width--
				p.Stages["hc_ffn_mix"] = s
			case "nonfinite", "data-sha":
				s := p.Stages["hc_ffn_mix"]
				b := make([]byte, 96)
				binary.LittleEndian.PutUint32(b, math.Float32bits(float32(math.NaN())))
				if e := os.WriteFile(filepath.Join(filepath.Dir(path), s.File), b, 0600); e != nil {
					t.Fatal(e)
				}
				if name == "nonfinite" {
					s.SHA = fmt.Sprintf("%x", sha256.Sum256(b))
					p.Stages["hc_ffn_mix"] = s
				}
			}
			original := want.Identity
			want = seal(t, path, p)
			if name == "position" || name == "route" || name == "history" {
				want.Identity = original
			}
			if name == "missing-field" || name == "duplicate" || name == "trailing" || name == "null-row" {
				b, e := os.ReadFile(path)
				if e != nil {
					t.Fatal(e)
				}
				s := string(b)
				switch name {
				case "missing-field":
					s = strings.Replace(s, `,"selected_row":0`, "", 1)
				case "duplicate":
					s = strings.Replace(s, `"selected_row":0`, `"selected_row":0,"selected_row":0`, 1)
				case "trailing":
					s += "{}"
				case "null-row":
					s = strings.Replace(s, `"selected_row":0`, `"selected_row":null`, 1)
				}
				if e := os.WriteFile(path, []byte(s), 0600); e != nil {
					t.Fatal(e)
				}
				want.ManifestSHA = fmt.Sprintf("%x", sha256.Sum256([]byte(s)))
			}
			_, err := load(path, want)
			if (err == nil) != (name == "ordinary") {
				t.Fatalf("%s admission: %v", name, err)
			}
		})
	}
}

func TestBatchRowSelection(t *testing.T) {
	b := make([]byte, 2048*4)
	for row := range 2048 {
		binary.LittleEndian.PutUint32(b[row*4:], uint32(row))
	}
	id := identity{Rows: 2048, SelectedRow: 2047}
	if got := binary.LittleEndian.Uint32(selectedRow(b, id, 1)); got != 2047 {
		t.Fatalf("selected row %d, want 2047", got)
	}
}

func TestRow2045SourceMutation(t *testing.T) {
	if os.Getenv("L00_ROW_MUTANT_CHILD") == "1" {
		t.Skip("mutation recursion")
	}
	data, e := os.ReadFile("main.go")
	if e != nil {
		t.Fatal(e)
	}
	old := "id.SelectedRow * width * 4"
	source := strings.Replace(string(data), old, "(id.SelectedRow - 2) * width * 4", 1)
	if source == string(data) {
		t.Fatal("row-selection source site not found")
	}
	dir := t.TempDir()
	mutant := filepath.Join(dir, "main.go")
	if e := os.WriteFile(mutant, []byte(source), 0600); e != nil {
		t.Fatal(e)
	}
	abs, e := filepath.Abs("main.go")
	if e != nil {
		t.Fatal(e)
	}
	overlay, e := json.Marshal(map[string]any{"Replace": map[string]string{abs: mutant}})
	if e != nil {
		t.Fatal(e)
	}
	overlayPath := filepath.Join(dir, "overlay.json")
	if e := os.WriteFile(overlayPath, overlay, 0600); e != nil {
		t.Fatal(e)
	}
	cmd := exec.Command("go", "test", "-overlay", overlayPath, "-run", "^TestBatchRowSelection$", "-count=1", ".")
	cmd.Env = append(os.Environ(), "GOWORK=off", "L00_ROW_MUTANT_CHILD=1")
	out, err := cmd.CombinedOutput()
	if err == nil || !strings.Contains(string(out), "selected row 2045, want 2047") {
		t.Fatalf("source mutant must fail ordinary row assertion: %v\n%s", err, out)
	}
}

func TestCrossIdentity(t *testing.T) {
	p, _, w := fixture(t)
	a, e := load(p, w)
	if e != nil {
		t.Fatal(e)
	}
	rows, ids, e := compare(a, a)
	if e != nil || len(rows) != 10 || ids != 0 {
		t.Fatalf("ordinary compare: %v", e)
	}
	b := a
	b.identity.History = strings.Repeat("b", 64)
	if _, _, e := compare(a, b); e == nil {
		t.Fatal("history mismatch accepted")
	}
}

func TestFrozenInputs(t *testing.T) {
	if err := requireSameOperand("block", []float32{1}, []float32{1}); err != nil {
		t.Fatal(err)
	}
	for _, b := range [][]float32{{}, {2}} {
		if err := requireSameOperand("block", []float32{1}, b); err == nil {
			t.Fatal("unmatched frozen operand accepted")
		}
	}
}

func TestZeroReference(t *testing.T) {
	for _, value := range []float32{0, 1} {
		m := measure("zero", []float32{0}, []float32{value})
		if m.RelativeRMS != nil || m.RMS != float64(value) || m.MaxAbs != float64(value) {
			t.Fatalf("zero reference: %+v", m)
		}
	}
}

func TestDerivedOverflow(t *testing.T) {
	for _, route := range []string{"scalar-l00", "batched-l00"} {
		t.Run(route, func(t *testing.T) {
			x := extracted{identity: identity{Route: route}, floats: map[string][]float32{
				"ffn_out": make([]float32, 4096), "ffn_shexp": make([]float32, 4096),
				"ffn_moe_out": make([]float32, 4096), "ffn_expert_down": make([]float32, 24576),
			}}
			if route == "scalar-l00" {
				x.floats["ffn_expert_down"][0] = math.MaxFloat32
				x.floats["ffn_expert_down"][4096] = math.MaxFloat32
			} else {
				x.floats["ffn_moe_out"][0] = math.MaxFloat32
				x.floats["ffn_shexp"][0] = math.MaxFloat32
			}
			if err := deriveOperands(&x); err == nil {
				t.Fatal("finite raw operand overflow accepted")
			}
		})
	}
}
