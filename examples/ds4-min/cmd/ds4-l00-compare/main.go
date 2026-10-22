// ds4-l00-compare extracts sealed L00 operands and reports matched-stage errors.
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
)

type identity struct {
	Position    int    `json:"position"`
	Context     int    `json:"context"`
	History     string `json:"history_sha256"`
	Model       string `json:"model_sha256"`
	Source      string `json:"source_sha256"`
	Binary      string `json:"binary_sha256"`
	Route       string `json:"route"`
	Rows        int    `json:"rows"`
	SelectedRow int    `json:"selected_row"`
}

type stage struct {
	File  string `json:"file"`
	SHA   string `json:"sha256"`
	Kind  string `json:"kind"`
	Rows  int    `json:"rows"`
	Width int    `json:"width"`
}

type packet struct {
	Version  int              `json:"version"`
	Identity identity         `json:"identity"`
	Stages   map[string]stage `json:"stages"`
}

type expected struct {
	ManifestSHA string   `json:"manifest_sha256"`
	Identity    identity `json:"identity"`
}

type protocol struct {
	Version int      `json:"version"`
	Left    expected `json:"left"`
	Right   expected `json:"right"`
}

// rejectDuplicates walks the JSON token stream before typed decoding. The
// standard decoder otherwise silently accepts repeated identity fields.
func rejectDuplicates(d *json.Decoder) error {
	token, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := make(map[string]bool)
		for d.More() {
			k, e := d.Token()
			if e != nil {
				return e
			}
			name, ok := k.(string)
			if !ok || seen[name] {
				return fmt.Errorf("duplicate/invalid key %v", k)
			}
			seen[name] = true
			if e := rejectDuplicates(d); e != nil {
				return e
			}
		}
	case '[':
		for d.More() {
			if e := rejectDuplicates(d); e != nil {
				return e
			}
		}
	default:
		return fmt.Errorf("unexpected delimiter")
	}
	_, err = d.Token()
	return err
}

func decode(data []byte, out any) error {
	d := json.NewDecoder(bytes.NewReader(data))
	if err := rejectDuplicates(d); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return fmt.Errorf("trailing JSON")
	}
	if err := requireFields(data, reflect.TypeOf(out).Elem()); err != nil {
		return err
	}
	d = json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	return d.Decode(out)
}

func requireFields(data []byte, t reflect.Type) error {
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		return fmt.Errorf("null required value")
	}
	switch t.Kind() {
	case reflect.Struct:
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(data, &fields); err != nil {
			return err
		}
		for i := 0; i < t.NumField(); i++ {
			field := t.Field(i)
			name := field.Tag.Get("json")
			if name == "" {
				name = field.Name
			}
			value, ok := fields[name]
			if !ok {
				return fmt.Errorf("missing field %s", name)
			}
			if err := requireFields(value, field.Type); err != nil {
				return fmt.Errorf("%s: %w", name, err)
			}
		}
	case reflect.Map:
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(data, &fields); err != nil {
			return err
		}
		for name, value := range fields {
			if err := requireFields(value, t.Elem()); err != nil {
				return fmt.Errorf("%s: %w", name, err)
			}
		}
	}
	return nil
}

func validSHA(s string) bool { b, e := hex.DecodeString(s); return e == nil && len(b) == 32 }

func validateIdentity(x identity) error {
	if x.Position != 2047 || x.Context != 4096 {
		return fmt.Errorf("wrong fixture position/context")
	}
	for _, s := range []string{x.History, x.Model, x.Source, x.Binary} {
		if !validSHA(s) {
			return fmt.Errorf("missing/invalid identity hash")
		}
	}
	if x.Route == "scalar-l00" {
		if x.Rows != 1 || x.SelectedRow != 0 {
			return fmt.Errorf("scalar row contract")
		}
	} else if x.Route == "batched-l00" {
		if x.Rows != 2048 || x.SelectedRow != 2047 {
			return fmt.Errorf("batch row contract")
		}
	} else {
		return fmt.Errorf("unsupported route")
	}
	return nil
}

type operand struct {
	kind  string
	width int
}

func required(route string) map[string]operand {
	m := map[string]operand{
		"hc_attn_post": {"float32-le", 16384}, "ffn_cur": {"float32-le", 4096}, "ffn_norm": {"float32-le", 4096},
		"hc_ffn_mix": {"float32-le", 24}, "hc_ffn_split": {"float32-le", 24}, "ffn_moe_topk": {"int32-le", 6},
		"ffn_moe_weights_scaled": {"float32-le", 6}, "ffn_shexp": {"float32-le", 4096}, "hc_ffn_post": {"float32-le", 16384},
	}
	if route == "scalar-l00" {
		m["ffn_expert_down"] = operand{"float32-le", 24576}
		m["ffn_out"] = operand{"float32-le", 4096}
	} else {
		m["ffn_moe_out"] = operand{"float32-le", 4096}
	}
	return m
}

type extracted struct {
	identity identity
	floats   map[string][]float32
	ids      []int32
}

func load(path string, want expected) (extracted, error) {
	var out extracted
	if err := validateIdentity(want.Identity); err != nil {
		return out, err
	}
	if !validSHA(want.ManifestSHA) {
		return out, fmt.Errorf("missing expected manifest hash")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return out, err
	}
	if fmt.Sprintf("%x", sha256.Sum256(data)) != want.ManifestSHA {
		return out, fmt.Errorf("manifest seal mismatch")
	}
	var p packet
	if err := decode(data, &p); err != nil {
		return out, err
	}
	if p.Version != 1 || p.Identity != want.Identity {
		return out, fmt.Errorf("packet identity mismatch")
	}
	required := required(p.Identity.Route)
	if len(p.Stages) != len(required) {
		return out, fmt.Errorf("incomplete/extra stage inventory")
	}
	out = extracted{identity: p.Identity, floats: make(map[string][]float32)}
	files := make(map[string]bool)
	for name, shape := range required {
		s, ok := p.Stages[name]
		if !ok {
			return out, fmt.Errorf("missing stage %s", name)
		}
		if s.Rows != p.Identity.Rows || s.Width != shape.width || s.Kind != shape.kind || !validSHA(s.SHA) {
			return out, fmt.Errorf("%s: shape/type/hash mismatch", name)
		}
		if s.File == "" || filepath.IsAbs(s.File) || filepath.Clean(s.File) != s.File || filepath.Base(s.File) != s.File || files[s.File] {
			return out, fmt.Errorf("%s: invalid/reused stage path", name)
		}
		files[s.File] = true
		path := filepath.Join(filepath.Dir(path), s.File)
		info, err := os.Lstat(path)
		if err != nil {
			return out, err
		}
		if !info.Mode().IsRegular() {
			return out, fmt.Errorf("%s: not regular file", name)
		}
		if info.Size() != int64(s.Rows*s.Width*4) {
			return out, fmt.Errorf("%s: full extent mismatch", name)
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			return out, err
		}
		if fmt.Sprintf("%x", sha256.Sum256(raw)) != s.SHA {
			return out, fmt.Errorf("%s: data seal mismatch", name)
		}
		// Inspect all rows, not only the selected row; partial non-finite packets
		// cannot become a qualified finite row through selective extraction.
		if s.Kind == "float32-le" {
			for i := 0; i < len(raw); i += 4 {
				v := math.Float32frombits(binary.LittleEndian.Uint32(raw[i:]))
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					return out, fmt.Errorf("%s: non-finite element %d", name, i/4)
				}
			}
		}
		row := selectedRow(raw, p.Identity, s.Width)
		if s.Kind == "int32-le" {
			out.ids = make([]int32, s.Width)
			for i := range out.ids {
				out.ids[i] = int32(binary.LittleEndian.Uint32(row[4*i:]))
				if out.ids[i] < 0 || out.ids[i] >= 256 {
					return out, fmt.Errorf("invalid expert id")
				}
			}
		} else {
			v := make([]float32, s.Width)
			for i := range v {
				v[i] = math.Float32frombits(binary.LittleEndian.Uint32(row[4*i:]))
			}
			out.floats[name] = v
		}
	}
	entries, err := os.ReadDir(filepath.Dir(path))
	if err != nil {
		return out, err
	}
	for _, entry := range entries {
		name := entry.Name()
		raw := strings.HasSuffix(name, ".bin") || strings.HasSuffix(name, ".f32le") || strings.HasSuffix(name, ".i32le") || strings.HasSuffix(name, ".f32") || strings.HasSuffix(name, ".i32")
		if raw && !files[name] {
			return out, fmt.Errorf("unlisted raw stage %s", name)
		}
	}
	if err := deriveOperands(&out); err != nil {
		return out, err
	}
	return out, nil
}

func deriveOperands(out *extracted) error {
	if out.identity.Route == "scalar-l00" {
		routed := make([]float32, 4096)
		for d := range routed {
			for slot := range 6 {
				routed[d] += out.floats["ffn_expert_down"][slot*4096+d]
			}
		}
		out.floats["ffn_moe_out"] = routed
	} else {
		block := make([]float32, 4096)
		for d := range block {
			block[d] = out.floats["ffn_moe_out"][d] + out.floats["ffn_shexp"][d]
		}
		out.floats["ffn_out"] = block
	}
	for _, name := range []string{"ffn_moe_out", "ffn_out"} {
		for i, value := range out.floats[name] {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return fmt.Errorf("derived %s: non-finite element %d", name, i)
			}
		}
	}
	return nil
}

func selectedRow(raw []byte, id identity, width int) []byte {
	offset := id.SelectedRow * width * 4
	return raw[offset : offset+width*4]
}

type metric struct {
	Stage                     string
	RMS, ReferenceRMS, MaxAbs float64
	RelativeRMS               *float64
	Changed                   int
}

func measure(name string, a, b []float32) metric {
	m := metric{Stage: name}
	var square, reference float64
	for i, x := range a {
		d := float64(b[i]) - float64(x)
		square += d * d
		reference += float64(x) * float64(x)
		if math.Abs(d) > m.MaxAbs {
			m.MaxAbs = math.Abs(d)
		}
		if math.Float32bits(x) != math.Float32bits(b[i]) {
			m.Changed++
		}
	}
	m.RMS = math.Sqrt(square / float64(len(a)))
	m.ReferenceRMS = math.Sqrt(reference / float64(len(a)))
	if reference > 0 {
		ratio := math.Sqrt(square / reference)
		m.RelativeRMS = &ratio
	}
	return m
}

func compare(left, right extracted) ([]metric, int, error) {
	a, b := left.identity, right.identity
	if a.Position != b.Position || a.Context != b.Context || a.History != b.History || a.Model != b.Model {
		return nil, 0, fmt.Errorf("cross-arm fixture mismatch")
	}
	var rows []metric
	for _, name := range []string{"hc_attn_post", "ffn_cur", "ffn_norm", "hc_ffn_mix", "hc_ffn_split", "ffn_moe_weights_scaled", "ffn_shexp", "ffn_moe_out", "ffn_out", "hc_ffn_post"} {
		rows = append(rows, measure(name, left.floats[name], right.floats[name]))
	}
	ids := 0
	for i := range left.ids {
		if left.ids[i] != right.ids[i] {
			ids++
		}
	}
	return rows, ids, nil
}

func run() error {
	protocolPath := flag.String("protocol", "", "expected sealed identities")
	leftPath := flag.String("left", "", "left full-stage manifest")
	rightPath := flag.String("right", "", "right full-stage manifest")
	label := flag.String("comparison", "", "baseline or frozen-pair; never an acceptance verdict")
	flag.Parse()
	if flag.NArg() != 0 || *protocolPath == "" || *leftPath == "" || *rightPath == "" || (*label != "baseline" && *label != "frozen-pair") {
		return fmt.Errorf("require -protocol -left -right -comparison=baseline|frozen-pair")
	}
	b, err := os.ReadFile(*protocolPath)
	if err != nil {
		return err
	}
	var p protocol
	if err := decode(b, &p); err != nil {
		return err
	}
	if p.Version != 1 {
		return fmt.Errorf("unsupported protocol")
	}
	left, err := load(*leftPath, p.Left)
	if err != nil {
		return fmt.Errorf("left: %w", err)
	}
	right, err := load(*rightPath, p.Right)
	if err != nil {
		return fmt.Errorf("right: %w", err)
	}
	rows, ids, err := compare(left, right)
	if err != nil {
		return err
	}
	if *label == "frozen-pair" {
		for _, name := range []string{"hc_attn_post", "ffn_out", "hc_ffn_split"} {
			if err := requireSameOperand(name, left.floats[name], right.floats[name]); err != nil {
				return err
			}
		}
	}
	return json.NewEncoder(os.Stdout).Encode(struct {
		Status, Comparison string
		Metrics            []metric
		ExpertIDsDifferent int
	}{"MEASURED_DIAGNOSTIC_NO_PARITY_VERDICT", *label, rows, ids})
}

func requireSameOperand(name string, a, b []float32) error {
	if len(a) != len(b) {
		return fmt.Errorf("frozen-pair %s extent mismatch", name)
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return fmt.Errorf("frozen-pair %s differs at %d", name, i)
		}
	}
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "ds4-l00-compare:", err)
		os.Exit(1)
	}
}
