package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
)

type headPacket struct {
	Version     int                  `json:"version"`
	Model       string               `json:"model_sha256"`
	History     string               `json:"history_sha256"`
	Source      string               `json:"source_sha256"`
	Binary      string               `json:"binary_sha256"`
	Position    *int                 `json:"position"`
	Context     int                  `json:"context"`
	Route       string               `json:"route"`
	ModelBasis  string               `json:"model_sha256_basis"`
	SourceBasis string               `json:"source_sha256_basis"`
	BinaryBasis string               `json:"binary_sha256_basis"`
	Stages      map[string]headStage `json:"stages"`
	RouteRecord *headStage           `json:"route_record,omitempty"`
}

type headStage struct {
	File string `json:"file"`
	SHA  string `json:"sha256"`
}

// checkJSONKeys rejects duplicate keys at every nesting depth. DisallowUnknownFields
// alone is insufficient: encoding/json otherwise silently takes the last value.
func checkJSONKeys(d *json.Decoder) error {
	t, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := t.(json.Delim)
	if !ok {
		return nil
	}
	if delim != '{' {
		return fmt.Errorf("packet arrays are unsupported")
	}
	seen := make(map[string]bool)
	for d.More() {
		key, err := d.Token()
		if err != nil {
			return err
		}
		s, ok := key.(string)
		if !ok || seen[s] {
			return fmt.Errorf("duplicate or invalid packet key %q", key)
		}
		seen[s] = true
		if err := checkJSONKeys(d); err != nil {
			return err
		}
	}
	_, err = d.Token()
	return err
}

func readHeadPacket(path string) (headPacket, map[string][]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return headPacket{}, nil, err
	}
	d := json.NewDecoder(bytes.NewReader(data))
	if err := checkJSONKeys(d); err != nil {
		return headPacket{}, nil, err
	}
	if _, err := d.Token(); err != io.EOF {
		return headPacket{}, nil, fmt.Errorf("trailing packet JSON")
	}
	var p headPacket
	d = json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(&p); err != nil {
		return p, nil, err
	}
	if p.Version != 1 || p.Position == nil || *p.Position != 2047 || p.Context != 4096 || len(p.Stages) != 4 {
		return p, nil, fmt.Errorf("invalid packet version, position or stage count")
	}
	if p.ModelBasis != "external-receipt-assertion" || p.SourceBasis != "actual-source-manifest-file" || p.BinaryBasis != "actual-executable-file" {
		return p, nil, fmt.Errorf("missing or unsupported provenance basis")
	}
	for _, hash := range []string{p.Model, p.History, p.Source, p.Binary} {
		if len(hash) != 64 {
			return p, nil, fmt.Errorf("missing packet provenance hash")
		}
		for _, c := range hash {
			if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
				return p, nil, fmt.Errorf("invalid provenance hash")
			}
		}
	}
	switch p.Route {
	case "raw-q8_0-q8_1", "aligned-q8_0-q8_1", "aligned-q8_0-folded-q8_1", "aligned-q8_0-scratch-q8_1":
	default:
		return p, nil, fmt.Errorf("unsupported or unread head route")
	}
	if (p.Route == "aligned-q8_0-folded-q8_1" || p.Route == "aligned-q8_0-scratch-q8_1") && p.RouteRecord == nil {
		return p, nil, fmt.Errorf("missing sealed C route record")
	}
	out := make(map[string][]byte)
	files := make(map[string]bool)
	if s := p.RouteRecord; s != nil {
		if s.File == "" || filepath.Base(s.File) != s.File || s.File == "." {
			return p, nil, fmt.Errorf("invalid route record path")
		}
		data, err := os.ReadFile(filepath.Join(filepath.Dir(path), s.File))
		if err != nil {
			return p, nil, err
		}
		if fmt.Sprintf("%x", sha256.Sum256(data)) != s.SHA || string(data) != p.Route+"\n" {
			return p, nil, fmt.Errorf("route record mismatch")
		}
		files[s.File] = true
	}
	for name, size := range map[string]int{"hc": 4096 * 4, "norm": 4096 * 4, "q81": 128 * 36, "logits": 129280 * 4} {
		s, ok := p.Stages[name]
		if !ok || s.File == "" || filepath.Base(s.File) != s.File || s.File == "." || files[s.File] {
			return p, nil, fmt.Errorf("missing or invalid %s stage", name)
		}
		files[s.File] = true
		b, err := os.ReadFile(filepath.Join(filepath.Dir(path), s.File))
		if err != nil {
			return p, nil, err
		}
		if len(b) != size || fmt.Sprintf("%x", sha256.Sum256(b)) != s.SHA {
			return p, nil, fmt.Errorf("%s extent or seal mismatch", name)
		}
		if name == "q81" {
			for block := range 128 {
				for _, off := range []int{0, 2} {
					bits := binary.LittleEndian.Uint16(b[block*36+off:])
					if bits&0x7c00 == 0x7c00 || off == 0 && bits&0x8000 != 0 {
						return p, nil, fmt.Errorf("invalid q8_1 half field")
					}
				}
			}
		} else {
			for i := 0; i < len(b); i += 4 {
				v := math.Float32frombits(binary.LittleEndian.Uint32(b[i:]))
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					return p, nil, fmt.Errorf("non-finite %s stage", name)
				}
			}
		}
		out[name] = b
	}
	entries, err := os.ReadDir(filepath.Dir(path))
	if err != nil {
		return p, nil, err
	}
	for _, e := range entries {
		if e.IsDir() || e.Type()&os.ModeSymlink != 0 || e.Name() != filepath.Base(path) && !files[e.Name()] {
			return p, nil, fmt.Errorf("unsealed packet entry %s", e.Name())
		}
	}
	return p, out, nil
}

func compareHeadPackets(left, right string, w io.Writer) error {
	var report bytes.Buffer
	a, x, err := readHeadPacket(left)
	if err != nil {
		return fmt.Errorf("left head packet: %w", err)
	}
	b, y, err := readHeadPacket(right)
	if err != nil {
		return fmt.Errorf("right head packet: %w", err)
	}
	if a.Model != b.Model || a.History != b.History || *a.Position != *b.Position {
		return fmt.Errorf("head packets have different model, history or position")
	}
	for _, name := range []string{"hc", "norm", "logits"} {
		var maxAbs, sum, reference float64
		different := 0
		for i := 0; i < len(x[name]); i += 4 {
			u := binary.LittleEndian.Uint32(x[name][i:])
			v := binary.LittleEndian.Uint32(y[name][i:])
			if u != v {
				different++
			}
			c, g := float64(math.Float32frombits(u)), float64(math.Float32frombits(v))
			d := g - c
			maxAbs = math.Max(maxAbs, math.Abs(d))
			sum += d * d
			reference += c * c
		}
		ratio := "UNREAD-zero-reference"
		if reference > 0 {
			ratio = fmt.Sprintf("%.9g", math.Sqrt(sum/reference))
		}
		fmt.Fprintf(&report, "%s changed=%d max_abs=%.9g rms=%.9g relative_rms=%s\n", name, different, maxAbs, math.Sqrt(sum/float64(len(x[name])/4)), ratio)
	}
	var scales, sums, codes, blocks int
	for block := range 128 {
		i := block * 36
		if !bytes.Equal(x["q81"][i:i+36], y["q81"][i:i+36]) {
			blocks++
		}
		if !bytes.Equal(x["q81"][i:i+2], y["q81"][i:i+2]) {
			scales++
		}
		if !bytes.Equal(x["q81"][i+2:i+4], y["q81"][i+2:i+4]) {
			sums++
		}
		for j := 4; j < 36; j++ {
			if x["q81"][i+j] != y["q81"][i+j] {
				codes++
			}
		}
	}
	fmt.Fprintf(&report, "q81 changed_blocks=%d scales=%d sums_unused=%d codes=%d\npacket admission PASS; numerical parity UNREAD\n", blocks, scales, sums, codes)
	_, err = io.Copy(w, &report)
	return err
}
