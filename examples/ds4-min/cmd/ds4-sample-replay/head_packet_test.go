package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func testHeadPacket(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	position := 2047
	p := headPacket{Version: 1, Model: strings.Repeat("1", 64), History: strings.Repeat("2", 64), Source: strings.Repeat("3", 64), Binary: strings.Repeat("4", 64), Position: &position, Route: "aligned-q8_0-q8_1", Stages: make(map[string]headStage)}
	p.ModelBasis, p.SourceBasis, p.BinaryBasis = "external-receipt-assertion", "actual-source-manifest-file", "actual-executable-file"
	p.Context = 4096
	for name, size := range map[string]int{"hc": 4096 * 4, "norm": 4096 * 4, "q81": 128 * 36, "logits": 129280 * 4} {
		data := make([]byte, size)
		p.Stages[name] = headStage{File: name + ".bin", SHA: headTestHash(data)}
		if err := os.WriteFile(filepath.Join(dir, name+".bin"), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	path := filepath.Join(dir, "packet.json")
	headTestManifest(t, path, p)
	return path
}

func headTestHash(data []byte) string { return fmtHash(sha256.Sum256(data)) }
func fmtHash(hash [32]byte) string {
	const hex = "0123456789abcdef"
	b := make([]byte, 64)
	for i, v := range hash {
		b[i*2] = hex[v>>4]
		b[i*2+1] = hex[v&15]
	}
	return string(b)
}

func headTestManifest(t *testing.T, path string, p headPacket) {
	t.Helper()
	b, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
}

func TestHeadPacketCRouteRecord(t *testing.T) {
	for _, route := range []string{"aligned-q8_0-folded-q8_1", "aligned-q8_0-scratch-q8_1"} {
		for _, mutation := range []string{"none", "missing", "null", "wrong-seal", "rehashed-body"} {
			t.Run(route+"/"+mutation, func(t *testing.T) {
				path := testHeadPacket(t)
				p, _, err := readHeadPacket(path)
				if err != nil {
					t.Fatal(err)
				}
				p.Route = route
				data := []byte(route + "\n")
				p.RouteRecord = &headStage{File: "head.route", SHA: headTestHash(data)}
				if mutation == "rehashed-body" {
					data = []byte("aligned-q8_0-q8_1\n")
					p.RouteRecord.SHA = headTestHash(data)
				}
				if mutation == "wrong-seal" {
					p.RouteRecord.SHA = strings.Repeat("0", 64)
				}
				if err := os.WriteFile(filepath.Join(filepath.Dir(path), "head.route"), data, 0600); err != nil {
					t.Fatal(err)
				}
				if mutation == "missing" || mutation == "null" {
					p.RouteRecord = nil
					if err := os.Remove(filepath.Join(filepath.Dir(path), "head.route")); err != nil {
						t.Fatal(err)
					}
				}
				headTestManifest(t, path, p)
				if mutation == "null" {
					b, err := os.ReadFile(path)
					if err != nil {
						t.Fatal(err)
					}
					b = append([]byte(`{"route_record":null,`), b[1:]...)
					if err := os.WriteFile(path, b, 0600); err != nil {
						t.Fatal(err)
					}
				}
				_, _, err = readHeadPacket(path)
				if mutation == "none" && err != nil {
					t.Fatal(err)
				}
				if mutation != "none" && err == nil {
					t.Fatal("C route mutation admitted")
				}
			})
		}
	}
}

func TestHeadPacketAdmissionMutations(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*testing.T, string, headPacket)
	}{
		{"missing-position", func(t *testing.T, path string, p headPacket) { p.Position = nil; headTestManifest(t, path, p) }},
		{"wrong-context", func(t *testing.T, path string, p headPacket) { p.Context = 8192; headTestManifest(t, path, p) }},
		{"missing-stage", func(t *testing.T, path string, p headPacket) { delete(p.Stages, "norm"); headTestManifest(t, path, p) }},
		{"wrong-shape", func(t *testing.T, path string, p headPacket) {
			os.WriteFile(filepath.Join(filepath.Dir(path), "norm.bin"), []byte{0}, 0600)
		}},
		{"duplicate-key", func(t *testing.T, path string, p headPacket) {
			b, _ := os.ReadFile(path)
			os.WriteFile(path, append([]byte(`{"version":1,`), b[1:]...), 0600)
		}},
		{"extra-file", func(t *testing.T, path string, p headPacket) {
			os.WriteFile(filepath.Join(filepath.Dir(path), "old.f32"), nil, 0600)
		}},
		{"unread-route", func(t *testing.T, path string, p headPacket) { p.Route = "UNREAD"; headTestManifest(t, path, p) }},
		{"path-escape", func(t *testing.T, path string, p headPacket) {
			s := p.Stages["norm"]
			s.File = "../norm.bin"
			p.Stages["norm"] = s
			headTestManifest(t, path, p)
		}},
		{"seal", func(t *testing.T, path string, p headPacket) {
			b, _ := os.ReadFile(filepath.Join(filepath.Dir(path), "q81.bin"))
			b[4] = 1
			os.WriteFile(filepath.Join(filepath.Dir(path), "q81.bin"), b, 0600)
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			path := testHeadPacket(t)
			p, _, err := readHeadPacket(path)
			if err != nil {
				t.Fatal(err)
			}
			test.mutate(t, path, p)
			if _, _, err := readHeadPacket(path); err == nil {
				t.Fatal("mutation admitted")
			}
		})
	}
}

func TestHeadPacketValueMutations(t *testing.T) {
	for _, test := range []struct {
		name   string
		stage  string
		offset int
		value  byte
		want   string
	}{
		{"coordinate", "norm", 2, 128, "norm changed=1"},
		{"code", "q81", 4, 1, "codes=1"},
		{"scale", "q81", 1, 60, "scales=1"},
		{"unused-sum", "q81", 3, 60, "sums_unused=1"},
	} {
		t.Run(test.name, func(t *testing.T) {
			left, right := testHeadPacket(t), testHeadPacket(t)
			p, data, err := readHeadPacket(right)
			if err != nil {
				t.Fatal(err)
			}
			b := data[test.stage]
			b[test.offset] = test.value
			s := p.Stages[test.stage]
			s.SHA = headTestHash(b)
			p.Stages[test.stage] = s
			if err := os.WriteFile(filepath.Join(filepath.Dir(right), s.File), b, 0600); err != nil {
				t.Fatal(err)
			}
			headTestManifest(t, right, p)
			var out bytes.Buffer
			if err := compareHeadPackets(left, right, &out); err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(out.String(), test.want) {
				t.Fatalf("mutation not reported: %s", out.String())
			}
		})
	}
}

func TestHeadPacketHistoryMismatch(t *testing.T) {
	left, right := testHeadPacket(t), testHeadPacket(t)
	p, _, err := readHeadPacket(right)
	if err != nil {
		t.Fatal(err)
	}
	p.History = strings.Repeat("5", 64)
	headTestManifest(t, right, p)
	if err := compareHeadPackets(left, right, &bytes.Buffer{}); err == nil {
		t.Fatal("different history admitted")
	}
}

func TestHeadPacketNonFiniteSealedMutation(t *testing.T) {
	for _, name := range []string{"norm", "q81"} {
		t.Run(name, func(t *testing.T) {
			path := testHeadPacket(t)
			p, data, err := readHeadPacket(path)
			if err != nil {
				t.Fatal(err)
			}
			b := data[name]
			if name == "norm" {
				b[2], b[3] = 128, 127
			} else {
				b[1] = 124
			}
			s := p.Stages[name]
			s.SHA = headTestHash(b)
			p.Stages[name] = s
			if err := os.WriteFile(filepath.Join(filepath.Dir(path), s.File), b, 0600); err != nil {
				t.Fatal(err)
			}
			headTestManifest(t, path, p)
			if _, _, err := readHeadPacket(path); err == nil {
				t.Fatal("sealed non-finite mutation admitted")
			}
		})
	}
}

type headFailedWriter struct{}

func (headFailedWriter) Write([]byte) (int, error) { return 0, errors.New("output unavailable") }

func TestHeadPacketReportFailure(t *testing.T) {
	if err := compareHeadPackets(testHeadPacket(t), testHeadPacket(t), headFailedWriter{}); err == nil {
		t.Fatal("report failure admitted")
	}
}
