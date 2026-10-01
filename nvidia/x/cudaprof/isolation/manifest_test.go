package isolation

import (
	"encoding/json"
	"strings"
	"testing"
)

func manifestFixture() Manifest {
	hash := strings.Repeat("0", 64)
	return Manifest{Version: 1, Role: "gpu-isolation-binding", Sources: []Source{{Attachment: "receipt.json", SHA256: hash}}, Hops: []Hop{{CaptureSHA256: hash, ReceiptSHA256: hash}}, Consumer: "fixture"}
}
func TestReadManifest(t *testing.T) {
	m := manifestFixture()
	raw, _ := json.Marshal(m)
	if _, hash, err := ReadManifest(strings.NewReader(string(raw))); err != nil || len(hash) != 64 {
		t.Fatal(hash, err)
	}
	for _, change := range []func(*Manifest){func(m *Manifest) { m.Role = "other" }, func(m *Manifest) { m.Version = 2 }, func(m *Manifest) { m.Sources[0].Attachment = "../outside" }, func(m *Manifest) { m.Sources[0].SHA256 = "bad" }, func(m *Manifest) { m.Hops[0].ReceiptSHA256 = strings.Repeat("1", 64) }, func(m *Manifest) { m.Sources = append(m.Sources, m.Sources[0]) }} {
		m := manifestFixture()
		change(&m)
		raw, _ := json.Marshal(m)
		if _, _, err := ReadManifest(strings.NewReader(string(raw))); err == nil {
			t.Fatal("accepted invalid manifest")
		}
	}
	for _, raw := range []string{string(raw) + " {}", `{"version":1,"version":1}`, strings.Repeat("x", MaxBytes+1)} {
		if _, _, err := ReadManifest(strings.NewReader(raw)); err == nil {
			t.Fatal("accepted invalid JSON")
		}
	}
}
