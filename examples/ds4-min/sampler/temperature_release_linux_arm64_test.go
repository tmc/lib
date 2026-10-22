//go:build linux && arm64

package sampler

import (
	"compress/gzip"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTemperatureReleaseRows(t *testing.T) {
	tests := []struct {
		name, file, sha string
		before, after   uint64
		token           int
	}{
		{"G2-seed41-step10", "step-0010.f32.gz.b64", "98663dbbbcda7e5f98db1bc92c656ebabcc15bfc32a54f994581b7f0268cfc05", 0x19a9ad331d7dd1d3, 0x4a34afac94d9f2b4, 74807},
		{"G2-seed69-step8", "step-0008.f32.gz.b64", "12e582946fa4b1594a574f6d29aaf893a23386a0a36e99394e77d393e27906f2", 0x3e514739d53ff89c, 0x651fe777c35e440f, 5967},
		{"G2-seed149-step5", "step-0005.f32.gz.b64", "d91b9134f257825c9ff0ea44e3a7b9f2a98ff775c76b8635ed272884b4645cdd", 0x85bcdf82cd7db25c, 0x1bde26874095b500, 121064},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			encoded, err := os.ReadFile(filepath.Join("testdata", "temperature", test.file))
			if err != nil {
				t.Fatal(err)
			}
			reader, err := gzip.NewReader(base64.NewDecoder(base64.StdEncoding, strings.NewReader(string(encoded))))
			if err != nil {
				t.Fatal(err)
			}
			defer reader.Close()
			data, err := io.ReadAll(io.LimitReader(reader, 129280*4+1))
			if err != nil {
				t.Fatal(err)
			}
			if len(data) != 129280*4 || fmt.Sprintf("%x", sha256.Sum256(data)) != test.sha {
				t.Fatal("retained release row length or hash changed")
			}
			logits := make([]float32, 129280)
			for i := range logits {
				logits[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
				if !isFinite(logits[i]) {
					t.Fatal("retained release row is not finite")
				}
			}
			rng := RNG{State: test.before}
			got := New(len(logits)).Sample(logits, Params{Temperature: 1.3, TopP: 1}, &rng)
			if got != test.token || rng.State != test.after {
				t.Fatalf("release temperature mismatch: token=%d state=%016x want token=%d state=%016x", got, rng.State, test.token, test.after)
			}
		})
	}
	t.Log("retained release temperature cases=3")
}
