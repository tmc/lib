package dsv4cpu

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// TestQKVRetainedScalarDots replays the supplied quantized inputs only. It
// neither binds a model nor executes the C diagnostic again.
func TestQKVRetainedScalarDots(t *testing.T) {
	dir := os.Getenv("DS4_CPU_QKV_REPLAY")
	if dir == "" {
		t.Skip("DS4_CPU_QKV_REPLAY not set")
	}
	read := func(name, hash string) []byte {
		t.Helper()
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(b)) != hash {
			t.Fatalf("retained %s identity differs", name)
		}
		return b
	}
	input := read("QKV.input", "7ac382795eef76b0e5b4cd29e7b6eb7b68530ad2b64055ed6673661ee86d2e11")
	output := read("QKV.C.f32", "0070586dc08a20dbb702f0cccfdf23b7af67422ec3d832f7bb94343c26b1ee4a")
	const weights = (1024*128 + 32768*32 + 512*128) * 34
	const floats = 1024 + 512 + 4096 + 4096 + 1024 + 32768 + 512 + 32768 + 512
	const quantized = 4096 + 128*4 + 1024 + 32*4
	if len(input) != weights+floats*4+quantized || len(output) != 346400*4 {
		t.Fatal("retained QKV frame dimensions differ")
	}
	p := input[weights+floats*4:]
	activation := func(n int) []Q80Activation {
		x := make([]Q80Activation, n/32)
		for b := range x {
			for j := range x[b].Values {
				x[b].Values[j] = int8(p[b*32+j])
			}
			x[b].Scale = math.Float32frombits(binary.LittleEndian.Uint32(p[n+b*4:]))
		}
		p = p[n+n/32*4:]
		return x
	}
	qa := activation(4096)
	qb := activation(1024)
	weightOffset, valueOffset := 0, 177952
	for _, tc := range []struct {
		name    string
		rows    int
		changed int
		x       []Q80Activation
	}{{"qa", 1024, 895, qa}, {"qb", 32768, 26207, qb}, {"kv", 512, 433, qa}} {
		t.Run(tc.name, func(t *testing.T) {
			rowBytes := len(tc.x) * 34
			changed := 0
			for row := 0; row < tc.rows; row++ {
				w := input[weightOffset+row*rowBytes:]
				want := binary.LittleEndian.Uint32(output[(valueOffset+row)*4:])
				got := math.Float32bits(scalarQ80Dot(w, tc.x))
				if got != want {
					t.Fatalf("row %d scalar=%08x C=%08x", row, got, want)
				}
				if runtime.GOOS == "linux" && runtime.GOARCH == "arm64" && math.Float32bits(nativeQ80Dot(w, tc.x)) != want {
					t.Fatalf("row %d Linux native reduction differs", row)
				}
				if math.Float32bits(dotprodQ80Dot(w, tc.x)) != want {
					changed++
				}
			}
			if changed != tc.changed {
				t.Fatalf("DOTPROD changed=%d want=%d", changed, tc.changed)
			}
			t.Logf("scalar exact=%d/%d; DOTPROD changed=%d", tc.rows, tc.rows, changed)
		})
		weightOffset += tc.rows * len(tc.x) * 34
		valueOffset += tc.rows
	}
}
