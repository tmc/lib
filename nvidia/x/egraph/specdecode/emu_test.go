package specdecode

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// An emuBuf is one kernel buffer for runEmu. Every buffer is read back after
// the launch; out-only buffers start zeroed with len(data) bytes.
type emuBuf struct {
	name  string
	ctype string
	data  []byte
}

// runEmu compiles hopper.Prelude+src as host C++ against cudahostemu.Header, runs
// launch (a C++ statement that may name the buffers), and returns every
// buffer's final contents.
func runEmu(t *testing.T, src, launch string, bufs ...emuBuf) map[string][]byte {
	t.Helper()
	res, err := runEmuErr(t, src, launch, bufs...)
	if err != nil {
		t.Fatalf("emulated kernel: %v\n--- source ---\n%s", err, numbered(hopper.Prelude+src))
	}
	return res
}

// runEmuErr is runEmu returning the compile or run error, for tests that
// expect the emulation to fail.
func runEmuErr(t *testing.T, src, launch string, bufs ...emuBuf) (map[string][]byte, error) {
	t.Helper()
	cbufs := make([]cudahostemu.Buffer, len(bufs))
	for i, b := range bufs {
		cbufs[i] = cudahostemu.Buffer{Name: b.name, CType: b.ctype, Data: b.data}
	}
	res, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
		Source:  hopper.Prelude + src,
		Launch:  launch,
		Buffers: cbufs,
		Flags:   strings.Fields(os.Getenv("EG_EMU_FLAGS")),
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler; emulated kernel not run")
	}
	return res, err
}

func numbered(s string) string {
	var b strings.Builder
	for i, line := range strings.Split(s, "\n") {
		fmt.Fprintf(&b, "%4d %s\n", i+1, line)
	}
	return b.String()
}

func f32Bytes(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f))
	}
	return b
}

func bytesF32(b []byte) []float32 {
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v
}

func u16Bytes(v []uint16) []byte {
	b := make([]byte, 2*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint16(b[2*i:], x)
	}
	return b
}

func bytesU16(b []byte) []uint16 {
	v := make([]uint16, len(b)/2)
	for i := range v {
		v[i] = binary.LittleEndian.Uint16(b[2*i:])
	}
	return v
}

func i32Bytes(v []int32) []byte {
	return specI32Bytes(v)
}

func bytesI32(b []byte) []int32 {
	return specBytesI32(b)
}

func u32Bytes(v []uint32) []byte {
	return u32le(v)
}

func bytesU32(b []byte) []uint32 {
	return le32u(b)
}

// poison returns n bytes of 0xff (NaN in every float format the kernels
// use), standing in for a fresh device allocation a kernel must overwrite.
func poison(n int) []byte { return bytes.Repeat([]byte{0xff}, n) }

func agree(a, b []float64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.IsNaN(a[i]) || math.IsNaN(b[i]) || math.Abs(a[i]-b[i]) > tol {
			return false
		}
	}
	return true
}

func formatI32List(v []int32) string {
	parts := make([]string, len(v))
	for i, x := range v {
		parts[i] = strconv.Itoa(int(x))
	}
	return strings.Join(parts, ", ")
}
