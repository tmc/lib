//go:build linux

package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// Blocks 1 and 71 are copied from the pinned layer0 oracle attention heads,
// input SHA256 e5d46f94373c8952789ab69884caa327b42873d57456243cb966cba9890eb767.
// Expected records are copied from the physical NVCC quantizer receipt,
// packed SHA256 17c846b22f954df6bc038cbf19d3bb666dbe74c72735dc10295dda77554024bf.
// The former default-mode records come from the same-input NVRTC receipt,
// packed SHA256 6be55ce0346a9a3798a81b7a83439c88a576299ded0775f3034f1ffd33ea8de2.
// Block 1 distinguishes codes; block 71 distinguishes scale precision.
func TestCUDAQualificationQ80OracleQuantizer(t *testing.T) {
	blocks := []struct {
		block        int
		input        [32]uint32
		oracleScale  uint32
		oracleCodes  [32]int8
		defaultScale uint32
		defaultCodes [32]int8
	}{
		{
			block: 1,
			input: [32]uint32{
				0xbdac328c, 0xbc91b49e, 0xbe397183, 0x3e91b49e, 0xbe11b49e, 0xbdc6b07a, 0xbe8475a7, 0xbd53ef71,
				0xbdd3ef71, 0xbcee6d5f, 0xbdee6d5f, 0xbe2c328c, 0xbe91b49e, 0xbdd3ef71, 0xbe8475a7, 0xbe53ef71,
				0x3e8475a7, 0x3dac328c, 0x3d2c328c, 0xbcee6d5f, 0xbcac328c, 0xbd6e6d5f, 0x3dc6b07a, 0xbdac328c,
				0xbe8475a7, 0xbd2c328c, 0xbe397183, 0x3e2c328c, 0x3dee6d5f, 0x3dac328c, 0xbe0475a7, 0xbdc6b07a,
			},
			oracleScale:  0x3b12da53,
			oracleCodes:  [32]int8{-38, -8, -81, 127, -63, -43, -115, -23, -46, -13, -52, -75, -127, -46, -115, -92, 115, 38, 19, -13, -9, -26, 43, -38, -115, -19, -81, 75, 52, 38, -58, -43},
			defaultScale: 0x3b12da53,
			defaultCodes: [32]int8{-38, -8, -81, 127, -64, -43, -115, -23, -46, -13, -52, -75, -127, -46, -115, -92, 115, 38, 19, -13, -9, -26, 43, -38, -115, -19, -81, 75, 52, 38, -58, -43},
		},
		{
			block: 71,
			input: [32]uint32{
				0x3d2bd6a5, 0x3e1f9075, 0xbd446306, 0xbe757bc8, 0xbef57bc8, 0x3d757bc8, 0x3e934a44, 0x3df57bc8,
				0x3f070414, 0x3f070414, 0x3e1f9075, 0xbe5cef67, 0x3e757bc8, 0x3d757bc8, 0xbdabd6a5, 0xbef57bc8,
				0xbe2bd6a5, 0xbec46306, 0x3f070414, 0xbe1f9075, 0x3e934a44, 0x3f5cef67, 0x3e381cd6, 0xbe934a44,
				0x3f446306, 0x3dc46306, 0x3e5cef67, 0x3e870414, 0xbe2bd6a5, 0xbe381cd6, 0x3df57bc8, 0x3b757bc8,
			},
			oracleScale:  0x3bdeacc0,
			oracleCodes:  [32]int8{6, 23, -7, -35, -71, 9, 42, 18, 78, 78, 23, -32, 35, 9, -12, -71, -25, -56, 78, -23, 42, 127, 26, -42, 113, 14, 32, 39, -25, -26, 18, 1},
			defaultScale: 0x3bdeacc1,
			defaultCodes: [32]int8{6, 23, -7, -35, -71, 9, 42, 18, 78, 78, 23, -32, 35, 9, -12, -71, -25, -56, 78, -23, 42, 127, 26, -42, 113, 14, 32, 39, -25, -26, 18, 1},
		},
	}
	x := make([]float32, len(blocks)*32)
	want, old := make([]byte, len(blocks)*36), make([]byte, len(blocks)*36)
	for b, block := range blocks {
		for i, bits := range block.input {
			x[b*32+i] = math.Float32frombits(bits)
		}
		binary.LittleEndian.PutUint32(want[b*36:], block.oracleScale)
		binary.LittleEndian.PutUint32(old[b*36:], block.defaultScale)
		for i := range 32 {
			want[b*36+4+i] = byte(block.oracleCodes[i])
			old[b*36+4+i] = byte(block.defaultCodes[i])
		}
	}
	check := func(got, expected []byte) {
		t.Helper()
		if len(got) != len(expected) {
			t.Fatalf("quantizer bytes: got %d, want %d", len(got), len(expected))
		}
		for b, block := range blocks {
			if !bytes.Equal(got[b*36:(b+1)*36], expected[b*36:(b+1)*36]) {
				t.Fatalf("oracle source block %d: got %x, want %x", block.block, got[b*36:(b+1)*36], expected[b*36:(b+1)*36])
			}
		}
	}
	r := cudaSynthFoundation(t)
	in := cudaSynthUpload(t, r, x)
	out := cudaExpertAllocBytes(t, r, len(want))
	for pass := range 2 {
		if pass == 1 {
			// Loading the other module must not replace the quantizer handle.
			if err := r.device.WithContext(r.loadMathKernels); err != nil {
				t.Fatal(err)
			}
		}
		if err := out.CopyFromHost(bytes.Repeat([]byte{0xa5}, out.Len())); err != nil {
			t.Fatal(err)
		}
		cudaQualificationMeasure(t, r, func() error { return r.encodeQuantQ80(in, out, len(x)) })
		got := make([]byte, out.Len())
		if err := out.CopyToHost(got); err != nil {
			t.Fatal(err)
		}
		check(got, want)
	}
	// The default compiler mode is only a negative control, not a production route.
	compiled, err := nvrtcx.Compile(cudaQ80QuantSource, nvrtcx.WithName("q80_default_negative.cu"), nvrtcx.WithArch(r.major, r.minor))
	if err != nil {
		t.Fatal(err)
	}
	var wrong []byte
	err = r.device.WithContext(func() error {
		mod, err := compiled.Load(r.device)
		if err != nil {
			return err
		}
		defer func() {
			if err := mod.Unload(); err != nil {
				t.Error(err)
			}
		}()
		fn, err := mod.GetFunction("ds4_quant_q8_0")
		if err != nil {
			return err
		}
		if err := out.CopyFromHost(bytes.Repeat([]byte{0xa5}, out.Len())); err != nil {
			return err
		}
		cudaQualificationMeasure(t, r, func() error {
			return cudalaunch.Kernel(fn).Grid(uint32(len(blocks))).Block(32).On(r.device.DefaultStream()).Launch(in, out)
		})
		wrong = make([]byte, out.Len())
		return out.CopyToHost(wrong)
	})
	if err != nil {
		t.Fatal(err)
	}
	check(wrong, old)
	if bytes.Equal(wrong, want) {
		t.Fatal("default mode did not discriminate")
	}
}
