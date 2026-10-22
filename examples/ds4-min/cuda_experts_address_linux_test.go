//go:build linux

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4expertcache"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// Explicit admission keeps an ordinary host suite from allocating a device.
// Qualification additionally requires the external GPU lock and no skips.
func TestCUDAExpertAddressNativeOracle(t *testing.T) {
	if os.Getenv("DS4_CUDA_ADDRESS_NATIVE") != "1" {
		t.Skip("native address gate not admitted")
	}
	r := cudaSynthFoundation(t)
	const width, hidden, experts = 512, 256, 4
	ids := []int{3, 1, 3, 0}
	ids32 := []int32{3, 1, 3, 0}
	g := cudaExpertFixture(16, width, hidden, experts, 17)
	u := cudaExpertFixture(16, width, hidden, experts, 29)
	d := cudaExpertFixture(10, hidden, width, experts, 43)
	raw := append(append(append([]byte{}, g...), u...), d...)
	identity := dsv4expertcache.Identity{Model: sha256.Sum256(raw), Quant: sha256.Sum256([]byte("IQ2_XXS/Q2_K raw")), Layout: sha256.Sum256([]byte("component expert row block"))}
	strides := [3]int{len(g) / experts, len(u) / experts, len(d) / experts}
	tr := dsv4expertcache.Triplet{
		Gate: dsv4expertcache.Tensor{Offset: 0, Bytes: uint64(len(g)), Stride: uint64(strides[0]), Experts: experts},
		Up:   dsv4expertcache.Tensor{Offset: uint64(len(g)), Bytes: uint64(len(u)), Stride: uint64(strides[1]), Experts: experts},
		Down: dsv4expertcache.Tensor{Offset: uint64(len(g) + len(u)), Bytes: uint64(len(d)), Stride: uint64(strides[2]), Experts: experts},
	}
	source, err := dsv4expertcache.NewSource(bytes.NewReader(raw), uint64(len(raw)), identity, map[int]dsv4expertcache.Triplet{0: tr})
	if err != nil {
		t.Fatal(err)
	}
	upload, err := dsv4expertcache.NewCUDAUpload(r.device, 3, strides, uint64(3*(strides[0]+strides[1]+strides[2])))
	if upload != nil {
		t.Cleanup(func() {
			if err := upload.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	owner, err := dsv4expertcache.NewCUDASelections(source, upload)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := owner.Close(); err != nil {
			t.Error(err)
		}
	})
	selection, err := owner.Select(0, ids, false)
	if err != nil {
		t.Fatal(err)
	}
	launch, err := newCUDAExpertAddressLaunch(selection, r.stream(), len(ids), width, hidden)
	if launch != nil {
		t.Cleanup(func() {
			if err := launch.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := nvrtcx.Compile(cudaProbeSource+cudaExpertSource+cudaExpertAddressSource, nvrtcx.WithDeviceArch(r.device), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	module, err := compiled.LoadModule(r.device)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := module.Close(); err != nil {
			t.Error(err)
		}
	})
	gateKernel, err := module.Function("ds4_iq2_xxs_pair_address_f32")
	if err != nil {
		t.Fatal(err)
	}
	downKernel, err := module.Function("ds4_q2_k_address_f32")
	if err != nil {
		t.Fatal(err)
	}
	x, sx := make([]float32, width), make([]float32, hidden*len(ids))
	for i := range x {
		x[i] = float32(i%29-14) / 16
	}
	for i := range sx {
		sx[i] = float32((i*7+i/hidden)%31-15) / 16
	}
	dx, dsx := cudaSynthUpload(t, r, x), cudaSynthUpload(t, r, sx)
	goFull, upFull, downFull := cudaTestAlloc(t, r, hidden*len(ids)+2), cudaTestAlloc(t, r, hidden*len(ids)+2), cudaTestAlloc(t, r, width*len(ids)+2)
	for _, dst := range []struct {
		n   int
		put func([]float32) error
	}{{goFull.Len(), goFull.CopyFromHost}, {upFull.Len(), upFull.CopyFromHost}, {downFull.Len(), downFull.CopyFromHost}} {
		p := make([]float32, dst.n)
		for i := range p {
			p[i] = 12345
		}
		if err := dst.put(p); err != nil {
			t.Fatal(err)
		}
	}
	goOut, upOut, downOut := goFull.Slice(1, goFull.Len()-1), upFull.Slice(1, upFull.Len()-1), downFull.Slice(1, downFull.Len()-1)
	if err := launch.gateUp(gateKernel, dx, goOut, upOut); err != nil {
		t.Fatal(err)
	}
	if err := launch.down(downKernel, dsx, downOut); err != nil {
		t.Fatal(err)
	}
	if err := r.stream().Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name       string
		raw        []byte
		typ        uint32
		cols, rows int
		input      []float32
		slotted    bool
		got        []float32
	}{
		{"gate", g, 16, width, hidden, x, false, cudaTestRead(t, goFull)},
		{"up", u, 16, width, hidden, x, false, cudaTestRead(t, upFull)},
		{"down", d, 10, hidden, width, sx, true, cudaTestRead(t, downFull)},
	} {
		if tc.got[0] != 12345 || tc.got[len(tc.got)-1] != 12345 {
			t.Fatalf("%s output canary changed", tc.name)
		}
		want := cudaExpertReference(t, tc.raw, tc.typ, tc.cols, tc.rows, ids32, tc.input, tc.slotted)
		if err := cudaQualificationClose(tc.got[1:len(tc.got)-1], want, 1e-4, 1e-4); err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
	}
	badBytes := make([]uint64, len(ids))
	for i := range badBytes {
		badBytes[i] = uint64(strides[2])
	}
	badBytes[0]--
	if err := launch.tables[5].CopyFromHost(badBytes); err != nil {
		t.Fatal(err)
	}
	if err := launch.down(downKernel, dsx, downOut); err != nil {
		t.Fatal(err)
	}
	if err := launch.Close(); err != nil {
		t.Fatal(err)
	}
	bad := cudaTestRead(t, downFull)
	if bad[0] != 12345 || bad[len(bad)-1] != 12345 {
		t.Fatal("malformed extent changed canary")
	}
	for _, v := range bad[1 : 1+width] {
		if !math.IsNaN(float64(v)) {
			t.Fatal("malformed extent was consumed")
		}
	}
	if err := launch.gateUp(gateKernel, dx, goOut, upOut); err == nil {
		t.Fatal("post-completion launch admitted")
	}
}
