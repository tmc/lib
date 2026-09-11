//go:build linux

package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

const (
	wantIQ2GridSHA256  = "05826b5d3e472a3a78f196be62ac78acf81df0f909626e12ab9fa2a5d490dd54"
	wantIQ2SignsSHA256 = "a76742a603f8beca5212ecce0f1f02f11a4887fd2f7c8b15aca0ea3eb3380c31"
)

// TestCUDAExpertTableDigests pins the byte-level representation of the CUDA
// IQ2_XXS codebook tables against the registered SHA-256 digests shared across
// both Metal and CUDA backends.
func TestCUDAExpertTableDigests(t *testing.T) {
	reSigns := regexp.MustCompile(`ksigns_iq2xs\[128\]\s*=\s*\{([^}]+)\};`)
	reGrid := regexp.MustCompile(`iq2xxs_grid\[256\]\s*=\s*\{([^}]+)\};`)

	// 1. Signs table: 128 bytes
	mSigns := reSigns.FindStringSubmatch(cudaExpertSource)
	if mSigns == nil {
		t.Fatal("ds4_cuda_ksigns_iq2xs table literal not found in cudaExpertSource")
	}
	signTokens := strings.Split(mSigns[1], ",")
	signs := make([]byte, 0, 128)
	for _, tok := range signTokens {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		val, err := strconv.ParseUint(tok, 0, 8)
		if err != nil {
			t.Fatalf("invalid sign byte %q: %v", tok, err)
		}
		signs = append(signs, byte(val))
	}
	if len(signs) != 128 {
		t.Fatalf("expected 128 signs entries, got %d", len(signs))
	}
	gotSignsHash := sha256.Sum256(signs)
	if hex.EncodeToString(gotSignsHash[:]) != wantIQ2SignsSHA256 {
		t.Fatalf("CUDA signs digest mismatch: got %s, want %s", hex.EncodeToString(gotSignsHash[:]), wantIQ2SignsSHA256)
	}

	// 2. Grid table: 256 uint64 entries (2048 bytes little-endian)
	mGrid := reGrid.FindStringSubmatch(cudaExpertSource)
	if mGrid == nil {
		t.Fatal("ds4_cuda_iq2xxs_grid table literal not found in cudaExpertSource")
	}
	gridTokens := strings.Split(mGrid[1], ",")
	gridBytes := make([]byte, 0, 256*8)
	buf := make([]byte, 8)
	var count int
	for _, tok := range gridTokens {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		val, err := strconv.ParseUint(tok, 0, 64)
		if err != nil {
			t.Fatalf("invalid grid entry %q: %v", tok, err)
		}
		binary.LittleEndian.PutUint64(buf, val)
		gridBytes = append(gridBytes, buf...)
		count++
	}
	if count != 256 {
		t.Fatalf("expected 256 grid entries, got %d", count)
	}
	gotGridHash := sha256.Sum256(gridBytes)
	if hex.EncodeToString(gotGridHash[:]) != wantIQ2GridSHA256 {
		t.Fatalf("CUDA grid digest mismatch: got %s, want %s", hex.EncodeToString(gotGridHash[:]), wantIQ2GridSHA256)
	}
}

// TestCUDAExpertTableStorageMigrationBitIdentity directly proves Gate (b):
// a reference kernel compiled with __constant__ tables produces Float32bits-identical
// outputs to the production kernel reading __device__ const tables on the production shape.
func TestCUDAExpertTableStorageMigrationBitIdentity(t *testing.T) {
	r := cudaSynthFoundation(t)

	const (
		cols    = 2048
		rows    = 2048
		experts = 256
	)
	ids := []int32{0, 15, 255, 12, 100, 3}
	slots := len(ids)

	// Compile a reference module where the tables are kept in __constant__
	constRefSrc := strings.Replace(cudaMathSource,
		"__device__ const unsigned char ds4_cuda_ksigns_iq2xs",
		"__device__ __constant__ unsigned char ds4_cuda_ksigns_iq2xs", 1)
	constRefSrc = strings.Replace(constRefSrc,
		"__device__ const unsigned long long ds4_cuda_iq2xxs_grid",
		"__device__ __constant__ unsigned long long ds4_cuda_iq2xxs_grid", 1)

	compiled, err := nvrtcx.Compile(constRefSrc,
		nvrtcx.WithName("ds4_math_const_ref.cu"),
		nvrtcx.WithArch(r.major, r.minor),
	)
	if err != nil {
		t.Fatalf("compile const ref: %v", err)
	}
	mod, err := compiled.Load(r.device)
	if err != nil {
		t.Fatalf("load const ref: %v", err)
	}
	defer func() { _ = mod.Unload() }()

	refFn, err := mod.GetFunction("ds4_iq2_xxs_pair_selected_q8_1")
	if err != nil {
		t.Fatalf("get ref function: %v", err)
	}

	dIDs, err := cudamem.Alloc[int32](r.device, len(ids))
	if err != nil {
		t.Fatal(err)
	}
	defer dIDs.Free()
	if err := dIDs.CopyFromHost(ids); err != nil {
		t.Fatal(err)
	}

	gData := cudaExpertFastFixture(16, cols, rows, experts, 101)
	uData := cudaExpertFastFixture(16, cols, rows, experts, 202)
	gTensor := cudaSynthTensor(t, r, "gate.weight", 16, []uint64{uint64(cols), uint64(rows), uint64(experts)}, gData)
	uTensor := cudaSynthTensor(t, r, "up.weight", 16, []uint64{uint64(cols), uint64(rows), uint64(experts)}, uData)

	inFloats := make([]float32, cols)
	for i := range inFloats {
		inFloats[i] = float32(math.Sin(float64(i)*0.41)) * 2.5
	}
	dInFloats := cudaSynthUpload(t, r, inFloats)
	defer dInFloats.Free()

	q81In, err := cudamem.Alloc[byte](r.device, cols/32*36)
	if err != nil {
		t.Fatal(err)
	}
	defer q81In.Free()
	if err := r.encodeQuantQ81(dInFloats, q81In, cols); err != nil {
		t.Fatal(err)
	}

	// Run production kernel (__device__ const)
	gateProd, err := cudamem.Alloc[float32](r.device, slots*rows)
	if err != nil {
		t.Fatal(err)
	}
	defer gateProd.Free()
	upProd, err := cudamem.Alloc[float32](r.device, slots*rows)
	if err != nil {
		t.Fatal(err)
	}
	defer upProd.Free()

	if err := r.encodeMatvecIQ2SelectedQ81(gTensor, uTensor, dIDs, q81In, gateProd, upProd); err != nil {
		t.Fatal(err)
	}

	// Run reference kernel (__constant__)
	gateRef, err := cudamem.Alloc[float32](r.device, slots*rows)
	if err != nil {
		t.Fatal(err)
	}
	defer gateRef.Free()
	upRef, err := cudamem.Alloc[float32](r.device, slots*rows)
	if err != nil {
		t.Fatal(err)
	}
	defer upRef.Free()

	blocks := (rows*slots + cudaExpertWarps - 1) / cudaExpertWarps
	if err := cudalaunch.Kernel(refFn).Grid(uint32(blocks)).Block(32*cudaExpertWarps).On(r.device.DefaultStream()).Launch(
		gTensor.device, uTensor.device, dIDs, q81In, gateRef, upRef,
		uint32(cols), uint32(rows), uint32(slots), uint32(experts)); err != nil {
		t.Fatal(err)
	}

	if err := r.device.DefaultStream().Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}

	gProdHost := make([]float32, slots*rows)
	uProdHost := make([]float32, slots*rows)
	gRefHost := make([]float32, slots*rows)
	uRefHost := make([]float32, slots*rows)

	_ = gateProd.CopyToHost(gProdHost)
	_ = upProd.CopyToHost(uProdHost)
	_ = gateRef.CopyToHost(gRefHost)
	_ = upRef.CopyToHost(uRefHost)

	for i := 0; i < slots*rows; i++ {
		if math.Float32bits(gProdHost[i]) != math.Float32bits(gRefHost[i]) {
			t.Fatalf("gate row %d: __device__ const %v (%#08x) != __constant__ %v (%#08x)",
				i, gProdHost[i], math.Float32bits(gProdHost[i]), gRefHost[i], math.Float32bits(gRefHost[i]))
		}
		if math.Float32bits(uProdHost[i]) != math.Float32bits(uRefHost[i]) {
			t.Fatalf("up row %d: __device__ const %v (%#08x) != __constant__ %v (%#08x)",
				i, uProdHost[i], math.Float32bits(uProdHost[i]), uRefHost[i], math.Float32bits(uRefHost[i]))
		}
	}
}
