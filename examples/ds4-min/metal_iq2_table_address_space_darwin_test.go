//go:build darwin && arm64

package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
	"unsafe"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// TestMetalIQ2TableAddressSpace answers one question: on Apple silicon, does a
// divergent index into a program-scope `constant` table cost what the same
// pattern costs on CUDA?
//
// It matters because the CUDA port found the IQ2 codebook tables in
// __constant__, where a warp's divergent accesses serialize against the
// broadcast unit, and moving them to __device__ const cut expert-ffn 23.03 ->
// 11.98 ms, 48% of the phase. The Metal kernels index the same two tables with
// the same data-dependent indices (metal_matmul_expert_darwin.go:425-427), so
// the structural precondition is present here too.
//
// What is NOT present is the one-line fix. MSL rejects a program-scope variable
// outside the constant address space -- "program scope variable must reside in
// constant address space" -- so the Metal analogue means passing both tables in
// as device buffers and threading them through three kernels. That is real
// surface area to spend speculatively, and Apple's constant address space is not
// CUDA's broadcast constant cache, so the CUDA result does not transfer by
// assumption.
//
// So this test builds the device-buffer variant of the production kernel inside
// the test, from the production source, and runs both arms on identical inputs.
// It asserts bit identity (the arms differ only in where the same bytes live)
// and reports the timing difference without asserting on it. A timing number is
// not a gate; it is the input to the decision about whether to do the port.
func TestMetalIQ2TableAddressSpace(t *testing.T) {
	grid, signs := registeredIQ2Tables(t)

	src, err := deviceTableVariantMSL(metalMatmulExpertPreservedMSL)
	if err != nil {
		t.Fatal(err)
	}

	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Skipf("Metal device: %v", err)
	}
	defer dev.Close()

	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()

	// Arm A: production, tables in the constant address space.
	mm, err := newMetalMatmulPreservedExpert(ops)
	if err != nil {
		t.Fatal(err)
	}
	constantPipe := mm.iq2Pip

	// Arm B: same kernel, tables arriving as device buffers.
	lib, err := dev.CompileLibrary(src)
	if err != nil {
		t.Fatalf("compiling the device-table variant: %v", err)
	}
	// The source carries the kReverseBlocks function constant, so the variant
	// has to be specialized the same way production is, with reverse=false.
	reverse := false
	constants := apple.NewMTLFunctionConstantValues()
	defer objectivec.ObjectFromID(constants.GetID()).Release()
	constants.SetConstantValueTypeAtIndex(unsafe.Pointer(&reverse), apple.MTLDataTypeBool, 0)
	fn, err := lib.NewFunctionWithNameConstantValuesError("matmul_iq2_xxs_pair_selected_f32", constants)
	if err != nil {
		t.Fatalf("device-table variant function: %v", err)
	}
	defer objectivec.ObjectFromID(fn.GetID()).Release()
	devicePipe, err := dev.Raw().NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		t.Fatalf("device-table variant pipeline: %v", err)
	}

	// Production geometry for one decoded token: ffn_gate_exps and ffn_up_exps
	// are [4096 2048 256] and six experts are selected. The expert count is cut
	// to 8 because the kernel only reads the selected rows, so it changes the
	// buffer size and nothing the test measures.
	const (
		cols    = 4096
		rows    = 2048
		experts = 8
		slots   = 6
		tokens  = 1
	)
	blocks := cols / 256
	weightBytes := rows * blocks * 66 * experts

	arena := metalmem.NewArena(dev)
	defer arena.Free()

	rng := uint32(0x12345678)
	next := func() byte {
		rng = rng*1664525 + 1013904223
		return byte(rng >> 24)
	}
	mkWeights := func(name string) boundMetalTensor {
		data := metalmem.AllocIn[byte](arena, name, weightBytes)
		host := data.HostSlice()
		for i := range host {
			host[i] = next()
		}
		return boundMetalTensor{name: name, typeID: 16, dimensions: []uint64{cols, rows, experts}, data: data}
	}
	gate, up := mkWeights("gate"), mkWeights("up")

	in := make([]float32, cols*tokens)
	for i := range in {
		in[i] = float32(int(next())-128) / 64
	}
	ids := []int32{0, 1, 2, 3, 4, 5}

	idsBuf, err := metalmem.AllocFromSlice(dev, ids)
	if err != nil {
		t.Fatal(err)
	}
	defer idsBuf.Free()
	inBuf, err := metalmem.AllocFromSlice(dev, in)
	if err != nil {
		t.Fatal(err)
	}
	defer inBuf.Free()
	gridBuf, err := metalmem.AllocFromSlice(dev, grid)
	if err != nil {
		t.Fatal(err)
	}
	defer gridBuf.Free()
	signsBuf, err := metalmem.AllocFromSlice(dev, signs)
	if err != nil {
		t.Fatal(err)
	}
	defer signsBuf.Free()

	outCount := rows * slots * tokens
	alloc := func() (metalmem.Slice[float32], metalmem.Slice[float32]) {
		g, err := metalmem.Alloc[float32](dev, outCount)
		if err != nil {
			t.Fatal(err)
		}
		u, err := metalmem.Alloc[float32](dev, outCount)
		if err != nil {
			t.Fatal(err)
		}
		return g, u
	}
	gA, uA := alloc()
	defer gA.Free()
	defer uA.Free()
	gB, uB := alloc()
	defer gB.Free()
	defer uB.Free()

	// run dispatches one arm and returns its wall time. withTables selects arm B:
	// the extra buffers are bound only for the variant that declares them.
	run := func(pipe apple.MTLComputePipelineState, gOut, uOut metalmem.Slice[float32], withTables bool) (time.Duration, error) {
		batch, err := newMetalBatch(ops)
		if err != nil {
			return 0, err
		}
		enc := batch.encoder()
		enc.SetComputePipelineState(pipe)
		enc.SetBufferWithOffsetAtIndex(gate.data.Buffer(), gate.data.Offset(), 0)
		enc.SetBufferWithOffsetAtIndex(up.data.Buffer(), up.data.Offset(), 1)
		enc.SetBufferWithOffsetAtIndex(idsBuf.Buffer(), idsBuf.Offset(), 2)
		enc.SetBufferWithOffsetAtIndex(inBuf.Buffer(), inBuf.Offset(), 3)
		enc.SetBufferWithOffsetAtIndex(gOut.Buffer(), gOut.Offset(), 4)
		enc.SetBufferWithOffsetAtIndex(uOut.Buffer(), uOut.Offset(), 5)
		libmetal.SetScalar(enc, uint32(cols), 6)
		libmetal.SetScalar(enc, uint32(rows), 7)
		libmetal.SetScalar(enc, uint32(experts), 8)
		libmetal.SetScalar(enc, uint32(slots), 9)
		libmetal.SetScalar(enc, uint32(tokens), 10)
		if withTables {
			enc.SetBufferWithOffsetAtIndex(gridBuf.Buffer(), gridBuf.Offset(), 11)
			enc.SetBufferWithOffsetAtIndex(signsBuf.Buffer(), signsBuf.Offset(), 12)
		}
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			apple.MTLSize{Width: uint(rows), Height: uint(slots), Depth: uint(tokens)},
			libmetal.Threads(32),
		)
		batch.encoded++
		start := time.Now()
		if err := batch.Wait(context.Background()); err != nil {
			return 0, err
		}
		return time.Since(start), nil
	}

	// Warm both pipelines before timing: the first dispatch of a pipeline pays
	// for its AGX compile, which would otherwise land entirely on whichever arm
	// runs first and read as a difference between the address spaces.
	for i := 0; i < 3; i++ {
		if _, err := run(constantPipe, gA, uA, false); err != nil {
			t.Fatalf("warming the constant arm: %v", err)
		}
		if _, err := run(devicePipe, gB, uB, true); err != nil {
			t.Fatalf("warming the device arm: %v", err)
		}
	}

	// Bit identity. The two arms read the same table bytes from different
	// address spaces and do arithmetic in the same order, so every output must
	// match exactly. This is the gate; the timing below is not.
	gotA, gotU := append([]float32(nil), gA.HostSlice()...), append([]float32(nil), uA.HostSlice()...)
	gotB, gotBU := append([]float32(nil), gB.HostSlice()...), append([]float32(nil), uB.HostSlice()...)
	var mismatches int
	for i := range gotA {
		if math.Float32bits(gotA[i]) != math.Float32bits(gotB[i]) {
			if mismatches < 4 {
				t.Errorf("gate[%d]: constant %v (%#08x) != device %v (%#08x)",
					i, gotA[i], math.Float32bits(gotA[i]), gotB[i], math.Float32bits(gotB[i]))
			}
			mismatches++
		}
		if math.Float32bits(gotU[i]) != math.Float32bits(gotBU[i]) {
			mismatches++
		}
	}
	if mismatches != 0 {
		t.Fatalf("%d of %d outputs differ between the constant and device table arms", mismatches, 2*len(gotA))
	}

	// A non-zero output is the receipt that the dispatch did work at all. Random
	// weight bytes through a real codebook cannot plausibly sum to zero across
	// 12,288 rows, so an all-zero result means the kernel returned early.
	var nonzero int
	for _, v := range gotA {
		if v != 0 {
			nonzero++
		}
	}
	if nonzero == 0 {
		t.Fatal("every gate output is zero: the dispatch produced nothing, so bit identity proved nothing")
	}

	// Interleaved A/B, alternating order per trial, so thermal drift and any
	// monotonic slowdown fall on both arms equally rather than on whichever ran
	// second. Paired deltas per trial, reported as a range: a range that
	// straddles zero is a null result, not a small win.
	// Interleaved A/B, alternating order per trial, so thermal drift and any
	// monotonic slowdown fall on both arms equally rather than on whichever ran
	// second.
	//
	// Reported as a median and a sign test rather than a mean and a range. A
	// single scheduling hiccup on one dispatch moves a 12-trial mean of a
	// sub-millisecond kernel by more than the effect being measured, and the
	// first version of this test did exactly that: means of 0.660 and 1.158 ms
	// with a delta range of [-0.263, +3.113] that straddled zero on the strength
	// of one outlier. How many trials favour each arm does not care about the
	// size of an outlier, only its sign.
	const trials = 51
	var deltas []float64
	var as, bs []float64
	for trial := 0; trial < trials; trial++ {
		var da, db time.Duration
		var err error
		if trial%2 == 0 {
			if da, err = run(constantPipe, gA, uA, false); err != nil {
				t.Fatal(err)
			}
			if db, err = run(devicePipe, gB, uB, true); err != nil {
				t.Fatal(err)
			}
		} else {
			if db, err = run(devicePipe, gB, uB, true); err != nil {
				t.Fatal(err)
			}
			if da, err = run(constantPipe, gA, uA, false); err != nil {
				t.Fatal(err)
			}
		}
		a, b := da.Seconds()*1000, db.Seconds()*1000
		as, bs = append(as, a), append(bs, b)
		deltas = append(deltas, b-a)
	}
	median := func(xs []float64) float64 {
		c := append([]float64(nil), xs...)
		sort.Float64s(c)
		return c[len(c)/2]
	}
	quartiles := func(xs []float64) (float64, float64) {
		c := append([]float64(nil), xs...)
		sort.Float64s(c)
		return c[len(c)/4], c[3*len(c)/4]
	}
	var favourDevice, favourConstant int
	for _, d := range deltas {
		if d < 0 {
			favourDevice++
		} else if d > 0 {
			favourConstant++
		}
	}
	dq1, dq3 := quartiles(deltas)
	t.Logf("IQ2 expert dispatch, cols=%d rows=%d slots=%d, %d matched interleaved trials", cols, rows, slots, trials)
	t.Logf("  constant-address-space tables: median %6.3f ms", median(as))
	t.Logf("  device-buffer tables:          median %6.3f ms", median(bs))
	t.Logf("  paired delta (device - constant): median %+6.3f ms, IQR [%+.3f, %+.3f]", median(deltas), dq1, dq3)
	t.Logf("  sign test: %d/%d trials favour device buffers, %d favour constant", favourDevice, trials, favourConstant)

	// The decision this test exists to inform, stated so a reader does not have
	// to infer it from the numbers. Thresholds are deliberately lopsided: the
	// port costs real surface area across three kernels, so a marginal win does
	// not justify it, while a clear loss settles the question permanently.
	switch {
	case favourDevice >= trials*3/4:
		t.Logf("  VERDICT: the CUDA finding reproduces on Metal. The port is worth its surface area.")
	case favourConstant >= trials*3/4:
		t.Logf("  VERDICT: Apple's constant address space is the right home for these tables.")
		t.Logf("           Do not port the CUDA change. The two backends differ here, and the")
		t.Logf("           Metal kernel is already in the faster configuration.")
	default:
		t.Logf("  VERDICT: null. Neither address space wins consistently, so the port would")
		t.Logf("           spend surface area on three kernels for no measured gain.")
	}
}

// registeredIQ2Tables returns the codebook contents parsed out of the shader
// source, checked against the digests registered in
// metal_expert_table_digest_darwin_test.go. Checking them here makes the tables
// a registered input to the comparison rather than whatever the parse happened
// to produce.
func registeredIQ2Tables(t *testing.T) ([]uint64, []byte) {
	t.Helper()
	parse := func(re *regexp.Regexp, bits int) []uint64 {
		m := re.FindStringSubmatch(metalIQ2CodebooksMSL)
		if m == nil {
			t.Fatalf("table literal not found for %v", re)
		}
		var out []uint64
		for _, tok := range strings.Split(m[1], ",") {
			if tok = strings.TrimSpace(tok); tok == "" {
				continue
			}
			v, err := strconv.ParseUint(tok, 0, bits)
			if err != nil {
				t.Fatalf("invalid table entry %q: %v", tok, err)
			}
			out = append(out, v)
		}
		return out
	}
	gridVals := parse(regexp.MustCompile(`iq2xxs_grid\[256\]\s*=\s*\{([^}]+)\};`), 64)
	signVals := parse(regexp.MustCompile(`ksigns_iq2xs\[128\]\s*=\s*\{([^}]+)\};`), 8)
	if len(gridVals) != 256 || len(signVals) != 128 {
		t.Fatalf("parsed %d grid and %d sign entries, want 256 and 128", len(gridVals), len(signVals))
	}
	signs := make([]byte, len(signVals))
	for i, v := range signVals {
		signs[i] = byte(v)
	}
	gridBytes := make([]byte, 8*len(gridVals))
	for i, v := range gridVals {
		binary.LittleEndian.PutUint64(gridBytes[8*i:], v)
	}
	if got := hex.EncodeToString(sha256Sum(gridBytes)); got != wantIQ2GridSHA256 {
		t.Fatalf("grid digest %s != registered %s", got, wantIQ2GridSHA256)
	}
	if got := hex.EncodeToString(sha256Sum(signs)); got != wantIQ2SignsSHA256 {
		t.Fatalf("signs digest %s != registered %s", got, wantIQ2SignsSHA256)
	}
	return gridVals, signs
}

func sha256Sum(b []byte) []byte {
	s := sha256.Sum256(b)
	return s[:]
}

// deviceTableVariantMSL rewrites the production shader so the IQ2 kernel reads
// its codebooks from device buffers instead of the program-scope constant
// arrays. Every substitution is counted, because a rewrite that silently
// matched nothing would leave both arms running identical code and the
// comparison would report a null result that means nothing.
func deviceTableVariantMSL(src string) (string, error) {
	subs := []struct {
		old, new string
		want     int
	}{
		{
			old:  "constant uint &tokens [[buffer(10)]],",
			new:  "constant uint &tokens [[buffer(10)]],\n    device const ulong *ds4_dev_iq2xxs_grid [[buffer(11)]],\n    device const uchar *ds4_dev_ksigns_iq2xs [[buffer(12)]],",
			want: 1,
		},
		{
			old:  "constant const uchar *gg = reinterpret_cast<constant const uchar *>(ds4_metal_iq2xxs_grid + ag[l]);",
			new:  "device const uchar *gg = reinterpret_cast<device const uchar *>(ds4_dev_iq2xxs_grid + ag[l]);",
			want: 1,
		},
		{
			old:  "constant const uchar *gu = reinterpret_cast<constant const uchar *>(ds4_metal_iq2xxs_grid + au[l]);",
			new:  "device const uchar *gu = reinterpret_cast<device const uchar *>(ds4_dev_iq2xxs_grid + au[l]);",
			want: 1,
		},
		{
			old:  "uchar signg = ds4_metal_ksigns_iq2xs[(sg >> (7 * l)) & 127], signu = ds4_metal_ksigns_iq2xs[(su >> (7 * l)) & 127];",
			new:  "uchar signg = ds4_dev_ksigns_iq2xs[(sg >> (7 * l)) & 127], signu = ds4_dev_ksigns_iq2xs[(su >> (7 * l)) & 127];",
			want: 1,
		},
	}
	out := src
	for _, s := range subs {
		if got := strings.Count(out, s.old); got != s.want {
			return "", fmt.Errorf("rewriting the shader: found %d occurrences of %q, want %d -- the production source moved and this variant would not be the change it claims to be", got, s.old, s.want)
		}
		out = strings.Replace(out, s.old, s.new, s.want)
	}
	if strings.Contains(out[strings.Index(out, "kernel void matmul_iq2_xxs_pair_selected_f32"):], "ds4_metal_iq2xxs_grid") {
		return "", fmt.Errorf("rewriting the shader: the IQ2 kernel still references the constant-space grid")
	}
	return out, nil
}
