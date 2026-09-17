//go:build linux

package main

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaMoERouting is one adversarial expert assignment. Random selections at a
// small group width essentially never collide, so a random routing would leave
// the gather kernels' reuse path untaken and the comparison vacuous. Every
// case here fixes the collision structure on purpose.
type cudaMoERouting struct {
	name    string
	tokens  int
	slots   int
	experts int
	ids     []int32
}

// cudaMoERoutings covers, in order: one expert taking every pair; one expert
// taking exactly one pair per token, so a tile spans distinct activations;
// every pair on its own expert, so the tile loop is degenerate; a skew with
// one heavy expert; tile boundary counts; ids the per-token kernels reject;
// a routing that leaves every bin but the last empty; and full-width groups
// with dense and sparse expert coverage.
func cudaMoERoutings() []cudaMoERouting {
	const slots = 6
	single := make([]int32, 5*slots)
	for i := range single {
		single[i] = 3
	}
	perToken := make([]int32, 9*slots)
	for token := range 9 {
		for slot := range slots {
			if slot == 0 {
				perToken[token*slots+slot] = 3 // bin 3: nine pairs, nine distinct tokens
				continue
			}
			perToken[token*slots+slot] = int32((token+slot)%5 + 4)
		}
	}
	distinct := make([]int32, 2*slots)
	for i := range distinct {
		distinct[i] = int32(i)
	}
	skew := make([]int32, 5*slots)
	for i := range skew {
		if i < 5 {
			skew[i] = int32(i + 1)
			continue
		}
		skew[i] = 0
	}
	var boundary []int32
	for bin, count := range []int{cudaMoEGatherTile - 1, cudaMoEGatherTile, cudaMoEGatherTile + 1, 2*cudaMoEGatherTile + 1} {
		for range count {
			boundary = append(boundary, int32(bin))
		}
	}
	// Keep the boundary bins intact while padding to whole tokens.
	for len(boundary)%slots != 0 {
		boundary = append(boundary, 8)
	}
	invalid := make([]int32, 3*slots)
	for i := range invalid {
		switch i % 5 {
		case 0:
			invalid[i] = -1
		case 1:
			invalid[i] = 9 // == experts
		case 2:
			invalid[i] = 1000
		default:
			invalid[i] = int32(i % 9)
		}
	}
	last := make([]int32, 2*slots)
	for i := range last {
		last[i] = 8
	}
	// At TILE=8, these counts maximize sum ceil(count/TILE) for 1536
	// pairs over 256 experts: 160*2 + 96 = 416 weight-row tiles.
	// Interleave the heavy experts so each token has distinct selections.
	var wide []int32
	for range 9 {
		for expert := range 160 {
			wide = append(wide, int32(expert))
		}
	}
	for expert := 160; expert < 256; expert++ {
		wide = append(wide, int32(expert))
	}
	sparse := make([]int32, 256*slots)
	for i := range sparse {
		sparse[i] = int32(250 + i%slots)
	}
	return []cudaMoERouting{
		{"single-expert", 5, slots, 9, single},
		{"one-pair-per-token", 9, slots, 9, perToken},
		{"all-distinct", 2, slots, 16, distinct},
		{"skew", 5, slots, 9, skew},
		{"tile-boundary", len(boundary) / slots, slots, 9, boundary},
		{"invalid-ids", 3, slots, 9, invalid},
		{"last-bin-only", 2, slots, 9, last},
		{"full-width-tile8-worst-case", 256, slots, 256, wide},
		{"full-width-sparse", 256, slots, 256, sparse},
	}
}

// TestCUDAMoEGatherConstants checks the two compile-time constants the Go side
// mirrors. A drift between them silently changes the tile the kernel keeps in
// registers while the Go comment and the tests keep claiming the old value.
func TestCUDAMoEGatherConstants(t *testing.T) {
	for _, c := range []struct {
		define string
		want   int
	}{
		{"DS4_MOE_IQ2_GATHER_TILE", cudaMoEIQ2GatherTile},
		{"DS4_MOE_Q2_GATHER_TILE", cudaMoEQ2GatherTile},
		{"DS4_MOE_GATHER_TILE", cudaMoEGatherTile},
		{"DS4_MOE_MAX_EXPERTS", cudaMoEMaxExperts},
	} {
		want := fmt.Sprintf("#define %s %d", c.define, c.want)
		if !strings.Contains(cudaExpertGatherSource, want) {
			t.Errorf("cuda_experts_gather.cu does not contain %q", want)
		}
	}
	for _, kernel := range []string{"ds4_moe_group_csr", "ds4_iq2_xxs_pair_selected_q8_1_gather", "ds4_q2_k_selected_q8_1_gather"} {
		if !strings.Contains(cudaMathSource, kernel) {
			t.Errorf("cudaMathSource does not contain %q", kernel)
		}
	}
}

// cudaMoEGroupReference is an independent counting sort. It shares no code
// with ds4_moe_group_csr, which scans per bin rather than scattering per pair.
func cudaMoEGroupReference(ids []int32, experts int) (offsets, pairs, active []int32) {
	bin := func(id int32) int {
		if id < 0 || int(id) >= experts {
			return experts
		}
		return int(id)
	}
	bins := experts + 1
	offsets = make([]int32, bins+1)
	for _, id := range ids {
		offsets[bin(id)+1]++
	}
	for b := range bins {
		if offsets[b+1] > 0 {
			active = append(active, int32(b))
		}
		offsets[b+1] += offsets[b]
	}
	pairs = make([]int32, len(ids))
	next := append([]int32(nil), offsets[:bins]...)
	for p, id := range ids {
		b := bin(id)
		pairs[next[b]] = int32(p)
		next[b]++
	}
	return offsets, pairs, active
}

func cudaMoEUploadIDs(t *testing.T, r *cudaFoundation, ids []int32) cudamem.Slice[int32] {
	t.Helper()
	d, err := cudamem.Alloc[int32](r.device, len(ids))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := d.Free(); err != nil {
			t.Error(err)
		}
	})
	if err := d.CopyFromHost(ids); err != nil {
		t.Fatal(err)
	}
	return d
}

func cudaMoEGroup(t *testing.T, r *cudaFoundation, ids []int32, experts int) *cudaMoEGrouping {
	t.Helper()
	g, err := newCUDAMoEGrouping(r.device, experts, len(ids))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := g.Free(); err != nil {
			t.Error(err)
		}
	})
	d := cudaMoEUploadIDs(t, r, ids)
	cudaQualificationMeasure(t, r, func() error { return r.encodeMoEGroupCSR(d, g) })
	return g
}

func cudaMoEReadInts(t *testing.T, s cudamem.Slice[int32]) []int32 {
	t.Helper()
	v := make([]int32, s.Len())
	if err := s.CopyToHost(v); err != nil {
		t.Fatal(err)
	}
	return v
}

// TestCUDAMoEGroupCSR checks the grouping kernel against an independent host
// counting sort on every adversarial routing, then confirms the comparison can
// fail by regrouping a rotated routing.
func TestCUDAMoEGroupCSR(t *testing.T) {
	r := cudaSynthFoundation(t)
	for _, rc := range cudaMoERoutings() {
		t.Run(rc.name, func(t *testing.T) {
			g := cudaMoEGroup(t, r, rc.ids, rc.experts)
			offsets, pairs, active := cudaMoEGroupReference(rc.ids, rc.experts)
			gotOffsets, gotPairs := cudaMoEReadInts(t, g.offsets), cudaMoEReadInts(t, g.pairs)
			gotCount := cudaMoEReadInts(t, g.count)[0]
			gotActive := cudaMoEReadInts(t, g.active)[:gotCount]
			if !cudaIntsEqual(gotOffsets, offsets) {
				t.Fatalf("offsets: got %v want %v", gotOffsets, offsets)
			}
			if !cudaIntsEqual(gotPairs, pairs) {
				t.Fatalf("pairs: got %v want %v", gotPairs, pairs)
			}
			if int(gotCount) != len(active) || !cudaIntsEqual(gotActive, active) {
				t.Fatalf("active: got %v want %v", gotActive, active)
			}
			t.Logf("%s: %d pairs over %d non-empty bins, device offsets %v", rc.name, len(rc.ids), gotCount, gotOffsets)
		})
	}
	// Control: the same reference against a rotated routing must disagree, so a
	// grouping kernel that ignored its ids could not pass the loop above.
	rc := cudaMoERoutings()[4]
	rotated := append(append([]int32(nil), rc.ids[1:]...), rc.ids[0])
	g := cudaMoEGroup(t, r, rotated, rc.experts)
	_, pairs, _ := cudaMoEGroupReference(rc.ids, rc.experts)
	if cudaIntsEqual(cudaMoEReadInts(t, g.pairs), pairs) {
		t.Fatal("rotated routing produced the unrotated listing; the comparison cannot fail")
	}
}

func cudaIntsEqual(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// cudaMoEBitsEqual compares float32 results by encoding, not by value, so an
// unwritten poison pattern cannot pass as a legitimate NaN and so the
// comparison admits no tolerance at all.
func cudaMoEBitsEqual(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d outputs want %d", name, len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: output %d is %#08x (%g), per-pair kernel produced %#08x (%g)",
				name, i, math.Float32bits(got[i]), got[i], math.Float32bits(want[i]), want[i])
		}
	}
}

// cudaMoEFill writes a distinguishable non-signaling pattern over a result
// buffer. It is not the invalid-expert poison, so an output the gather kernel
// never wrote cannot be mistaken for one it deliberately poisoned.
func cudaMoEFill(t *testing.T, s cudamem.Slice[float32]) {
	t.Helper()
	v := make([]float32, s.Len())
	unwritten := math.Float32frombits(0x7ff00001)
	for i := range v {
		v[i] = unwritten
	}
	if err := s.CopyFromHost(v); err != nil {
		t.Fatal(err)
	}
}

// TestCUDAExpertsQ81GatherDifferential runs the expert-major gather kernels
// and the per-pair kernels over the same weights, activations and routing and
// requires exact float32 equality. The per-pair kernels are launched once per
// token, which is what the unbatched prefill path does today.
func TestCUDAExpertsQ81GatherDifferential(t *testing.T) {
	r := cudaSynthFoundation(t)
	for _, cols := range []int{256, 1024} {
		for _, rc := range cudaMoERoutings() {
			t.Run(fmt.Sprintf("%d/%s", cols, rc.name), func(t *testing.T) {
				got, want := cudaMoEGatherResults(t, r, cols, rc, rc.ids)
				for i, name := range []string{"gate", "up", "down"} {
					cudaMoEBitsEqual(t, name, got[i], want[i])
				}
			})
		}
	}
	// Control: driving the gather kernels from a rotated routing while the
	// per-pair kernels keep the original must produce a mismatch. Without it,
	// exact equality above could mean neither side read its ids.
	rc := cudaMoERoutings()[3]
	rotated := append(append([]int32(nil), rc.ids[1:]...), rc.ids[0])
	got, want := cudaMoEGatherResults(t, r, 256, rc, rotated)
	differed := false
	for i := range got {
		for j := range got[i] {
			if math.Float32bits(got[i][j]) != math.Float32bits(want[i][j]) {
				differed = true
			}
		}
	}
	if !differed {
		t.Fatal("a rotated routing matched the per-pair kernels; the differential cannot fail")
	}
	t.Log("control: rotated routing mismatched, so the differential is not vacuous")
}

// cudaMoEGatherResults runs one routing both ways and returns the gather and
// per-pair gate, up and down results. gatherIDs is normally rc.ids; the caller
// passes a different assignment only to prove the comparison can fail.
func cudaMoEGatherResults(t *testing.T, r *cudaFoundation, cols int, rc cudaMoERouting, gatherIDs []int32) (got, want [3][]float32) {
	t.Helper()
	const rows = 17
	pairs, tokens, slots := len(rc.ids), rc.tokens, rc.slots
	if pairs != tokens*slots {
		t.Fatalf("%s: %d ids is not %d tokens of %d slots", rc.name, pairs, tokens, slots)
	}

	// IQ2 reads one activation row per token; Q2 reads one per pair.
	tokenX, pairX := make([]float32, tokens*cols), make([]float32, pairs*cols)
	for i := range tokenX {
		tokenX[i] = float32((i*37+i/17)%257-128) / 128
	}
	for i := range pairX {
		pairX[i] = float32((i*13+i/cols*31+i/19)%251-125) / 128
	}
	q, sq := cudaExpertQ81Upload(t, r, tokenX, cols), cudaExpertQ81Upload(t, r, pairX, cols)

	dims := []uint64{uint64(cols), rows, uint64(rc.experts)}
	gate := cudaSynthTensor(t, r, "gather IQ2 gate", 16, dims, cudaExpertFixture(16, cols, rows, rc.experts, 17))
	up := cudaSynthTensor(t, r, "gather IQ2 up", 16, dims, cudaExpertFixture(16, cols, rows, rc.experts, 29))
	down := cudaSynthTensor(t, r, "gather Q2 down", 10, dims, cudaExpertFixture(10, cols, rows, rc.experts, 43))

	// Per-pair reference: one launch per token, exactly as prefill runs today.
	refGate, refUp, refDown := cudaTestAlloc(t, r, pairs*rows), cudaTestAlloc(t, r, pairs*rows), cudaTestAlloc(t, r, pairs*rows)
	for _, out := range []cudamem.Slice[float32]{refGate, refUp, refDown} {
		cudaMoEFill(t, out)
	}
	ids := cudaMoEUploadIDs(t, r, rc.ids)
	rowBytes := cols / 32 * 36
	for token := range tokens {
		tokenIDs := ids.Slice(token*slots, (token+1)*slots)
		outLo, outHi := token*slots*rows, (token+1)*slots*rows
		inLo, sLo := token*rowBytes, token*slots*rowBytes
		cudaQualificationMeasure(t, r, func() error {
			return r.encodeMatvecIQ2SelectedQ81(gate, up, tokenIDs, q.Slice(inLo, inLo+rowBytes),
				refGate.Slice(outLo, outHi), refUp.Slice(outLo, outHi))
		})
		cudaQualificationMeasure(t, r, func() error {
			return r.encodeMatvecQ2KSelectedQ81(down, tokenIDs, sq.Slice(sLo, sLo+slots*rowBytes), refDown.Slice(outLo, outHi))
		})
	}

	// Expert-major gather: one launch for the whole group.
	g := cudaMoEGroup(t, r, gatherIDs, rc.experts)
	gotGate, gotUp, gotDown := cudaTestAlloc(t, r, pairs*rows), cudaTestAlloc(t, r, pairs*rows), cudaTestAlloc(t, r, pairs*rows)
	for _, out := range []cudamem.Slice[float32]{gotGate, gotUp, gotDown} {
		cudaMoEFill(t, out)
	}
	cudaQualificationMeasure(t, r, func() error {
		return r.encodeMatvecIQ2SelectedQ81Gather(gate, up, g, slots, q, gotGate, gotUp)
	})
	cudaQualificationMeasure(t, r, func() error {
		return r.encodeMatvecQ2KSelectedQ81Gather(down, g, sq, gotDown)
	})

	t.Logf("%s cols=%d: %d pairs, %d tokens, %d non-empty bins, %d per-pair launches replaced by 1 gather launch",
		rc.name, cols, pairs, tokens, cudaMoEReadInts(t, g.count)[0], tokens)
	got = [3][]float32{cudaTestRead(t, gotGate), cudaTestRead(t, gotUp), cudaTestRead(t, gotDown)}
	want = [3][]float32{cudaTestRead(t, refGate), cudaTestRead(t, refUp), cudaTestRead(t, refDown)}
	return got, want
}

// TestCUDAMoEGatherShapeRejection checks that every shape the kernels cannot
// honour is refused before a launch rather than read past its buffer.
func TestCUDAMoEGatherShapeRejection(t *testing.T) {
	r := cudaSynthFoundation(t)
	const cols, rows, experts, slots, tokens = 256, 17, 9, 6, 3
	pairs := tokens * slots
	dims := []uint64{cols, rows, experts}
	gate := cudaSynthTensor(t, r, "reject IQ2 gate", 16, dims, cudaExpertFixture(16, cols, rows, experts, 17))
	up := cudaSynthTensor(t, r, "reject IQ2 up", 16, dims, cudaExpertFixture(16, cols, rows, experts, 29))
	down := cudaSynthTensor(t, r, "reject Q2 down", 10, dims, cudaExpertFixture(10, cols, rows, experts, 43))
	in := cudaExpertAllocBytes(t, r, cols/32*36*tokens)
	sin := cudaExpertAllocBytes(t, r, cols/32*36*pairs)
	gout, uout, dout := cudaTestAlloc(t, r, pairs*rows), cudaTestAlloc(t, r, pairs*rows), cudaTestAlloc(t, r, pairs*rows)
	g := cudaMoEGroup(t, r, make([]int32, pairs), experts)
	wide, err := newCUDAMoEGrouping(r.device, experts+1, pairs)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := wide.Free(); err != nil {
			t.Error(err)
		}
	})
	for label, err := range map[string]error{
		"IQ2 short input":     r.encodeMatvecIQ2SelectedQ81Gather(gate, up, g, slots, in.Slice(0, in.Len()-36), gout, uout),
		"IQ2 short output":    r.encodeMatvecIQ2SelectedQ81Gather(gate, up, g, slots, in, gout.Slice(0, gout.Len()-1), uout),
		"IQ2 slots":           r.encodeMatvecIQ2SelectedQ81Gather(gate, up, g, 0, in, gout, uout),
		"IQ2 ragged slots":    r.encodeMatvecIQ2SelectedQ81Gather(gate, up, g, 5, in, gout, uout),
		"IQ2 expert mismatch": r.encodeMatvecIQ2SelectedQ81Gather(gate, up, wide, slots, in, gout, uout),
		"IQ2 no grouping":     r.encodeMatvecIQ2SelectedQ81Gather(gate, up, nil, slots, in, gout, uout),
		"Q2 short input":      r.encodeMatvecQ2KSelectedQ81Gather(down, g, sin.Slice(0, sin.Len()-36), dout),
		"Q2 short output":     r.encodeMatvecQ2KSelectedQ81Gather(down, g, sin, dout.Slice(0, dout.Len()-1)),
		"Q2 expert mismatch":  r.encodeMatvecQ2KSelectedQ81Gather(down, wide, sin, dout),
		"Q2 no grouping":      r.encodeMatvecQ2KSelectedQ81Gather(down, nil, sin, dout),
	} {
		if err == nil {
			t.Errorf("accepted %s", label)
		}
	}
	if _, err := newCUDAMoEGrouping(r.device, cudaMoEMaxExperts+1, pairs); err == nil {
		t.Error("accepted a grouping wider than the counting sort's bins")
	}
}

// TestCUDAMoEGatherRegisters reports device register counts and local memory usage
// for the MoE gather kernels.
func TestCUDAMoEGatherRegisters(t *testing.T) {
	r := cudaSynthFoundation(t)
	if err := r.device.WithContext(r.loadMathKernels); err != nil {
		t.Fatal(err)
	}
	for _, k := range []struct {
		name string
		fn   driver.Function
	}{
		{"ds4_q2_k_selected_q8_1_gather", r.q2KGatherQ81Fn},
		{"ds4_iq2_xxs_pair_selected_q8_1_gather", r.iq2GatherQ81Fn},
	} {
		regs, err := k.fn.GetAttribute(driver.FuncAttributeNumRegs)
		if err != nil {
			t.Fatal(err)
		}
		localMem, err := k.fn.GetAttribute(driver.FuncAttributeLocalSizeBytes)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("kernel %s: %d registers, %d bytes local memory", k.name, regs, localMem)
	}
}
