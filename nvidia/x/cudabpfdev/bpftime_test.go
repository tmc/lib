package cudabpfdev

import (
	"context"
	"errors"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaptx"
)

// Reference corpus: eunomia-bpf/bpftime GPU parts @edcc8ec.
//
// TestBpftimeCudaCounterTransliterationRejected checks unsafe probe behavior
// from example/gpu/cuda-counter/cuda_probe.bpf.c at bpftime commit edcc8ec.
func TestBpftimeCudaCounterTransliterationRejected(t *testing.T) {
	b := budget()
	const wantReason = "warp-uniformity violation: lane-varying map value"
	cases := []struct {
		name string
		p    Program
	}{
		{
			name: "probe stores a per-thread timestamp at the uniform block key",
			p: Program{Version: 1, Registers: 2, Slots: 16, Code: []Instruction{
				{Op: Const, Dst: 0, Imm: 0}, // blockIdx.x is uniform within the block; use key 0 as its surrogate.
				{Op: Lane, Dst: 1},          // Lane stands in for the per-thread global timer result.
				{Op: MapAdd, A: 0, B: 1},    // Version 1 has no MapUpdate overwrite operation.
				{Op: Exit},
			}},
		},
		{
			name: "retprobe adds a per-thread delta to a uniform total key",
			p: Program{Version: 1, Registers: 2, Slots: 16, Code: []Instruction{
				{Op: Const, Dst: 0, Imm: 0}, // process total key is uniform; use key 0 as its surrogate.
				{Op: Lane, Dst: 1},          // Lane stands in for each thread's timestamp delta.
				{Op: MapAdd, A: 0, B: 1},
				{Op: Exit},
			}},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Verify(tc.p, b)
			var located *VerifyError
			if !errors.As(err, &located) {
				t.Fatalf("Verify() error = %T %v, want located VerifyError", err, err)
			}
			if located.PC != 2 || located.Reason != wantReason {
				t.Fatalf("VerifyError = {PC:%d Reason:%q}, want {PC:2 Reason:%q}", located.PC, located.Reason, wantReason)
			}
		})
	}

	// The source's lookup by blockIdx.x followed by delete by pid is not
	// representable: bytecode v1 has no lookup or delete instruction.
}

// TestBpftime_PerThreadVersusPerWarp demonstrates that thread counting requires explicit
// Ballot population count aggregation (__ballot_sync/popc) and produces different results
// from plain per-warp increments when running on a partially active warp.
// Reference: example/gpu/threadhist/threadhist.bpf.c (@edcc8ec).
func TestThreadCountDiffersFromWarpCountOnPartialMask(t *testing.T) {
	ctx := context.Background()
	b := Budget{MaxInstructions: 64, MaxSteps: 2000, MaxRegisters: 64, MaxSharedBytes: 1024, MaxSlots: 16}

	// Program A: Plain per-warp increment.
	// Transliteration using plain MapAdd counts warps (once per active warp), not threads.
	pWarp := Program{
		Version:   1,
		Registers: 3,
		Slots:     1,
		Code: []Instruction{
			{Op: Const, Dst: 0, Imm: 0}, // r0 = map slot 0
			{Op: Const, Dst: 1, Imm: 1}, // r1 = 1
			{Op: MapAdd, A: 0, B: 1},    // map[0] += 1 (once per active warp)
			{Op: Exit},
		},
	}

	// Program B: Thread counting using explicit Ballot population count aggregation.
	// Uses Ballot to collect the active thread mask, and a bounded popcount loop to sum
	// the set bits into a uniform register before updating the map.
	pThread := Program{
		Version:   1,
		Registers: 7,
		Slots:     1,
		Code: []Instruction{
			{Op: Const, Dst: 0, Imm: 0},        // pc 0: r0 = map slot 0
			{Op: Const, Dst: 1, Imm: 1},        // pc 1: r1 = 1
			{Op: Ballot, Dst: 2, A: 1},         // pc 2: r2 = active mask of lanes with r1 != 0
			{Op: Const, Dst: 4, Imm: 1},        // pc 3: r4 = bit mask (starts at 1)
			{Op: Const, Dst: 6, Imm: 1},        // pc 4: r6 = 1 (constant for unconditional skip)
			{Op: And, Dst: 5, A: 2, B: 4},      // pc 5: loop start: r5 = r2 & r4
			{Op: JumpNonzero, A: 5, Target: 8}, // pc 6: if bit is set, jump to pc 8 (add 1)
			{Op: JumpNonzero, A: 6, Target: 9}, // pc 7: else jump to pc 9 (skip add)
			{Op: Add, Dst: 3, A: 3, B: 1},      // pc 8: r3 += 1 (accumulate popcount)
			{Op: Add, Dst: 4, A: 4, B: 4},      // pc 9: r4 <<= 1 (double mask)
			{Op: Repeat, Target: 5, Imm: 32},   // pc 10: repeat 32 times
			{Op: MapAdd, A: 0, B: 3},           // pc 11: map[0] += r3 (total active threads)
			{Op: Exit},
		},
	}

	// Verify both programs.
	if _, err := Verify(pWarp, b); err != nil {
		t.Fatalf("Verify(pWarp) failed: %v", err)
	}
	if _, err := Verify(pThread, b); err != nil {
		t.Fatalf("Verify(pThread) failed: %v", err)
	}

	// Partially active warps: test various divergence / partial activation patterns.
	cases := []struct {
		name          string
		activeMask    uint32
		wantWarpMap   uint32
		wantThreadMap uint32
	}{
		{name: "single-lane", activeMask: 0x00000001, wantWarpMap: 1, wantThreadMap: 1},
		{name: "four-lanes", activeMask: 0x0000000f, wantWarpMap: 1, wantThreadMap: 4},
		{name: "five-lanes", activeMask: 0x0000001f, wantWarpMap: 1, wantThreadMap: 5},
		{name: "alternating", activeMask: 0x55555555, wantWarpMap: 1, wantThreadMap: 16},
		{name: "full-warp", activeMask: 0xffffffff, wantWarpMap: 1, wantThreadMap: 32},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resWarp, err := Evaluate(ctx, pWarp, b, tc.activeMask)
			if err != nil {
				t.Fatalf("Evaluate(pWarp) failed: %v", err)
			}
			resThread, err := Evaluate(ctx, pThread, b, tc.activeMask)
			if err != nil {
				t.Fatalf("Evaluate(pThread) failed: %v", err)
			}

			if resWarp.Map[0] != tc.wantWarpMap {
				t.Errorf("warp count = %d, want %d", resWarp.Map[0], tc.wantWarpMap)
			}
			if resThread.Map[0] != tc.wantThreadMap {
				t.Errorf("thread count = %d, want %d", resThread.Map[0], tc.wantThreadMap)
			}

			// On partially active warps with >1 lane, warp counting and thread counting differ.
			if tc.wantThreadMap > 1 && resWarp.Map[0] == resThread.Map[0] {
				t.Errorf("partially active warp (mask=0x%x): warp count (%d) unexpectedly matched thread count (%d)",
					tc.activeMask, resWarp.Map[0], resThread.Map[0])
			}
		})
	}
}

const stubPTX = `.version 9.0
.target sm_121a
.address_size 64

.func __bpftime_cuda__kernel_trace
{
    ret;
}

.visible .entry affine(
    .param .u64 out
)
{
    .reg .b64 %addr;
    .reg .b32 %value;
    call.uni __bpftime_cuda__kernel_trace;
    ld.param.u64 %addr, [out];
    mov.u32 %value, 7;
    st.global.u32 [%addr], %value;
    ret;
}
`

func emittedBody(t *testing.T, text, start string) string {
	t.Helper()
	i := strings.Index(text, start)
	if i < 0 {
		t.Fatalf("body start %q not found", start)
	}
	i += len(start)
	j := strings.Index(text[i:], "\n}\n")
	if j < 0 {
		t.Fatalf("body close not found after %q", start)
	}
	return text[i : i+j]
}

func checkEmittedProgramBody(t *testing.T, body string, p Program, prefix string, wantReturn bool) {
	t.Helper()
	if !strings.Contains(body, ".reg .b32 "+prefix+"r<"+strconv.Itoa(int(p.Registers))+">;") {
		t.Fatalf("missing program register declaration for %d registers:\n%s", p.Registers, body)
	}
	for r := uint8(0); r < p.Registers; r++ {
		want := "mov.u32 " + prefix + "r" + strconv.Itoa(int(r)) + ", 0;"
		if !strings.Contains(body, want) {
			t.Errorf("missing zero initialization %q", want)
		}
	}
	for pc, in := range p.Code {
		if in.Op == Repeat {
			want := "mov.u32 " + prefix + "loop" + strconv.Itoa(pc) + ", 0;"
			if !strings.Contains(body, want) {
				t.Errorf("missing loop counter initialization %q", want)
			}
		}
	}
	for _, want := range []string{
		"activemask.b32 " + prefix + "actual_mask;",
		"vote.sync.ballot.b32 " + prefix + "mask, " + prefix + "pred, " + prefix + "actual_mask;",
		"bfind.u32 " + prefix + "leader, " + prefix + "reverse_mask;",
		"setp.eq.u32 " + prefix + "is_leader, " + prefix + "lane, " + prefix + "leader;",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("missing warp leader structure %q", want)
		}
	}
	mapAdds := 0
	for _, in := range p.Code {
		if in.Op == MapAdd {
			mapAdds++
		}
	}
	if got := strings.Count(body, "atom.global.add.u32"); got != mapAdds {
		t.Errorf("u32 map atom count = %d, want %d from program", got, mapAdds)
	}
	if got := strings.Count(body, "atom.global.add.u64"); got != mapAdds {
		t.Errorf("u64 miss atom count = %d, want %d from program", got, mapAdds)
	}
	if got := strings.Count(body, "and.pred "+prefix+"write_map, "+prefix+"is_leader"); got != 2*mapAdds {
		t.Errorf("leader-predicate count = %d, want %d from program", got, 2*mapAdds)
	}
	if mapAdds > 0 && !strings.Contains(body, "@"+prefix+"write_map atom.global.add.u32") {
		t.Error("MapAdd is not guarded by the warp-leader predicate")
	}
	if strings.HasSuffix(strings.TrimSpace(body), "ret;") != wantReturn {
		t.Errorf("body return terminator = %t, want %t", strings.HasSuffix(strings.TrimSpace(body), "ret;"), wantReturn)
	}
}

// TestStubCallSiteLocated covers opt-in stub attachment inspired by
// example/gpu/kernel_trace at bpftime commit edcc8ec.
func TestStubCallSiteLocated(t *testing.T) {
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "out", Type: "u64", IsPointer: true})
	if err != nil {
		t.Fatal(err)
	}
	newSource := func(text string) (*ArtifactSource, *cudakernel.Artifact) {
		t.Helper()
		a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(text), "affine", schema, "sm_121a")
		if err != nil {
			t.Fatal(err)
		}
		s, err := NewArtifactSource(a, Geometry{Grid: [3]uint32{1, 1, 1}, Block: [3]uint32{32, 1, 1}})
		if err != nil {
			t.Fatal(err)
		}
		return s, a
	}
	source, artifact := newSource(stubPTX)
	p := program(Instruction{Op: Const, Dst: 1, Imm: 1}, Instruction{Op: MapAdd, B: 1}, Instruction{Op: Exit})
	b := budget()
	b.MaxRegisters = 128

	// 1. Opt-in stub-based attach policy.
	policy := InjectionPolicy{
		Enabled:       true,
		Allowlist:     []KernelIdentity{{artifact.Digest, artifact.EntryPoint, artifact.Target}},
		StubFunctions: []string{"__bpftime_cuda__kernel_trace"},
	}

	inj, err := PrepareInjection(source, p, b, policy)
	if err != nil {
		t.Fatalf("PrepareInjection failed: %v", err)
	}
	candidate, reason := inj.Candidate()
	if candidate == nil {
		t.Fatalf("candidate preparation failed: %s", reason)
	}

	candPTX := string(candidate.Raw)
	checkEmittedProgramBody(t, emittedBody(t, emitTrampoline(p, 0), "{\n"), p, "__cudabpfdev_site0_", false)
	checkEmittedProgramBody(t, emittedBody(t, emitProbeFunction(p, "__cudabpfdev_probe_0"), ".func __cudabpfdev_probe_0\n{\n"), p, "__cudabpfdev_probe_0_", true)

	// Verify the stub call was rewritten to point to the injected probe function.
	if !strings.Contains(candPTX, "call.uni __cudabpfdev_probe_0;") {
		t.Fatalf("expected rewritten stub call to __cudabpfdev_probe_0, got:\n%s", candPTX)
	}
	if strings.Contains(candPTX, "call.uni __bpftime_cuda__kernel_trace;") {
		t.Fatal("candidate still calls original stub __bpftime_cuda__kernel_trace")
	}

	// Verify the probe function was emitted and the original empty stub definition remains.
	if !strings.Contains(candPTX, ".func __cudabpfdev_probe_0") {
		t.Fatal("candidate missing injected probe function definition")
	}
	if !strings.Contains(candPTX, ".func __bpftime_cuda__kernel_trace") {
		t.Fatal("candidate missing original empty stub function definition")
	}

	// Verify kernel identity, ABI, and parseability by cudaptx.
	if candidate.Digest == artifact.Digest {
		t.Fatal("candidate digest did not change from original")
	}
	if !reflect.DeepEqual(candidate.Schema, artifact.Schema) {
		t.Fatal("candidate schema changed from original")
	}
	parsedMod, err := cudaptx.ParseBytes(candidate.Raw)
	if err != nil {
		t.Fatalf("candidate is not parseable by cudaptx: %v", err)
	}
	if len(parsedMod.Kernels) < 2 {
		t.Fatalf("expected at least 2 functions/kernels in module, got %d", len(parsedMod.Kernels))
	}

	// 2. Fallback when stub is not opted into (policy.StubFunctions is empty).
	noStubPolicy := InjectionPolicy{
		Enabled:   true,
		Allowlist: []KernelIdentity{{artifact.Digest, artifact.EntryPoint, artifact.Target}},
		Entry:     true,
	}
	injNoStub, err := PrepareInjection(source, p, b, noStubPolicy)
	if err != nil {
		t.Fatalf("PrepareInjection failed: %v", err)
	}
	if cand, r := injNoStub.Candidate(); cand != nil || !strings.Contains(r, "instruction outside straight-line contract") {
		t.Fatalf("expected fallback when stub not opted into, got cand=%v reason=%q", cand != nil, r)
	}

	// 3. Fallback when stub function is not empty (contains arbitrary logic).
	nonEmptyStubPTX := strings.Replace(stubPTX, ".func __bpftime_cuda__kernel_trace\n{\n    ret;\n}",
		".func __bpftime_cuda__kernel_trace\n{\n    mov.u32 %value, 1;\n    ret;\n}", 1)
	nonEmptyArtifact, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(nonEmptyStubPTX), "affine", schema, "sm_121a")
	if err != nil {
		t.Fatal(err)
	}
	nonEmptySource, err := NewArtifactSource(nonEmptyArtifact, Geometry{Grid: [3]uint32{1, 1, 1}, Block: [3]uint32{32, 1, 1}})
	if err != nil {
		t.Fatal(err)
	}
	nonEmptyPolicy := InjectionPolicy{
		Enabled:       true,
		Allowlist:     []KernelIdentity{{nonEmptyArtifact.Digest, nonEmptyArtifact.EntryPoint, nonEmptyArtifact.Target}},
		StubFunctions: []string{"__bpftime_cuda__kernel_trace"},
	}
	injNonEmpty, err := PrepareInjection(nonEmptySource, p, b, nonEmptyPolicy)
	if err != nil {
		t.Fatalf("PrepareInjection failed: %v", err)
	}
	if cand, r := injNonEmpty.Candidate(); cand != nil || !strings.Contains(r, "stub function is not empty") {
		t.Fatalf("expected fallback for non-empty stub, got cand=%v reason=%q", cand != nil, r)
	}

	// A declared stub ABI is rejected, even if its call-site arguments are present.
	parameterStubPTX := strings.Replace(stubPTX,
		".func __bpftime_cuda__kernel_trace\n{",
		".func __bpftime_cuda__kernel_trace(.param .u32 x)\n{", 1)
	parameterStubPTX = strings.Replace(parameterStubPTX,
		"call.uni __bpftime_cuda__kernel_trace;",
		"call.uni __bpftime_cuda__kernel_trace, (%value);", 1)
	parameterSource, parameterArtifact := newSource(parameterStubPTX)
	parameterPolicy := InjectionPolicy{Enabled: true, Allowlist: []KernelIdentity{{parameterArtifact.Digest, parameterArtifact.EntryPoint, parameterArtifact.Target}}, StubFunctions: []string{"__bpftime_cuda__kernel_trace"}}
	injParameter, err := PrepareInjection(parameterSource, p, b, parameterPolicy)
	if err != nil {
		t.Fatal(err)
	}
	if cand, r := injParameter.Candidate(); cand != nil || !strings.Contains(r, "stub function has parameters or returns") {
		t.Fatalf("expected parameterized stub fallback, got candidate=%t reason=%q", cand != nil, r)
	}

	// An empty but unlisted device function is outside the admitted module contract.
	unlistedPTX := strings.Replace(stubPTX, ".visible .entry affine(", ".func helper\n{\n    ret;\n}\n\n.visible .entry affine(", 1)
	unlistedSource, unlistedArtifact := newSource(unlistedPTX)
	unlistedPolicy := InjectionPolicy{Enabled: true, Allowlist: []KernelIdentity{{unlistedArtifact.Digest, unlistedArtifact.EntryPoint, unlistedArtifact.Target}}, StubFunctions: []string{"__bpftime_cuda__kernel_trace"}, Entry: true}
	injUnlisted, err := PrepareInjection(unlistedSource, p, b, unlistedPolicy)
	if err != nil {
		t.Fatal(err)
	}
	if cand, r := injUnlisted.Candidate(); cand != nil || !strings.Contains(r, "module outside injection contract") {
		t.Fatalf("expected unlisted function fallback, got candidate=%t reason=%q", cand != nil, r)
	}

	// Duplicate entry names fail closed with a located reason.
	duplicateEntryPTX := strings.Replace(stubPTX, ".func __bpftime_cuda__kernel_trace", ".visible .entry affine()\n{\n    ret;\n}\n\n.func __bpftime_cuda__kernel_trace", 1)
	duplicateSource, duplicateArtifact := newSource(duplicateEntryPTX)
	duplicatePolicy := InjectionPolicy{Enabled: true, Allowlist: []KernelIdentity{{duplicateArtifact.Digest, duplicateArtifact.EntryPoint, duplicateArtifact.Target}}, Entry: true}
	injDuplicate, err := PrepareInjection(duplicateSource, p, b, duplicatePolicy)
	if err != nil {
		t.Fatal(err)
	}
	if cand, r := injDuplicate.Candidate(); cand != nil || !strings.Contains(r, "multiple entry kernels with entry name") {
		t.Fatalf("expected duplicate entry fallback, got candidate=%t reason=%q", cand != nil, r)
	}

	// A bodiless declaration is not a defined empty stub.
	bodilessPTX := strings.Replace(stubPTX, ".func __bpftime_cuda__kernel_trace\n{\n    ret;\n}\n", ".extern .func __bpftime_cuda__kernel_trace;\n", 1)
	bodilessSource, bodilessArtifact := newSource(bodilessPTX)
	bodilessPolicy := InjectionPolicy{Enabled: true, Allowlist: []KernelIdentity{{bodilessArtifact.Digest, bodilessArtifact.EntryPoint, bodilessArtifact.Target}}, StubFunctions: []string{"__bpftime_cuda__kernel_trace"}}
	injBodiless, err := PrepareInjection(bodilessSource, p, b, bodilessPolicy)
	if err != nil {
		t.Fatal(err)
	}
	if cand, r := injBodiless.Candidate(); cand != nil || !strings.Contains(r, "stub function has no body") {
		t.Fatalf("expected bodiless stub fallback, got candidate=%t reason=%q", cand != nil, r)
	}

	// Two calls receive distinct function definitions and both parse without duplicate names.
	multipleCallsPTX := strings.Replace(stubPTX, "call.uni __bpftime_cuda__kernel_trace;", "call.uni __bpftime_cuda__kernel_trace;\n    call.uni __bpftime_cuda__kernel_trace;", 1)
	multipleSource, multipleArtifact := newSource(multipleCallsPTX)
	multiplePolicy := InjectionPolicy{Enabled: true, Allowlist: []KernelIdentity{{multipleArtifact.Digest, multipleArtifact.EntryPoint, multipleArtifact.Target}}, StubFunctions: []string{"__bpftime_cuda__kernel_trace"}}
	injMultiple, err := PrepareInjection(multipleSource, p, b, multiplePolicy)
	if err != nil {
		t.Fatal(err)
	}
	multipleCandidate, reason := injMultiple.Candidate()
	if multipleCandidate == nil {
		t.Fatalf("multiple call-site preparation failed: %s", reason)
	}
	multiplePTX := string(multipleCandidate.Raw)
	if strings.Count(multiplePTX, "call.uni __cudabpfdev_probe_") != 2 || !strings.Contains(multiplePTX, "call.uni __cudabpfdev_probe_0;") || !strings.Contains(multiplePTX, "call.uni __cudabpfdev_probe_1;") {
		t.Fatalf("two call sites were not routed to distinct probes:\n%s", multiplePTX)
	}
	multipleModule, err := cudaptx.ParseBytes(multipleCandidate.Raw)
	if err != nil {
		t.Fatalf("multiple call-site candidate is not parseable by cudaptx: %v", err)
	}
	functionNames := map[string]bool{}
	for _, k := range multipleModule.Kernels {
		if functionNames[k.Name] {
			t.Fatalf("duplicate function name %q", k.Name)
		}
		functionNames[k.Name] = true
	}
	if !functionNames["__cudabpfdev_probe_0"] || !functionNames["__cudabpfdev_probe_1"] {
		t.Fatalf("probe functions missing from parsed module: %v", functionNames)
	}

	// The allowance fits the original kernel plus one probe, but not both stub sites.
	layout, err := inspectInjection(multipleArtifact, multiplePolicy)
	if err != nil {
		t.Fatal(err)
	}
	translation, err := Translate(p, b, multipleArtifact.Target)
	if err != nil {
		t.Fatal(err)
	}
	multipleBudget := b
	multipleBudget.MaxRegisters = layout.registers + translation.DeclaredRegisters + 1
	injOverBudget, err := PrepareInjection(multipleSource, p, multipleBudget, multiplePolicy)
	if err != nil {
		t.Fatal(err)
	}
	if cand, r := injOverBudget.Candidate(); cand != nil || !strings.Contains(r, "combined register declaration budget exceeded") {
		t.Fatalf("expected combined probe budget fallback, got candidate=%t reason=%q", cand != nil, r)
	}
}
