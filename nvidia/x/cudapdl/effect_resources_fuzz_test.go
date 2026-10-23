package cudapdl_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudapdl"
)

var (
	fuzzTargets = []string{
		"sm_90",
		"sm_90a",
		"sm_100a",
		"sm_121a",
		"sm_80",
		"sm_95",
		"invalid",
		"",
	}
	fuzzResources = []string{
		"handoff",
		"aux",
		"scratch",
		"",
	}
	fuzzExecGroups = []string{
		"",
		"thread",
		"warp",
		"cta",
		"cluster",
		"grid",
		"wgmma",
		"tcgen05",
		"invalid_group",
	}
)

func FuzzPDLEffectResourceEligibility(f *testing.F) {
	// Seed corpus covering all required positive, boundary, aliasing, overflow, and target cases.
	seeds := [][]byte{
		// 1. Valid admitted pair with independent preamble (WaitAfterIndependentPreamble) on sm_90, sm_100a, sm_121a.
		{0, 0, 0, 0},
		{0, 2, 0, 0},
		{0, 3, 0, 0},
		// 2. Valid admitted pair with entry wait (WaitAtEntryBeforeDependentBody).
		{1, 0, 0, 0},
		{1, 3, 0, 0},
		// 3. Valid admitted pair with matching concrete [Base, Base+Bytes) on "handoff" and disjoint "aux".
		{2, 2, 16, 32},
		// 4. Range [Base, Base+Bytes) uint64 addition overflow (Base: ^uint64(0)-7, Bytes: 16).
		{3, 0, 7, 16},
		// 5. Range aliasing between consumer BeforeWait and producer BeforeLaunchDependents / AfterLaunchDependents.
		{4, 0, 0, 0},
		{4, 0, 1, 0},
		// 6. Inconsistent partial range tracking across phases (Base != 0 in producer, Base == 0 in consumer on "handoff").
		{5, 0, 0, 0},
		// 7. Invalid trigger/wait placements (TriggerBeforeSharedWritesComplete, WaitAfterDependentAccess, Unknown).
		{6, 0, 0, 0},
		{6, 0, 1, 0},
		{6, 0, 2, 0},
		{6, 0, 3, 0},
		// 8. Resource boundary & overflow cases:
		//    - StaticSharedMemory: ^uint32(0), DynamicSharedMemory: 4096
		{7, 0, 0, 0},
		//    - TMEMBytes: 16384 on sm_121a vs sm_100a
		{8, 3, 0, 0},
		{8, 2, 0, 0},
		//    - ExecutionGroup: "wgmma" on sm_121a vs sm_90a
		{9, 3, 0, 0},
		{9, 1, 0, 0},
		//    - ThreadsPerBlock: 2048, AllocatedRegisters: 256, PeakLiveGPR > AllocatedRegisters
		{10, 0, 0, 0},
		{10, 0, 1, 0},
		{10, 0, 2, 0},
		//    - RequireSingleSMCoResidency: true with oversized shared memory, and estimated target sm_95
		{11, 3, 0, 0},
		{11, 5, 0, 0},
		// 9. Arbitrary bounded synthesized descriptors.
		{12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		{13, 3, 2, 1, 0, 255, 128, 64, 32, 16, 8, 4, 2, 1},
	}
	for _, seed := range seeds {
		f.Add(seed)
	}

	driverLoaded, compilerLoaded := driver.Loaded(), nvrtc.Loaded()

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 || len(data) > 128 {
			return
		}

		target, prod, cons := buildFuzzCase(data)

		got := cudapdl.CheckEligibility(target, prod, cons)
		adapterGot := cudapdl.NewPlanAdapter(target).Evaluate(prod, cons)
		if got != adapterGot {
			t.Fatalf("PlanAdapter.Evaluate(%q) = %+v, CheckEligibility = %+v", target, adapterGot, got)
		}

		if got.Eligible {
			if got.Policy != cudapdl.PolicyProgrammatic {
				t.Fatalf("eligible pair has Policy=%v, want PolicyProgrammatic", got.Policy)
			}
			if got.FallbackReason != "" {
				t.Fatalf("eligible pair has non-empty FallbackReason=%q", got.FallbackReason)
			}
			if got.Provenance != cudapdl.ProvenanceOwnerDeclared {
				t.Fatalf("eligible pair has Provenance=%v, want ProvenanceOwnerDeclared", got.Provenance)
			}
			if !got.NoOverlapStillCorrect {
				t.Fatal("eligible pair must set NoOverlapStillCorrect=true")
			}
			if got.OverlapGuaranteed {
				t.Fatal("eligible pair must never set OverlapGuaranteed=true")
			}
			if !prod.HasLaunchDependents || !cons.HasWait || prod.IsVendor || cons.IsVendor {
				t.Fatalf("eligible pair violated basic trigger/wait/vendor requirements: prod=%+v cons=%+v", prod, cons)
			}
			if prod.Resources == nil && prod.Artifact == nil {
				t.Fatal("eligible producer has nil Resources and nil Artifact")
			}
			if cons.Resources == nil && cons.Artifact == nil {
				t.Fatal("eligible consumer has nil Resources and nil Artifact")
			}
			if cons.HasIndependentWork != got.ConsumerHasIndependentPreamble {
				t.Fatalf("ConsumerHasIndependentPreamble=%v != cons.HasIndependentWork=%v", got.ConsumerHasIndependentPreamble, cons.HasIndependentWork)
			}

			verifyFuzzInvalidMutations(t, target, prod, cons)
		} else {
			if got.Policy != cudapdl.PolicyStandardFallback {
				t.Fatalf("ineligible pair has Policy=%v, want PolicyStandardFallback", got.Policy)
			}
			if got.FallbackReason == "" {
				t.Fatal("ineligible pair has empty FallbackReason")
			}
			if got.Provenance != cudapdl.ProvenanceNone {
				t.Fatalf("ineligible pair has Provenance=%v, want ProvenanceNone", got.Provenance)
			}
			if got.OverlapGuaranteed {
				t.Fatal("ineligible pair must not report OverlapGuaranteed=true")
			}
		}

		if driver.Loaded() != driverLoaded || nvrtc.Loaded() != compilerLoaded {
			t.Fatal("fuzz evaluation loaded a native library")
		}
	})
}

func verifyFuzzInvalidMutations(t *testing.T, target string, prod, cons cudapdl.KernelDesc) {
	t.Helper()

	var handoffEffect cudapdl.PhaseEffect
	for _, eff := range prod.Effects.BeforeLaunchDependents {
		if eff.Kind == cudapdl.EffectMemory && (eff.Access == cudapdl.AccessWrite || eff.Access == cudapdl.AccessReadWrite || eff.Access == cudapdl.AccessAtomic) {
			handoffEffect = eff
			break
		}
	}

	mutations := []struct {
		name   string
		mutate func(p, c *cudapdl.KernelDesc)
	}{
		{
			name: "producer TriggerBeforeSharedWritesComplete",
			mutate: func(p, _ *cudapdl.KernelDesc) {
				p.Effects.TriggerPlacement = cudapdl.TriggerBeforeSharedWritesComplete
			},
		},
		{
			name: "consumer WaitAfterDependentAccess",
			mutate: func(_, c *cudapdl.KernelDesc) {
				c.Effects.WaitPlacement = cudapdl.WaitAfterDependentAccess
			},
		},
		{
			name: "consumer HasIndependentWork inverted",
			mutate: func(_, c *cudapdl.KernelDesc) {
				c.HasIndependentWork = !c.HasIndependentWork
			},
		},
		{
			name: "producer Effects cleared",
			mutate: func(p, _ *cudapdl.KernelDesc) {
				p.Effects = cudapdl.KernelEffects{}
			},
		},
		{
			name: "consumer Resources cleared",
			mutate: func(_, c *cudapdl.KernelDesc) {
				c.Resources = nil
				c.Artifact = nil
			},
		},
		{
			name: "producer StaticSharedMemory+DynamicSharedMemory uint32 overflow",
			mutate: func(p, _ *cudapdl.KernelDesc) {
				res := *p.Resources
				res.StaticSharedMemory = ^uint32(0)
				res.DynamicSharedMemory = 1
				p.Resources = &res
			},
		},
		{
			name: "consumer BeforeWait writes producer handoff resource",
			mutate: func(_, c *cudapdl.KernelDesc) {
				c.HasIndependentWork = true
				c.Effects.WaitPlacement = cudapdl.WaitAfterIndependentPreamble
				c.Effects.BeforeWait = []cudapdl.PhaseEffect{
					{
						Kind:     cudapdl.EffectMemory,
						Resource: handoffEffect.Resource,
						Access:   cudapdl.AccessWrite,
						Base:     handoffEffect.Base,
						Bytes:    handoffEffect.Bytes,
					},
				}
			},
		},
	}

	for _, m := range mutations {
		pCopy := cloneKernelDescForFuzz(prod)
		cCopy := cloneKernelDescForFuzz(cons)
		m.mutate(&pCopy, &cCopy)

		res := cudapdl.CheckEligibility(target, pCopy, cCopy)
		if res.Eligible || res.Policy != cudapdl.PolicyStandardFallback || res.FallbackReason == "" || res.Provenance != cudapdl.ProvenanceNone || res.OverlapGuaranteed {
			t.Fatalf("known-invalid mutation %q on admitted pair was not rejected: %+v", m.name, res)
		}
	}
}

func cloneKernelDescForFuzz(d cudapdl.KernelDesc) cudapdl.KernelDesc {
	out := d
	if d.Resources != nil {
		r := *d.Resources
		out.Resources = &r
	}
	out.Effects.BeforeLaunchDependents = append([]cudapdl.PhaseEffect(nil), d.Effects.BeforeLaunchDependents...)
	out.Effects.AfterLaunchDependents = append([]cudapdl.PhaseEffect(nil), d.Effects.AfterLaunchDependents...)
	out.Effects.BeforeWait = append([]cudapdl.PhaseEffect(nil), d.Effects.BeforeWait...)
	out.Effects.AfterWait = append([]cudapdl.PhaseEffect(nil), d.Effects.AfterWait...)
	return out
}

func buildFuzzCase(data []byte) (string, cudapdl.KernelDesc, cudapdl.KernelDesc) {
	b := func(i int) byte {
		if i < len(data) {
			return data[i]
		}
		return 0
	}

	mode := b(0) % 14
	target := fuzzTargets[int(b(1))%len(fuzzTargets)]
	prod := testProducerDesc("fuzz_prod")
	cons := testConsumerDesc("fuzz_cons")

	switch mode {
	case 0:
		// Valid pair with independent preamble (when target is sm_90, sm_90a, sm_100a, sm_121a).
		if b(2)%2 == 1 {
			prod.Effects.BeforeLaunchDependents[0].Resource = "aux"
			cons.Effects.BeforeWait[0].Resource = "aux"
		}
	case 1:
		// Valid entry-sync pair (WaitAtEntryBeforeDependentBody).
		cons.HasIndependentWork = false
		cons.Effects.WaitPlacement = cudapdl.WaitAtEntryBeforeDependentBody
		cons.Effects.BeforeWait = nil
	case 2:
		// Concrete non-overlapping [Base, Base+Bytes) ranges.
		baseHandoff := uint64(0x1000) + uint64(b(2))*64
		bytesHandoff := uint64(1024) + uint64(b(3))*16
		baseAux := baseHandoff + bytesHandoff + 0x1000
		prod.Effects.BeforeLaunchDependents[1].Base = baseHandoff
		prod.Effects.BeforeLaunchDependents[1].Bytes = bytesHandoff
		cons.Effects.AfterWait[0].Base = baseHandoff
		cons.Effects.AfterWait[0].Bytes = bytesHandoff
		cons.Effects.BeforeWait[0].Base = baseAux
		cons.Effects.BeforeWait[0].Bytes = 1024
	case 3:
		// Range [Base, Base+Bytes) uint64 addition overflow.
		sub := uint64(b(2))
		add := uint64(b(3)) + sub + 1
		cons.Effects.BeforeWait = []cudapdl.PhaseEffect{
			{
				Kind:     cudapdl.EffectMemory,
				Resource: "aux",
				Access:   cudapdl.AccessRead,
				Base:     math.MaxUint64 - sub,
				Bytes:    add,
			},
		}
	case 4:
		// Range aliasing between consumer BeforeWait and producer BeforeLaunchDependents / AfterLaunchDependents.
		if b(2)%2 == 0 {
			prod.Effects.BeforeLaunchDependents[1].Base = 0x4000
			prod.Effects.BeforeLaunchDependents[1].Bytes = 4096
			cons.Effects.AfterWait[0].Base = 0x4000
			cons.Effects.AfterWait[0].Bytes = 4096
			cons.Effects.BeforeWait = []cudapdl.PhaseEffect{
				{Kind: cudapdl.EffectMemory, Resource: "scratch", Access: cudapdl.AccessRead, Base: 0x4800, Bytes: 256},
			}
		} else {
			prod.Effects.AfterLaunchDependents = []cudapdl.PhaseEffect{
				{Kind: cudapdl.EffectMemory, Resource: "scratch", Access: cudapdl.AccessWrite, Base: 0x8000, Bytes: 1024},
			}
			cons.Effects.BeforeWait = []cudapdl.PhaseEffect{
				{Kind: cudapdl.EffectMemory, Resource: "aux", Access: cudapdl.AccessRead, Base: 0x8200, Bytes: 256},
			}
		}
	case 5:
		// Inconsistent partial range tracking across phases on "handoff".
		prod.Effects.BeforeLaunchDependents[1].Base = 0x2000
		prod.Effects.BeforeLaunchDependents[1].Bytes = 1024
		cons.Effects.AfterWait[0].Base = 0
		cons.Effects.AfterWait[0].Bytes = 1024
	case 6:
		// Invalid trigger/wait placements.
		switch b(2) % 4 {
		case 0:
			prod.Effects.TriggerPlacement = cudapdl.TriggerBeforeSharedWritesComplete
		case 1:
			cons.Effects.WaitPlacement = cudapdl.WaitAfterDependentAccess
		case 2:
			prod.Effects.TriggerPlacement = cudapdl.TriggerPlacementUnknown
		case 3:
			cons.Effects.WaitPlacement = cudapdl.WaitPlacementUnknown
		}
	case 7:
		// StaticSharedMemory + DynamicSharedMemory uint32 overflow.
		prod.Resources.StaticSharedMemory = math.MaxUint32 - uint32(b(2))
		prod.Resources.DynamicSharedMemory = 4096 + uint32(b(3))
	case 8:
		// TMEMBytes boundary on target.
		prod.Resources.TMEMBytes = 16384 + uint32(b(2))*1024
		if b(3)%2 == 1 {
			cons.Resources.TMEMBytes = 256 * 1024
		}
	case 9:
		// ExecutionGroup boundary ("wgmma", "tcgen05", etc.).
		if b(2)%2 == 0 {
			prod.Resources.ExecutionGroup = "wgmma"
		} else {
			prod.Resources.ExecutionGroup = fuzzExecGroups[int(b(3))%len(fuzzExecGroups)]
		}
	case 10:
		// ThreadsPerBlock / AllocatedRegisters / PeakLiveGPR bounds.
		switch b(2) % 3 {
		case 0:
			prod.ThreadsPerBlock = 2048
		case 1:
			prod.Resources.AllocatedRegisters = 256
		case 2:
			prod.Resources.AllocatedRegisters = 32
			prod.Resources.PeakLiveGPR = 64
		}
	case 11:
		// RequireSingleSMCoResidency with oversized combined shared memory.
		prod.Resources.StaticSharedMemory = 65536
		cons.Resources.StaticSharedMemory = 65536
		cons.Effects.RequireSingleSMCoResidency = true
	default:
		// Modes 12..13: synthesize bounded descriptors directly from fuzz bytes.
		return decodeArbitraryFuzzPair(data)
	}

	return target, prod, cons
}

func decodeArbitraryFuzzPair(data []byte) (string, cudapdl.KernelDesc, cudapdl.KernelDesc) {
	var padded [64]byte
	copy(padded[:], data)

	target := fuzzTargets[int(padded[1])%len(fuzzTargets)]

	prod := cudapdl.KernelDesc{
		Name:                "synth_prod",
		IsVendor:            padded[2]&1 != 0,
		HasLaunchDependents: padded[2]&2 == 0,
		ThreadsPerBlock:     uint32(padded[3]) * 8,
		Resources: &cudakernel.ResourceRequirements{
			StaticSharedMemory:  uint32(binary.LittleEndian.Uint16(padded[4:6])) * 64,
			DynamicSharedMemory: uint32(binary.LittleEndian.Uint16(padded[6:8])) * 64,
			AllocatedRegisters:  uint32(padded[8]),
			PeakLiveGPR:         uint32(padded[9]),
			RegisterQuantum:     uint32(padded[10]&1) * 256,
			TMEMBytes:           uint32(padded[11]&3) * 16384,
			ExecutionGroup:      fuzzExecGroups[int(padded[12])%len(fuzzExecGroups)],
		},
		Effects: cudapdl.KernelEffects{
			TriggerPlacement:           cudapdl.TriggerPlacement(padded[13] % 4),
			BeforeLaunchDependents:     decodeFuzzEffects(padded[14:26]),
			AfterLaunchDependents:      decodeFuzzEffects(padded[26:34]),
			RequireSingleSMCoResidency: padded[34]&1 != 0,
		},
	}

	cons := cudapdl.KernelDesc{
		Name:               "synth_cons",
		IsVendor:           padded[35]&1 != 0,
		HasWait:            padded[35]&2 == 0,
		HasIndependentWork: padded[35]&4 != 0,
		ThreadsPerBlock:    uint32(padded[36]) * 8,
		Resources: &cudakernel.ResourceRequirements{
			StaticSharedMemory:  uint32(binary.LittleEndian.Uint16(padded[37:39])) * 64,
			DynamicSharedMemory: uint32(binary.LittleEndian.Uint16(padded[39:41])) * 64,
			AllocatedRegisters:  uint32(padded[41]),
			PeakLiveGPR:         uint32(padded[42]),
			RegisterQuantum:     uint32(padded[43]&1) * 256,
			TMEMBytes:           uint32(padded[44]&3) * 16384,
			ExecutionGroup:      fuzzExecGroups[int(padded[45])%len(fuzzExecGroups)],
		},
		Effects: cudapdl.KernelEffects{
			WaitPlacement:              cudapdl.WaitPlacement(padded[46] % 5),
			BeforeWait:                 decodeFuzzEffects(padded[47:55]),
			AfterWait:                  decodeFuzzEffects(padded[55:63]),
			RequireSingleSMCoResidency: padded[63]&1 != 0,
		},
	}

	return target, prod, cons
}

func decodeFuzzEffects(chunk []byte) []cudapdl.PhaseEffect {
	if len(chunk) < 4 {
		return nil
	}
	count := int(chunk[0] % 4) // 0..3 effects per phase
	if count == 0 {
		return nil
	}
	out := make([]cudapdl.PhaseEffect, 0, count)
	for i := 0; i < count && 1+i*3+2 < len(chunk); i++ {
		b0 := chunk[1+i*3]
		b1 := chunk[1+i*3+1]
		b2 := chunk[1+i*3+2]

		kind := cudapdl.EffectKind(b0 % 10)
		if kind == cudapdl.EffectPureCompute {
			out = append(out, cudapdl.PhaseEffect{Kind: cudapdl.EffectPureCompute})
			continue
		}
		res := fuzzResources[int(b1)%len(fuzzResources)]
		access := cudapdl.AccessMode((b1 >> 2) % 6)
		var base uint64
		if b2&0x80 != 0 {
			base = math.MaxUint64 - uint64(b2&0x0f)
		} else if b2&0x40 != 0 {
			base = uint64(b2&0x3f) * 256
		}
		bytes := uint64((b0>>4)&0x0f) * 256
		out = append(out, cudapdl.PhaseEffect{
			Kind:     kind,
			Resource: res,
			Access:   access,
			Base:     base,
			Bytes:    bytes,
		})
	}
	return out
}
