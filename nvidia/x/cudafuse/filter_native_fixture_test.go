package cudafuse

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

func prepareFilterNativeProbe(t *testing.T, b *filterNativeBuilt, eager bool) (*executionPlan, *filterProbe, map[string]executionInput, executionOps) {
	t.Helper()
	plan, err := snapshotExecution(b.Plan, eager)
	if err != nil {
		t.Fatal(err)
	}
	probe := &filterProbe{storage: make(map[driver.DevicePtr][]float32)}
	dev := new(cuda.Device)
	bindings := make(map[string]executionInput)
	bind := func(node *LazyNode, values []float32) {
		ptr := driver.DevicePtr(node.ID * 8192)
		probe.storage[ptr] = slices.Clone(values)
		bindings[node.Name()] = executionInput{
			description: executionBindingDescription{
				shape:  slices.Clone(node.Shape),
				dtype:  node.DType,
				length: len(values),
				ptr:    uint64(ptr),
				bytes:  uint64(len(values)) * 4,
				device: dev,
			},
			borrow: func(*cuda.Device) (driver.DevicePtr, func(), error) {
				return ptr, func() { probe.releases++ }, nil
			},
		}
	}
	bind(b.NodeA, b.Case.DataA)
	bind(b.NodeB, b.Case.DataB)
	bind(b.NodeP, b.Case.Predicate)
	if b.NodeS != nil {
		bind(b.NodeS, []float32{b.Case.Scalar})
	}
	ops := probe.ops()
	baseCompile := ops.compile
	ops.compile = func(f FusedKernel) (*executionKernel, error) {
		k, err := baseCompile(f)
		if k != nil {
			srcSum := fmt.Sprintf("%x", sha256.Sum256([]byte(f.Source)))
			ptxSum := fmt.Sprintf("%x", sha256.Sum256([]byte("ptx:"+f.Name+":"+f.Source)))
			keySum := fmt.Sprintf("%x", sha256.Sum256([]byte("key:"+f.Name)))
			k.receipt = KernelReceipt{
				Name:         f.Name,
				SourceDigest: srcSum,
				PTXDigest:    ptxSum,
				Target:       filterNativeDefaultSM,
				CompilerKey:  keySum,
			}
		}
		return k, err
	}
	return plan, probe, bindings, ops
}

func TestFilterNativeFixtureContract(t *testing.T) {
	filterHostOnly(t)
	cases := filterNativeCases()
	if len(cases) != 16 {
		t.Fatalf("filterNativeCases() count = %d, want 16", len(cases))
	}
	for _, tc := range cases {
		for _, eager := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/eager=%v", tc.Name, eager), func(t *testing.T) {
				filterHostOnly(t)
				b, err := buildFilterNativeFixture(tc)
				if err != nil {
					t.Fatal(err)
				}
				switch tc.Mode {
				case "none":
					if b.Selected != 0 {
						t.Fatalf("none mode selected %d elements, want 0", b.Selected)
					}
				case "all":
					if b.Selected != tc.Length {
						t.Fatalf("all mode selected %d elements, want %d", b.Selected, tc.Length)
					}
				case "mixed":
					if b.Selected <= 0 || b.Selected >= tc.Length {
						t.Fatalf("mixed mode selected %d of %d elements, want strict subset", b.Selected, tc.Length)
					}
				}
				plan, probe, bindings, ops := prepareFilterNativeProbe(t, b, eager)
				var dev *cuda.Device
				for _, in := range bindings {
					dev = in.description.device
					break
				}
				e, err := executePlanInputs(context.Background(), plan, dev, bindings, ops)
				if e != nil {
					t.Cleanup(func() {
						if err := e.Close(); err != nil {
							t.Error(err)
						}
					})
				}
				if err != nil {
					t.Fatal(err)
				}
				got, err := e.Readback(context.Background())
				if err != nil {
					t.Fatal(err)
				}
				receipts := e.Receipts()
				hash := sha256.New()
				for _, r := range receipts {
					hash.Write([]byte(r.SourceDigest))
				}
				sourceDigest := fmt.Sprintf("%x", hash.Sum(nil))
				if err := compareFilterNativeResult(b, got, e.Shape(), filterNativeDefaultSM, sourceDigest); err != nil {
					t.Fatal(err)
				}
				if err := verifyFilterNativeStats(tc, b.Selected, eager, e.Stats(), receipts, false); err != nil {
					t.Fatal(err)
				}
				if err := e.Close(); err != nil {
					t.Fatal(err)
				}
				if err := verifyFilterNativeStats(tc, b.Selected, eager, e.Stats(), receipts, true); err != nil {
					t.Fatal(err)
				}
				if probe.frees != probe.allocations || probe.releases != len(bindings) {
					t.Fatalf("probe custody mismatch: frees=%d allocs=%d releases=%d bindings=%d", probe.frees, probe.allocations, probe.releases, len(bindings))
				}
			})
		}
	}
}

func TestFilterNativeFixtureComparatorControls(t *testing.T) {
	filterHostOnly(t)
	var mixedFilter, mixedPointwise, mixedScan, tail259Filter filterNativeCase
	for _, tc := range filterNativeCases() {
		switch tc.Name {
		case "mixed-signed-zero-33/filter":
			mixedFilter = tc
		case "mixed-signed-zero-33/pointwise":
			mixedPointwise = tc
		case "mixed-signed-zero-33/scan":
			mixedScan = tc
		case "mixed-tail-259/filter":
			tail259Filter = tc
		}
	}
	builtFilter, err := buildFilterNativeFixture(mixedFilter)
	if err != nil {
		t.Fatal(err)
	}
	builtPointwise, err := buildFilterNativeFixture(mixedPointwise)
	if err != nil {
		t.Fatal(err)
	}
	builtScan, err := buildFilterNativeFixture(mixedScan)
	if err != nil {
		t.Fatal(err)
	}
	builtTail259, err := buildFilterNativeFixture(tail259Filter)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("wrong-selected-count", func(t *testing.T) {
		short := slices.Clone(builtFilter.Want[:len(builtFilter.Want)-1])
		if err := compareFilterNativeResult(builtFilter, short, []int{len(short)}, "host-control", ""); err == nil {
			t.Fatal("comparator accepted truncated selected count")
		}
		if err := compareFilterNativeResult(builtFilter, short, builtFilter.WantShape, "host-control", ""); err == nil {
			t.Fatal("comparator accepted truncated output with old shape")
		}
		extra := append(slices.Clone(builtFilter.Want), 1.0)
		if err := compareFilterNativeResult(builtFilter, extra, []int{len(extra)}, "host-control", ""); err == nil {
			t.Fatal("comparator accepted extra selected count")
		}
		if err := compareFilterNativeResult(builtFilter, slices.Clone(builtFilter.Want), []int{len(builtFilter.Want) + 1}, "host-control", ""); err == nil {
			t.Fatal("comparator accepted mismatched shape extent")
		}
	})

	t.Run("wrong-element-order", func(t *testing.T) {
		for _, b := range []*filterNativeBuilt{builtFilter, builtPointwise, builtScan} {
			reordered := slices.Clone(b.Want)
			for i := 0; i < len(reordered)-1; i++ {
				if reordered[i] != reordered[len(reordered)-1] {
					reordered[i], reordered[len(reordered)-1] = reordered[len(reordered)-1], reordered[i]
					break
				}
			}
			if slices.Equal(reordered, b.Want) {
				t.Fatalf("failed to construct distinct order permutation for %s", b.Case.Name)
			}
			if err := compareFilterNativeResult(b, reordered, slices.Clone(b.WantShape), "host-control", ""); err == nil {
				t.Fatalf("comparator accepted unstable element order for %s", b.Case.Name)
			}
		}
	})

	t.Run("wrong-predicate-evaluation", func(t *testing.T) {
		// Control 1: treating -0.0 as true selects elements whose predicate is -0.0.
		negZeroSelected := make([]float32, 0, mixedFilter.Length)
		hasNegZero := false
		for i, p := range mixedFilter.Predicate {
			if p == 0 && math.Signbit(float64(p)) {
				hasNegZero = true
			}
			if p == 1 || (p == 0 && math.Signbit(float64(p))) {
				negZeroSelected = append(negZeroSelected, (mixedFilter.DataA[i]+mixedFilter.DataB[i])*mixedFilter.DataA[i])
			}
		}
		if !hasNegZero {
			t.Fatal("mixed fixture lacks -0.0 predicate entries")
		}
		if err := compareFilterNativeResult(builtFilter, negZeroSelected, []int{len(negZeroSelected)}, "host-control", ""); err == nil {
			t.Fatal("comparator accepted -0.0 predicate treated as true")
		}

		// Control 2: replacing a selected value with an unselected position's value (same count, wrong predicate index).
		wrongPredEval := slices.Clone(builtFilter.Want)
		for i, p := range mixedFilter.Predicate {
			if p == 0 {
				unselectedVal := (mixedFilter.DataA[i] + mixedFilter.DataB[i]) * mixedFilter.DataA[i]
				if unselectedVal != wrongPredEval[0] {
					wrongPredEval[0] = unselectedVal
					break
				}
			}
		}
		if err := compareFilterNativeResult(builtFilter, wrongPredEval, slices.Clone(builtFilter.WantShape), "host-control", ""); err == nil {
			t.Fatal("comparator accepted unselected element substituted for selected element")
		}

		// Control 3: non-0/1 predicates must be rejected by both the oracle and execution count validation.
		for _, bad := range []float32{-1.0, -0.5, 0.5, 2.0, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
			badCase := mixedFilter
			badCase.Predicate = slices.Clone(mixedFilter.Predicate)
			badCase.Predicate[mixedFilter.Length/2] = bad
			if _, _, _, err := evaluateFilterNativeOracle(badCase); err == nil {
				t.Fatalf("oracle accepted invalid predicate %v", bad)
			}
			if _, err := buildFilterNativeFixture(badCase); err == nil {
				t.Fatalf("fixture builder accepted invalid predicate %v", bad)
			}
			b, err := buildFilterNativeFixture(mixedFilter)
			if err != nil {
				t.Fatal(err)
			}
			b.Case.Predicate = slices.Clone(mixedFilter.Predicate)
			b.Case.Predicate[mixedFilter.Length/2] = bad
			plan, probe, bindings, ops := prepareFilterNativeProbe(t, b, false)
			var dev *cuda.Device
			for _, in := range bindings {
				dev = in.description.device
				break
			}
			e, err := executePlanInputs(context.Background(), plan, dev, bindings, ops)
			if e == nil || err == nil || probe.allocations != 4 {
				t.Fatalf("runtime accepted invalid predicate %v: e=%v err=%v allocs=%d", bad, e, err, probe.allocations)
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
		}
	})

	t.Run("wrong-nondivisible-tail-element", func(t *testing.T) {
		for _, b := range []*filterNativeBuilt{builtFilter, builtTail259} {
			if b.Case.Predicate[b.Case.Length-1] != 1 {
				t.Fatalf("fixture %s does not select final non-divisible tail element", b.Case.Name)
			}
			corruptTail := slices.Clone(b.Want)
			corruptTail[len(corruptTail)-1] += 0.25
			if err := compareFilterNativeResult(b, corruptTail, slices.Clone(b.WantShape), "host-control", ""); err == nil {
				t.Fatalf("comparator accepted corrupted non-divisible tail element for %s", b.Case.Name)
			}
			droppedTail := slices.Clone(b.Want[:len(b.Want)-1])
			if err := compareFilterNativeResult(b, droppedTail, []int{len(droppedTail)}, "host-control", ""); err == nil {
				t.Fatalf("comparator accepted dropped non-divisible tail element for %s", b.Case.Name)
			}
		}
	})

	t.Run("unwritten-nan-inf-poison", func(t *testing.T) {
		poison := math.Float32frombits(filterNativePoisonBits)
		for _, badVal := range []struct {
			name    string
			value   float32
			wantErr error
		}{
			{"reserved-unwritten-poison", poison, ErrOutputUnwritten},
			{"quiet-nan", float32(math.NaN()), errFilterNativePoison},
			{"pos-inf", float32(math.Inf(1)), errFilterNativePoison},
			{"neg-inf", float32(math.Inf(-1)), errFilterNativePoison},
		} {
			corrupt := slices.Clone(builtFilter.Want)
			corrupt[len(corrupt)-1] = badVal.value
			err := compareFilterNativeResult(builtFilter, corrupt, slices.Clone(builtFilter.WantShape), "host-control", "")
			if !errors.Is(err, badVal.wantErr) {
				t.Fatalf("%s error = %v, want %v", badVal.name, err, badVal.wantErr)
			}
		}

		// Verify checkExecutionOutput rejects unwritten poison in payload and overwritten guards.
		n := len(builtFilter.Want)
		guarded := make([]float32, n+2)
		guarded[0] = poison
		copy(guarded[1:n+1], builtFilter.Want)
		guarded[n+1] = poison
		if err := checkExecutionOutput(guarded, n); err != nil {
			t.Fatal(err)
		}
		unwritten := slices.Clone(guarded)
		unwritten[n] = poison
		if err := checkExecutionOutput(unwritten, n); !errors.Is(err, ErrOutputUnwritten) {
			t.Fatalf("unwritten tail element error = %v, want ErrOutputUnwritten", err)
		}
		badLowGuard := slices.Clone(guarded)
		badLowGuard[0] = 0
		if err := checkExecutionOutput(badLowGuard, n); !errors.Is(err, ErrOutputGuard) {
			t.Fatalf("low guard error = %v, want ErrOutputGuard", err)
		}
		badHighGuard := slices.Clone(guarded)
		badHighGuard[n+1] = 0
		if err := checkExecutionOutput(badHighGuard, n); !errors.Is(err, ErrOutputGuard) {
			t.Fatalf("high guard error = %v, want ErrOutputGuard", err)
		}
	})
}

func TestFilterNativeFixtureGateRefusalBeforeVendorInit(t *testing.T) {
	filterHostOnly(t)
	envMap := func(kv map[string]string) func(string) string {
		return func(k string) string { return kv[k] }
	}

	// 1. Closed gate (unset or "0") must refuse with errFilterNativeOptIn and never count as a pass.
	for _, val := range []string{"", "0", "false"} {
		_, err := filterNativeAdmission(envMap(map[string]string{
			filterNativeEnvVar: val,
			"GPU_LOCK_HELD":    "12345",
		}))
		if !errors.Is(err, errFilterNativeOptIn) {
			t.Fatalf("closed gate (%q) error = %v, want errFilterNativeOptIn", val, err)
		}
	}

	// 2. Missing or invalid GPU_LOCK_HELD must refuse before vendor init.
	for _, lockVal := range []string{"", "1", "0", "-7", "not-a-pid"} {
		_, err := filterNativeAdmission(envMap(map[string]string{
			filterNativeEnvVar:    "1",
			filterNativeTargetVar: filterNativeDefaultSM,
			"GPU_LOCK_HELD":       lockVal,
		}))
		if err == nil || errors.Is(err, errFilterNativeOptIn) {
			t.Fatalf("missing/invalid GPU_LOCK_HELD (%q) was not refused as lock failure: %v", lockVal, err)
		}
	}

	// 3. Unknown target must refuse with cudaerrors.ErrUnsupported before lock/vendor init.
	for _, badTarget := range []string{"sm_999", "sm_10", "cpu", "unknown_arch"} {
		_, err := filterNativeAdmission(envMap(map[string]string{
			filterNativeEnvVar:    "1",
			filterNativeTargetVar: badTarget,
			"GPU_LOCK_HELD":       "12345",
		}))
		if !errors.Is(err, cudaerrors.ErrUnsupported) {
			t.Fatalf("unknown target %q error = %v, want ErrUnsupported", badTarget, err)
		}
	}

	// 4. Plan capability contract and nil-stream execution refusal without vendor calls.
	b, err := buildFilterNativeFixture(filterNativeCases()[0])
	if err != nil {
		t.Fatal(err)
	}
	for _, known := range []string{"sm_90a", "sm_100a", "sm_121a"} {
		if err := b.Plan.CheckTarget(known); !errors.Is(err, cudaerrors.ErrHardwarePending) {
			t.Fatalf("CheckTarget(%q) = %v, want ErrHardwarePending", known, err)
		}
	}
	for _, unknown := range []string{"", "cpu", "sm_999", "sm_121zz"} {
		if err := b.Plan.CheckTarget(unknown); !errors.Is(err, cudaerrors.ErrUnsupported) {
			t.Fatalf("CheckTarget(%q) = %v, want ErrUnsupported", unknown, err)
		}
	}
	if owner, err := b.Plan.Execute(context.Background(), nil, nil); err == nil || owner != nil || !strings.Contains(err.Error(), "nil execution stream") {
		t.Fatalf("Execute(nil stream) = (%v, %v), want nil owner and nil execution stream error", owner, err)
	}
	if owner, err := b.Plan.ExecuteEager(context.Background(), nil, nil); err == nil || owner != nil || !strings.Contains(err.Error(), "nil execution stream") {
		t.Fatalf("ExecuteEager(nil stream) = (%v, %v), want nil owner and nil execution stream error", owner, err)
	}
}

func TestFilterNativeFixtureMutationEvidence(t *testing.T) {
	filterHostOnly(t)
	for _, failure := range []string{"", "compile", "allocation", "launch", "drain", "read", "cleanup", "unmutated"} {
		t.Run(failure, func(t *testing.T) {
			b, err := buildFilterNativeFixture(filterNativeCases()[0])
			// Use a mixed, nonempty case so inversion must disagree numerically.
			for _, tc := range filterNativeCases() {
				if tc.Name == "mixed-signed-zero-33/filter" {
					b, err = buildFilterNativeFixture(tc)
					break
				}
			}
			if err != nil {
				t.Fatal(err)
			}
			plan, probe, bindings, ops := prepareFilterNativeProbe(t, b, false)
			switch failure {
			case "compile":
				probe.failCompile = 1
			case "allocation":
				probe.failAlloc = 1
			case "launch":
				probe.failLaunch = 1
			case "drain":
				probe.failSync = true
			case "read":
				probe.failRead = 1
			case "cleanup":
				probe.failFree = true
			}
			var evidence filterCountMutationEvidence
			ops = evidence.observe(ops)
			compile := ops.compile
			ops.compile = func(source FusedKernel) (*executionKernel, error) {
				if failure != "unmutated" && strings.Contains(source.Source, "float filter_selected =") {
					source.Source = strings.Replace(source.Source, "== 1.0f ? 1.0f : 0.0f;", "== 0.0f ? 1.0f : 0.0f;", 1)
				}
				return compile(source)
			}
			var dev *cuda.Device
			for _, binding := range bindings {
				dev = binding.description.device
				break
			}
			owner, execErr := executePlanInputs(context.Background(), plan, dev, bindings, ops)
			var closeErr error
			if owner != nil {
				closeErr = owner.Close()
			}
			qualification := evidence.verify(owner, execErr, b.Case.Predicate)
			if failure == "" {
				if closeErr != nil || qualification != nil {
					t.Fatalf("semantic control = %v, cleanup = %v", qualification, closeErr)
				}
			} else if closeErr == nil && qualification == nil {
				t.Fatalf("%s failure counted as semantic discrimination", failure)
			}
			probe.failSync, probe.failFree = false, false
			if owner != nil {
				if err := owner.Close(); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}
