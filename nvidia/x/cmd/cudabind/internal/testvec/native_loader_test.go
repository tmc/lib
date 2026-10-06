//go:build native

package testvec

import (
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// loadWitnessOwnedModule retains either the raw module or its transferred
// owner, never both. It supports the generated global-variable consumers too.
func loadWitnessOwnedModule(t *testing.T, dev *cuda.Device, cm *nvrtcx.CompiledModule) (*vecModule, nvrtcx.OutputKind, error) {
	t.Helper()
	const kind = nvrtcx.OutputCUBIN
	if _, _, err := witnessImage(cm, kind); err != nil {
		return nil, kind, err
	}
	raw, _, err := cm.LoadOutputWithLog(dev, kind)
	pending := raw
	var owner *cuda.Module
	t.Cleanup(func() {
		if owner != nil {
			if err := owner.Close(); err != nil {
				t.Error(err)
			}
			return
		}
		if pending != 0 {
			if err := dev.WithContext(func() error {
				if err := pending.Close(); err != nil {
					return err
				}
				pending = 0
				return nil
			}); err != nil {
				t.Error(err)
			}
		}
	})
	if err != nil {
		return nil, kind, err
	}
	if raw == 0 {
		return nil, kind, fmt.Errorf("witness module: successful load returned no owner")
	}
	owner, err = cuda.OwnModule(dev, raw)
	if err != nil {
		return nil, kind, err
	}
	pending = 0
	m := &vecModule{dev: dev, mod: owner, raw: raw}
	for i, k := range vecModuleKernels {
		name := k.name
		if k.cxx {
			var ok bool
			name, ok = cm.LoweredName(k.name)
			if !ok {
				return nil, kind, fmt.Errorf("witness module: missing lowered name %q", k.name)
			}
		}
		m.kernels[i], err = owner.Function(name)
		if err != nil {
			return nil, kind, err
		}
	}
	return m, kind, nil
}

func checkWitnessRoles(got map[string]int) error {
	want := map[string]int{
		"TestKernels":                                             1,
		"TestVecModuleLayout":                                     1,
		"TestWitness_StaleBindingNamesField":                      2,
		"TestWitness_UsesKernelDefines":                           2,
		"TestLaunch_NativeEarlyRefusals":                          1,
		"TestExistingLayoutMutationAudit/baseline":                1,
		"TestExistingLayoutMutationAudit/same_layout_wider_field": 1,
		"TestExistingLayoutMutationAudit/packing":                 1,
		"TestExistingLayoutMutationAudit/nested_vector_alignment": 1,
	}
	for i := 0; i < 8; i++ {
		want[fmt.Sprintf("TestVecModuleParameterABI%d", i)] = 1
	}
	if len(got) != len(want) {
		return fmt.Errorf("witness receipt: actual compilation roles %d, require %d", len(got), len(want))
	}
	for name, n := range want {
		if got[name] != n {
			return fmt.Errorf("witness receipt: actual compilation role %s has %d, require %d", name, got[name], n)
		}
	}
	return nil
}
