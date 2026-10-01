package nvrtcx

import (
	"errors"
	"testing"
)

type versionGateDevice struct{}

func (versionGateDevice) ComputeCapability() (int, int, error) { return 9, 0, nil }

func TestSpliceAndCompileValidatesNVRTCVersionBeforeCompile(t *testing.T) {
	versionCalled := false
	compileCalled := false
	_, err := spliceAndCompile(versionGateDevice{}, "{{KERNEL}}", map[string]string{"KERNEL": "k"}, nil,
		func() (int32, int32, error) {
			versionCalled = true
			return 11, 7, nil
		},
		func(string, ...Option) (*CompiledModule, error) {
			compileCalled = true
			return nil, nil
		},
	)
	if !versionCalled || compileCalled || !errors.Is(err, ErrJITFusionCompilerUnsupported) {
		t.Fatalf("version gate calls: version=%t compile=%t err=%v", versionCalled, compileCalled, err)
	}
}
