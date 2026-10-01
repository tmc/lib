package nvrtcx_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestSpliceTemplate_EmptySubstitutionIsLegal(t *testing.T) {
	tmpl := `{{FORWARD_DEVICE_FUNCTION}}
{{LOSS_DEVICE_FUNCTION}}
__global__ void {{KERNEL_NAME}}() {}`

	subs := map[string]string{
		"FORWARD_DEVICE_FUNCTION": "__device__ void fwd() {}",
		"LOSS_DEVICE_FUNCTION":    "", // Empty string is legal per trainer.h:239
		"KERNEL_NAME":             "eval_kernel",
	}

	result, err := nvrtcx.SpliceTemplate(tmpl, subs)
	if err != nil {
		t.Fatalf("SpliceTemplate failed with empty substitution: %v", err)
	}

	if !strings.Contains(result, "__device__ void fwd() {}") {
		t.Fatalf("expected result to contain forward function, got:\n%s", result)
	}
	if !strings.Contains(result, "__global__ void eval_kernel() {}") {
		t.Fatalf("expected result to contain kernel name, got:\n%s", result)
	}
	if strings.Contains(result, "{LOSS_DEVICE_FUNCTION}") {
		t.Fatalf("loss placeholder was not replaced, got:\n%s", result)
	}

	// Also test extra substitution key error
	extraSubs := map[string]string{
		"FORWARD_DEVICE_FUNCTION": "__device__ void fwd() {}",
		"LOSS_DEVICE_FUNCTION":    "",
		"KERNEL_NAME":             "eval_kernel",
		"EXTRA_KEY":               "extra",
	}
	_, err = nvrtcx.SpliceTemplate(tmpl, extraSubs)
	if err == nil {
		t.Fatalf("expected error for extraneous substitution key, got nil")
	}
	if !strings.Contains(err.Error(), "EXTRA_KEY") {
		t.Fatalf("expected error to name 'EXTRA_KEY', got: %v", err)
	}
}

func TestSpliceTemplate_UnfilledPlaceholder_Errors(t *testing.T) {
	tmpl := `__global__ void {{KERNEL_NAME}}() {
	{{UNFILLED_BODY}}
}`

	subs := map[string]string{
		"KERNEL_NAME": "my_kernel",
	}

	_, err := nvrtcx.SpliceTemplate(tmpl, subs)
	if err == nil {
		t.Fatalf("expected error for unfilled placeholder, got nil")
	}
	if !strings.Contains(err.Error(), "UNFILLED_BODY") {
		t.Fatalf("expected error to name 'UNFILLED_BODY', got: %v", err)
	}
}

func TestSpliceTemplate_PreservesBraces(t *testing.T) {
	tmpl := "before }} {{1,2}} {{VALUE}} after"
	value := "x}}y {{1,2}} z"
	got, err := nvrtcx.SpliceTemplate(tmpl, map[string]string{"VALUE": value})
	if err != nil {
		t.Fatal(err)
	}
	want := "before }} {{1,2}} " + value + " after"
	if got != want {
		t.Fatalf("splice changed braces:\n got %q\nwant %q", got, want)
	}
}

func TestSpliceTemplate_NoRescan(t *testing.T) {
	for i := 0; i < 200; i++ {
		got, err := nvrtcx.SpliceTemplate("{{A}} {{B}}", map[string]string{"A": "x{{B}}x", "B": "substituted"})
		if err != nil {
			t.Fatal(err)
		}
		if got != "x{{B}}x substituted" {
			t.Fatalf("iteration %d: got %q", i, got)
		}
	}
}

func TestSpliceTemplate_BraceInitIsNotAPlaceholder(t *testing.T) {
	got, err := nvrtcx.SpliceTemplate("float y{x}; {{NAME}}", map[string]string{"NAME": "f"})
	if err != nil {
		t.Fatal(err)
	}
	if got != "float y{x}; f" {
		t.Fatalf("got %q", got)
	}
}

type mockDevice struct {
	major int
	minor int
	err   error
}

func (m *mockDevice) ComputeCapability() (int, int, error) {
	return m.major, m.minor, m.err
}

func TestSupportsJITFusion_FalseDeviceRefusesNoFallback(t *testing.T) {
	tests := []struct {
		name      string
		major     int
		minor     int
		wantAllow bool
	}{
		{"Volta_SM70", 7, 0, false},
		{"Pascal_SM61", 6, 1, false},
		{"Maxwell_SM50", 5, 0, false},
		{"Turing_SM75", 7, 5, true},
		{"Ampere_SM80", 8, 0, true},
		{"Hopper_SM90", 9, 0, true},
		{"Blackwell_SM121", 12, 1, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dev := &mockDevice{major: tt.major, minor: tt.minor}
			allowed, err := nvrtcx.SupportsJITFusion(dev)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if allowed != tt.wantAllow {
				t.Fatalf("got %v, want %v", allowed, tt.wantAllow)
			}
		})
	}

	// Refusal without fallback test
	refusedDev := &mockDevice{major: 7, minor: 0}
	_, err := nvrtcx.SpliceAndCompile(refusedDev, "__global__ void {NAME}() {}", map[string]string{"NAME": "foo"})
	if err == nil {
		t.Fatalf("expected SpliceAndCompile to refuse unsupported device, got nil")
	}
	if !errors.Is(err, nvrtcx.ErrJITFusionUnsupported) {
		t.Fatalf("expected ErrJITFusionUnsupported, got: %v", err)
	}
}

func TestSupportsJITFusion_ComputeCapabilityError(t *testing.T) {
	dev := &mockDevice{err: errors.New("device query failed")}
	if _, err := nvrtcx.SupportsJITFusion(dev); err == nil || !strings.Contains(err.Error(), "device query failed") {
		t.Fatalf("SupportsJITFusion error = %v", err)
	}
	if _, err := nvrtcx.SpliceAndCompile(dev, "__global__ void {{KERNEL}}() {}", map[string]string{"KERNEL": "k"}); err == nil || !strings.Contains(err.Error(), "device query failed") {
		t.Fatalf("SpliceAndCompile did not stop at the capability error: %v", err)
	}
}

func TestJITFusionRequiresNVRTC118(t *testing.T) {
	for _, tc := range []struct {
		major, minor int
		wantErr      bool
	}{{11, 7, true}, {11, 8, false}, {12, 0, false}} {
		err := nvrtcx.ValidateJITFusionCompilerVersion(tc.major, tc.minor)
		if (err != nil) != tc.wantErr {
			t.Fatalf("NVRTC %d.%d error = %v", tc.major, tc.minor, err)
		}
	}
}

type mockFuncSetter struct {
	lastAttr  driver.FunctionAttribute
	lastValue int32
	called    bool
	err       error
}

type sharedEmitter struct {
	name  string
	bytes int
}

func (s *sharedEmitter) Name() string                  { return s.name }
func (s *sharedEmitter) InputWidth() int               { return 1 }
func (s *sharedEmitter) OutputWidth() int              { return 1 }
func (s *sharedEmitter) PaddedOutputWidth() int        { return 1 }
func (s *sharedEmitter) FwdCtxBytes() (int, error)     { return 0, nil }
func (s *sharedEmitter) FwdCtxAlignedPerElement() bool { return true }
func (s *sharedEmitter) ShmemBytes() int               { return s.bytes }
func (s *sharedEmitter) EmitForward(name string) (string, error) {
	return cudair.EmitFromBody(s, name, "return input;")
}

func (m *mockFuncSetter) SetAttribute(attrib driver.FunctionAttribute, value int32) error {
	m.called = true
	m.lastAttr = attrib
	m.lastValue = value
	return m.err
}

func TestSharedMemOptIn_Above48KiB(t *testing.T) {
	node := &sharedEmitter{name: "large_shared", bytes: 65536}
	unit, err := cudair.EmitTree(node, "large_shared_eval")
	if err != nil {
		t.Fatal(err)
	}
	if unit.ShmemBytes != 65536 {
		t.Fatalf("EmitTree shmem = %d, want 65536", unit.ShmemBytes)
	}
	if err := nvrtcx.PrepareDeviceFuncLaunch(nil, cudair.Unit{ShmemBytes: 49152}); err != nil {
		t.Fatalf("small request: %v", err)
	}
	err = nvrtcx.PrepareDeviceFuncLaunch(nil, unit)
	if !errors.Is(err, nvrtcx.ErrSharedMemOptInRequired) {
		t.Fatalf("missing setter error = %v", err)
	}

	// Test the launch-side attribute setter with the resource request from a unit.
	setter := &mockFuncSetter{}
	if err := nvrtcx.PrepareDeviceFuncLaunch(setter, unit); err != nil {
		t.Fatalf("PrepareDeviceFuncLaunch failed: %v", err)
	}
	if !setter.called {
		t.Fatalf("expected SetAttribute to be called for 64 KiB")
	}
	if setter.lastAttr != driver.FuncAttributeMaxDynamicSharedSizeBytes {
		t.Fatalf("expected FuncAttributeMaxDynamicSharedSizeBytes (%d), got: %d",
			driver.FuncAttributeMaxDynamicSharedSizeBytes, setter.lastAttr)
	}
	if setter.lastValue != 65536 {
		t.Fatalf("expected value 65536, got %d", setter.lastValue)
	}

	// Small shmem does not call setter.
	smallSetter := &mockFuncSetter{}
	if err := nvrtcx.PrepareDeviceFuncLaunch(smallSetter, cudair.Unit{ShmemBytes: 16384}); err != nil {
		t.Fatalf("PrepareDeviceFuncLaunch failed: %v", err)
	}
	if smallSetter.called {
		t.Fatalf("expected SetAttribute not to be called for 16 KiB")
	}
	failing := &mockFuncSetter{err: errors.New("setter rejected")}
	if err := nvrtcx.PrepareDeviceFuncLaunch(failing, unit); !errors.Is(err, nvrtcx.ErrSharedMemOptInRequired) {
		t.Fatalf("setter failure error = %v", err)
	}
}
