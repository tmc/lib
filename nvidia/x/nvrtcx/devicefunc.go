package nvrtcx

import (
	"errors"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudair"
)

// MaxDefaultDynamicSharedSizeBytes is the 48 KiB default limit on dynamic
// shared memory per block defined by the CUDA architecture specification.
const MaxDefaultDynamicSharedSizeBytes = 49152 // 48 KiB

var (
	// ErrJITFusionUnsupported reports that a target CUDA device does not support
	// JIT fusion (compute capability < sm_75).
	ErrJITFusionUnsupported = fmt.Errorf("nvrtcx: JIT fusion unsupported: %w", cudaerrors.ErrUnsupported)

	// ErrSharedMemOptInRequired reports that requested dynamic shared memory
	// exceeds 48 KiB without the required CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES opt-in.
	ErrSharedMemOptInRequired       = errors.New("nvrtcx: dynamic shared memory exceeds 48 KiB default limit without opt-in")
	ErrJITFusionCompilerUnsupported = fmt.Errorf("nvrtcx: JIT fusion requires NVRTC 11.8 or newer: %w", cudaerrors.ErrUnsupported)
)

// Device represents a CUDA device capable of reporting compute capability.
type Device interface {
	ComputeCapability() (major, minor int, err error)
}

// FunctionAttributeSetter represents a CUDA kernel or function whose attributes
// can be set before launch.
type FunctionAttributeSetter interface {
	SetAttribute(attrib driver.FunctionAttribute, value int32) error
}

// SupportsJITFusion reports whether dev supports runtime JIT kernel fusion.
// Matching tiny-cuda-nn rtc_kernel.cu:91, JIT fusion requires Turing or higher
// (compute capability >= 7.5 / sm_75). On unsupported devices, callers receive
// a typed refusal and must not attempt fallback compilation.
func SupportsJITFusion(dev Device) (bool, error) {
	if dev == nil {
		return false, fmt.Errorf("nvrtcx: nil device")
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return false, fmt.Errorf("nvrtcx: compute capability: %w", err)
	}
	cc := major*10 + minor
	if cc < 75 {
		return false, nil
	}
	return true, nil
}

// PrepareDeviceFuncLaunch opts the loaded function into Unit's shared-memory
// request before launch. A missing setter or a rejected opt-in is a typed error.
func PrepareDeviceFuncLaunch(fn FunctionAttributeSetter, unit cudair.Unit) error {
	if unit.ShmemBytes < 0 {
		return fmt.Errorf("nvrtcx: negative dynamic shared memory: %d", unit.ShmemBytes)
	}
	if unit.ShmemBytes <= MaxDefaultDynamicSharedSizeBytes {
		return nil
	}
	if fn == nil {
		return fmt.Errorf("%w: loaded function cannot set attributes", ErrSharedMemOptInRequired)
	}
	if unit.ShmemBytes > math.MaxInt32 {
		return fmt.Errorf("%w: requested %d bytes exceeds driver attribute range", ErrSharedMemOptInRequired, unit.ShmemBytes)
	}
	if err := fn.SetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes, int32(unit.ShmemBytes)); err != nil {
		return fmt.Errorf("%w: requested %d bytes: %w", ErrSharedMemOptInRequired, unit.ShmemBytes, err)
	}
	return nil
}

// minimumFusionCompilerVersion is the minimum NVRTC version accepted by the
// JIT-fusion route in the CUDA reference path.
const minimumFusionCompilerMajor = 11
const minimumFusionCompilerMinor = 8

// ValidateJITFusionCompilerVersion checks the compiler version required by the
// JIT-fusion route.
func ValidateJITFusionCompilerVersion(major, minor int) error {
	if major < minimumFusionCompilerMajor || (major == minimumFusionCompilerMajor && minor < minimumFusionCompilerMinor) {
		return fmt.Errorf("%w: found NVRTC %d.%d", ErrJITFusionCompilerUnsupported, major, minor)
	}
	return nil
}

// templatePlaceholder matches the reserved {{IDENT}} template form.
var templatePlaceholder = regexp.MustCompile(`\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}`)

// SpliceTemplate substitutes reserved {{IDENT}} placeholders into an outer
// kernel template, matching tiny-cuda-nn trainer.h:237-239. Other brace forms
// remain byte-for-byte literal, and inserted values are not rescanned. Since
// {{IDENT}} is reserved, write a C++ nested-brace initializer such as {{x}} as
// { {x} }.
//
// Requirements:
//   - An unfilled placeholder is an error naming it.
//   - A placeholder filled with the empty string is legal and produces valid code (the has_external_dL_dy case).
//   - A substitution key with no corresponding placeholder is an error, not a silent no-op.
func SpliceTemplate(tmpl string, subs map[string]string) (string, error) {
	// Find placeholders only in the original template. Replacement bytes are
	// copied directly and are never interpreted as more template syntax.
	matches := templatePlaceholder.FindAllStringSubmatchIndex(tmpl, -1)
	placeholders := make(map[string]bool, len(matches))
	for _, match := range matches {
		placeholders[tmpl[match[2]:match[3]]] = true
	}

	// 1. Check for substitution keys with no corresponding placeholder
	var extraKeys []string
	for k := range subs {
		if !placeholders[k] {
			extraKeys = append(extraKeys, k)
		}
	}
	if len(extraKeys) > 0 {
		sort.Strings(extraKeys)
		return "", fmt.Errorf("nvrtcx: substitution key %q has no corresponding placeholder in template", extraKeys[0])
	}

	// 2. Check for unfilled placeholders in template
	var missingKeys []string
	for p := range placeholders {
		if _, ok := subs[p]; !ok {
			missingKeys = append(missingKeys, p)
		}
	}
	if len(missingKeys) > 0 {
		sort.Strings(missingKeys)
		return "", fmt.Errorf("nvrtcx: unfilled placeholder %q in template", missingKeys[0])
	}

	// Copy literal spans and replacements in one left-to-right pass.
	var out strings.Builder
	last := 0
	for _, match := range matches {
		out.WriteString(tmpl[last:match[0]])
		out.WriteString(subs[tmpl[match[2]:match[3]]])
		last = match[1]
	}
	out.WriteString(tmpl[last:])
	return out.String(), nil
}

// SpliceAndCompile checks that dev supports JIT fusion, splices subs into tmpl,
// and compiles the result with NVRTC. If dev reports SupportsJITFusion() == false,
// it returns ErrJITFusionUnsupported with no fallback compilation.
func SpliceAndCompile(dev Device, tmpl string, subs map[string]string, opts ...Option) (*CompiledModule, error) {
	return spliceAndCompile(dev, tmpl, subs, opts, nvrtc.Version, Compile)
}

func spliceAndCompile(dev Device, tmpl string, subs map[string]string, opts []Option,
	version func() (int32, int32, error), compile func(string, ...Option) (*CompiledModule, error),
) (*CompiledModule, error) {
	ok, err := SupportsJITFusion(dev)
	if err != nil {
		return nil, fmt.Errorf("nvrtcx: check JIT fusion support: %w", err)
	}
	if !ok {
		return nil, fmt.Errorf("%w: device does not meet sm_75 requirement", ErrJITFusionUnsupported)
	}
	compilerMajor, compilerMinor, err := version()
	if err != nil {
		return nil, fmt.Errorf("nvrtcx: query NVRTC version for JIT fusion: %w", err)
	}
	if err := ValidateJITFusionCompilerVersion(int(compilerMajor), int(compilerMinor)); err != nil {
		return nil, err
	}

	src, err := SpliceTemplate(tmpl, subs)
	if err != nil {
		return nil, fmt.Errorf("nvrtcx: splice template: %w", err)
	}

	return compile(src, opts...)
}

// Spliced units pass through the regular Compile cache. Its key covers source,
// program name, ordered compiler flags, in-memory headers, name expressions,
// tracked external dependency contents, NVRTC version and path, output kind,
// and cache format. Untracked external dependencies disable caching. JIT
// options are not part of that key; SpliceAndCompile forwards them, so calls
// with the same compile inputs but different JIT options can share the cached
// CompiledModule and its stored JIT options. See nvrtcx package docs and the
// Lane 40 finding in the implementation handoff directory.
