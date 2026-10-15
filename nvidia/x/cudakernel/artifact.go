package cudakernel

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"strings"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// Kind represents the format of a compiled CUDA kernel artifact.
type Kind uint8

const (
	// KindPTX is human-readable or virtual PTX intermediate assembly text.
	KindPTX Kind = iota + 1

	// KindCUBIN is a native ELF executable binary containing machine SASS instructions.
	KindCUBIN

	// KindLTOIR is link-time optimization intermediate representation.
	// It cannot be loaded directly and must be linked with nvjitlink first.
	KindLTOIR

	// KindTileIR is CUDA Tile intermediate representation.
	KindTileIR
)

func (k Kind) String() string {
	switch k {
	case KindPTX:
		return "PTX"
	case KindCUBIN:
		return "CUBIN"
	case KindLTOIR:
		return "LTOIR"
	case KindTileIR:
		return "TileIR"
	default:
		return fmt.Sprintf("Kind(%d)", uint8(k))
	}
}

// RequiresLink reports whether an artifact must be linked via nvjitlink before loading.
func (k Kind) RequiresLink() bool {
	return k == KindLTOIR
}

// JITOptions specifies driver JIT compilation options applied when loading PTX.
type JITOptions struct {
	MaxRegisters    uint32 `json:"max_registers,omitempty"`
	ThreadsPerBlock uint32 `json:"threads_per_block,omitempty"`
	OptLevel        uint32 `json:"opt_level,omitempty"` // 0 through 4
	// OptLevelSet distinguishes explicit level zero from the driver default.
	// A positive OptLevel is explicit whether or not OptLevelSet is true.
	OptLevelSet bool `json:"opt_level_set,omitempty"`
}

// Validate checks the supported driver optimization level. Unset options use
// the driver default; explicit zero requires OptLevelSet.
func (o JITOptions) Validate() error {
	if o.OptLevel > 4 {
		return fmt.Errorf("cudakernel: invalid JIT optimization level %d", o.OptLevel)
	}
	return nil
}

// DriverOptions translates JITOptions into driver.JITOption flags and values
// suitable for cuModuleLoadDataEx. Call Validate before passing them to a
// driver. Explicit zero emits an option; unset leaves the driver default.
func (o JITOptions) DriverOptions() ([]driver.JITOption, []uintptr) {
	var opts []driver.JITOption
	var vals []uintptr

	if o.MaxRegisters > 0 {
		opts = append(opts, driver.JITMaxRegisters)
		vals = append(vals, uintptr(o.MaxRegisters))
	}
	if o.ThreadsPerBlock > 0 {
		opts = append(opts, driver.JITThreadsPerBlock)
		vals = append(vals, uintptr(o.ThreadsPerBlock))
	}
	if o.OptLevelSet || o.OptLevel > 0 {
		opts = append(opts, driver.JITOptimizationLevel)
		vals = append(vals, uintptr(o.OptLevel))
	}
	return opts, vals
}

// ResourceRequirements records hardware resources and execution bounds
// required by a kernel.
type ResourceRequirements struct {
	StaticSharedMemory  uint32 `json:"static_shared_memory"`
	DynamicSharedMemory uint32 `json:"dynamic_shared_memory,omitempty"`
	AllocatedRegisters  uint32 `json:"allocated_registers"`
	PeakLiveGPR         uint32 `json:"peak_live_gpr"`
	RegisterQuantum     uint32 `json:"register_quantum,omitempty"`
	TMEMBytes           uint32 `json:"tmem_bytes,omitempty"`
	ExecutionGroup      string `json:"execution_group,omitempty"`
}

// Dependency records external inputs (such as headers) affecting compilation identity.
type Dependency struct {
	Path     string `json:"path"`
	Digest   string `json:"digest,omitempty"`
	Tracked  bool   `json:"tracked"`
	ErrorMsg string `json:"error_msg,omitempty"`
}

// Artifact is an immutable descriptor for a compiled kernel image and its ABI.
type Artifact struct {
	Kind         Kind                 `json:"kind"`
	Digest       string               `json:"digest"`
	Raw          []byte               `json:"raw"`
	EntryPoint   string               `json:"entry_point"`
	Schema       Schema               `json:"schema"`
	Target       string               `json:"target"`
	Resources    ResourceRequirements `json:"resources"`
	JITOptions   JITOptions           `json:"jit_options,omitempty"`
	Toolchain    string               `json:"toolchain,omitempty"`
	CompilerOpts []string             `json:"compiler_opts,omitempty"`
	Dependencies []Dependency         `json:"dependencies,omitempty"`
}

// ArtifactOption configures optional fields of an Artifact.
type ArtifactOption func(*Artifact)

// WithResources sets resource requirements on an Artifact.
func WithResources(res ResourceRequirements) ArtifactOption {
	return func(a *Artifact) {
		a.Resources = res
	}
}

// WithJIT sets JITOptions on an Artifact.
func WithJIT(jit JITOptions) ArtifactOption {
	return func(a *Artifact) {
		a.JITOptions = jit
	}
}

// WithToolchain records the toolchain name and version on an Artifact.
func WithToolchain(toolchain string) ArtifactOption {
	return func(a *Artifact) {
		a.Toolchain = toolchain
	}
}

// WithCompilerOptions records the compilation options used to produce an Artifact.
func WithCompilerOptions(opts []string) ArtifactOption {
	return func(a *Artifact) {
		a.CompilerOpts = append([]string(nil), opts...)
	}
}

// WithDependencies records external dependencies and their validation state.
func WithDependencies(deps []Dependency) ArtifactOption {
	return func(a *Artifact) {
		a.Dependencies = append([]Dependency(nil), deps...)
	}
}

// NewArtifact creates an immutable Artifact descriptor and validates its integrity.
func NewArtifact(kind Kind, raw []byte, entryPoint string, schema Schema, target string, opts ...ArtifactOption) (*Artifact, error) {
	if len(raw) == 0 {
		return nil, errors.New("cudakernel: empty artifact bytes")
	}
	if entryPoint == "" {
		return nil, errors.New("cudakernel: empty entry point")
	}
	if strings.ContainsRune(entryPoint, 0) {
		return nil, errors.New("cudakernel: entry point contains NUL")
	}
	if target == "" {
		return nil, errors.New("cudakernel: empty target architecture")
	}

	h := sha256.Sum256(raw)
	digest := hex.EncodeToString(h[:])

	a := &Artifact{
		Kind:       kind,
		Digest:     digest,
		Raw:        append([]byte(nil), raw...),
		EntryPoint: entryPoint,
		Schema:     schema,
		Target:     target,
	}

	for _, opt := range opts {
		if opt != nil {
			opt(a)
		}
	}

	if err := a.Validate(); err != nil {
		return nil, fmt.Errorf("cudakernel: validate artifact: %w", err)
	}

	return a, nil
}

// Bytes returns a safe copy of the raw artifact bytes.
func (a *Artifact) Bytes() []byte {
	if a == nil || len(a.Raw) == 0 {
		return nil
	}
	return append([]byte(nil), a.Raw...)
}

// RequiresLink reports whether this artifact must be linked prior to execution.
func (a *Artifact) RequiresLink() bool {
	if a == nil {
		return false
	}
	return a.Kind.RequiresLink()
}

// Validate verifies artifact integrity, schema parameters, ELF format (for CUBIN),
// and target compatibility.
func (a *Artifact) Validate() error {
	if a == nil {
		return errors.New("cudakernel: nil artifact")
	}
	if err := a.JITOptions.Validate(); err != nil {
		return err
	}
	if len(a.Raw) == 0 {
		return errors.New("cudakernel: empty raw bytes")
	}

	// Verify cryptographic digest.
	sum := sha256.Sum256(a.Raw)
	computedDigest := hex.EncodeToString(sum[:])
	if a.Digest != computedDigest {
		return fmt.Errorf("cudakernel: corrupted artifact: digest mismatch (want %s, got %s)", a.Digest, computedDigest)
	}

	if a.EntryPoint == "" {
		return errors.New("cudakernel: missing entry point")
	}

	if err := a.Schema.Validate(); err != nil {
		return fmt.Errorf("cudakernel: invalid schema: %w", err)
	}

	if a.Target == "" {
		return errors.New("cudakernel: missing target architecture")
	}

	// Validate target compatibility rules.
	if err := ValidateTargetArch(a.Target); err != nil {
		return fmt.Errorf("cudakernel: invalid target: %w", err)
	}

	// For CUBIN artifacts, validate ELF structure.
	if a.Kind == KindCUBIN {
		info, err := ParseCUBIN(a.Raw)
		if err != nil {
			return fmt.Errorf("cudakernel: invalid cubin elf: %w", err)
		}
		// If entry point was specified, verify it matches or is present in symbols when symbols are extracted.
		if len(info.Symbols) > 0 {
			found := false
			for _, sym := range info.Symbols {
				if sym == a.EntryPoint || strings.Contains(sym, a.EntryPoint) {
					found = true
					break
				}
			}
			if !found {
				return fmt.Errorf("cudakernel: entry point %q not found in CUBIN symbols", a.EntryPoint)
			}
		}
	}

	// Validate resource limits against target specifications.
	//
	// Both checks are skipped for a target nothing has measured. A limit
	// invented for an unknown architecture does not become safe by being
	// enforced: the estimated 48 KB would reject a kernel that a newer GPU
	// runs comfortably, and the unmeasured TMEM flag would reject Tensor
	// Memory on the first part that has more of it. Refusing to guess here
	// costs a check that ptxas and the driver both repeat; guessing costs a
	// false rejection nobody can explain.
	targetInfo, err := TargetSM(a.Target)
	if err == nil && !targetInfo.Estimated {
		if a.Resources.StaticSharedMemory > targetInfo.MaxSharedMemoryBytes {
			return fmt.Errorf("cudakernel: static shared memory (%d B) exceeds target %s limit (%d B)",
				a.Resources.StaticSharedMemory, a.Target, targetInfo.MaxSharedMemoryBytes)
		}
		if a.Resources.TMEMBytes > 0 && !targetInfo.SupportsTMEM {
			return fmt.Errorf("cudakernel: target %s does not support TMEM (requested %d B)", a.Target, a.Resources.TMEMBytes)
		}
	}

	return nil
}

// ValidateTarget checks whether the artifact can execute on a device targeting deviceArch.
func (a *Artifact) ValidateTarget(deviceArch string) error {
	if a == nil {
		return errors.New("cudakernel: nil artifact")
	}
	return ValidateTargetCompatibility(a.Target, deviceArch)
}

// Load loads the compiled module onto the provided CUDA device.
// Attempting to load an unlinked artifact (e.g. LTOIR) returns an explicit error.
// A nonzero module belongs to the caller even when an error is returned, for
// example when restoring the previous context fails after loading. The caller
// must unload it in dev's context before closing dev.
func (a *Artifact) Load(dev *cuda.Device) (driver.Module, error) {
	if a == nil {
		return 0, errors.New("cudakernel: nil artifact")
	}
	if err := a.JITOptions.Validate(); err != nil {
		return 0, err
	}
	if a.RequiresLink() {
		return 0, fmt.Errorf("cudakernel: artifact kind %s requires linking before loading", a.Kind)
	}
	if dev == nil {
		return 0, errors.New("cudakernel: nil device passed to Load")
	}

	mod, err := loadInContext(dev.WithContext, func() (driver.Module, error) {
		var mod driver.Module
		var loadErr error
		switch a.Kind {
		case KindPTX:
			ptxBytes := append(bytes.TrimRight(a.Raw, "\x00"), 0)
			opts, vals := a.JITOptions.DriverOptions()
			if len(opts) > 0 {
				mod, loadErr = driver.ModuleLoadDataEx(ptxBytes, uint32(len(opts)), &opts[0], &vals[0])
			} else {
				mod, loadErr = driver.ModuleLoadData(ptxBytes)
			}
			runtime.KeepAlive(ptxBytes)
		case KindCUBIN, KindTileIR:
			rawBytes := append([]byte(nil), a.Raw...)
			mod, loadErr = driver.ModuleLoadData(rawBytes)
			runtime.KeepAlive(rawBytes)
		default:
			return 0, fmt.Errorf("cudakernel: artifact kind %s cannot be loaded directly", a.Kind)
		}
		return mod, loadErr
	})
	if err != nil {
		return mod, fmt.Errorf("cudakernel: load module on %s: %w", a.Target, err)
	}
	return mod, nil
}

func loadInContext(run func(func() error) error, load func() (driver.Module, error)) (driver.Module, error) {
	var mod driver.Module
	err := run(func() error {
		var err error
		mod, err = load()
		return err
	})
	return mod, err
}

// MarshalJSON serializes the Artifact into JSON.
func (a *Artifact) MarshalJSON() ([]byte, error) {
	type Alias Artifact
	return json.Marshal(&struct {
		*Alias
	}{
		Alias: (*Alias)(a),
	})
}

// UnmarshalJSON deserializes an Artifact from JSON and validates it.
func (a *Artifact) UnmarshalJSON(data []byte) error {
	type Alias Artifact
	aux := &struct {
		*Alias
	}{
		Alias: (*Alias)(a),
	}
	if err := json.Unmarshal(data, aux); err != nil {
		return fmt.Errorf("cudakernel: unmarshal artifact: %w", err)
	}
	return a.Validate()
}
