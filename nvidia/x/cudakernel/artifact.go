package cudakernel

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"runtime"
	"slices"
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

// WorkspaceRequirements declares per-dispatch global device scratch. It is a
// caller declaration, not a measurement or a property authenticated by Digest.
// The zero value requires no scratch. Positive Bytes names exactly one ptr
// parameter in the artifact schema. Alignment applies to the device address,
// not to the parameter-bank slot. Contents are unspecified on entry and are
// private to the dispatch until completion; inputs and outputs must not alias it.
type WorkspaceRequirements struct {
	Bytes     uint64 `json:"bytes"`
	Alignment uint32 `json:"alignment"`
	Parameter string `json:"parameter"`
}

// Validate checks the declaration without allocating storage or resolving a
// schema. Artifact.Validate additionally checks the named device-pointer slot.
func (w WorkspaceRequirements) Validate() error {
	if w.Bytes == 0 {
		if w.Alignment != 0 || w.Parameter != "" {
			return errors.New("cudakernel: zero workspace requires zero alignment and empty parameter")
		}
		return nil
	}
	if w.Alignment == 0 || w.Alignment&(w.Alignment-1) != 0 {
		return errors.New("cudakernel: workspace alignment must be a positive power of two")
	}
	if w.Parameter == "" || strings.ContainsRune(w.Parameter, 0) {
		return errors.New("cudakernel: workspace requires a named device-pointer parameter")
	}
	return nil
}

func validateWorkspaceSchema(w WorkspaceRequirements, schema Schema) error {
	if err := w.Validate(); err != nil {
		return err
	}
	if w.Bytes == 0 {
		return nil
	}
	for _, p := range schema.Params {
		if p.Name == w.Parameter {
			if !p.IsPointer || p.Type != "ptr" || p.ByteWidth != 8 {
				return errors.New("cudakernel: workspace parameter must be an eight-byte device pointer")
			}
			return nil
		}
	}
	return fmt.Errorf("cudakernel: workspace parameter %q is absent from schema", w.Parameter)
}

// ResourceRequirements records hardware resources and execution bounds
// required by a kernel.
type ResourceRequirements struct {
	Workspace           WorkspaceRequirements `json:"workspace,omitzero"`
	StaticSharedMemory  uint32                `json:"static_shared_memory"`
	DynamicSharedMemory uint32                `json:"dynamic_shared_memory,omitempty"`
	AllocatedRegisters  uint32                `json:"allocated_registers"`
	PeakLiveGPR         uint32                `json:"peak_live_gpr"`
	RegisterQuantum     uint32                `json:"register_quantum,omitempty"`
	TMEMBytes           uint32                `json:"tmem_bytes,omitempty"`
	ExecutionGroup      string                `json:"execution_group,omitempty"`
}

// Dependency records external inputs (such as headers) affecting compilation identity.
type Dependency struct {
	Path     string `json:"path"`
	Digest   string `json:"digest,omitempty"`
	Tracked  bool   `json:"tracked"`
	ErrorMsg string `json:"error_msg,omitempty"`
}

// ArtifactSpec is mutable construction data for an [Artifact]. Constructors
// copy its slices after applying options. Changing a spec does not change an
// artifact already constructed from it.
type ArtifactSpec struct {
	Kind         Kind                 `json:"kind"`
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

// Artifact is an immutable descriptor for a compiled kernel image and its ABI.
// Its zero value is invalid. Copies share immutable data and compare equal;
// independently constructed artifacts have distinct identities, even when
// their bytes match. Constructors and accessors do not share writable slices
// with callers. Replacing a caller's Artifact value does not change copies
// captured earlier by a registry or executable owner.
type Artifact struct {
	data *artifactData
}

type artifactData struct {
	spec   ArtifactSpec
	digest string
}

// ArtifactOption configures mutable construction data. Options receive a spec,
// not the published Artifact. Retaining that spec does not retain artifact data.
type ArtifactOption func(*ArtifactSpec)

// WithResources sets resource requirements on an Artifact.
func WithResources(res ResourceRequirements) ArtifactOption {
	return func(a *ArtifactSpec) {
		a.Resources = res
	}
}

// WithJIT sets JITOptions on an Artifact.
func WithJIT(jit JITOptions) ArtifactOption {
	return func(a *ArtifactSpec) {
		a.JITOptions = jit
	}
}

// WithToolchain records the toolchain name and version on an Artifact.
func WithToolchain(toolchain string) ArtifactOption {
	return func(a *ArtifactSpec) {
		a.Toolchain = toolchain
	}
}

// WithCompilerOptions records the compilation options used to produce an Artifact.
func WithCompilerOptions(opts []string) ArtifactOption {
	return func(a *ArtifactSpec) {
		a.CompilerOpts = append([]string(nil), opts...)
	}
}

// WithDependencies records external dependencies and their validation state.
func WithDependencies(deps []Dependency) ArtifactOption {
	return func(a *ArtifactSpec) {
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

	a := &ArtifactSpec{
		Kind:       kind,
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

	return NewArtifactFromSpec(*a)
}

// NewArtifactFromSpec copies and validates all construction data. Digest is
// computed from the final bytes, including any changes made by options.
func NewArtifactFromSpec(spec ArtifactSpec) (*Artifact, error) {
	if spec.EntryPoint == "" || strings.ContainsRune(spec.EntryPoint, 0) {
		return nil, errors.New("cudakernel: invalid entry point")
	}
	owned := copyArtifactSpec(spec)
	h := sha256.Sum256(owned.Raw)
	a := &Artifact{data: &artifactData{spec: owned, digest: hex.EncodeToString(h[:])}}
	if err := a.Validate(); err != nil {
		return nil, fmt.Errorf("cudakernel: validate artifact: %w", err)
	}

	return a, nil
}

func copyArtifactSpec(s ArtifactSpec) ArtifactSpec {
	s.Raw = slices.Clone(s.Raw)
	s.Schema.Params = slices.Clone(s.Schema.Params)
	s.CompilerOpts = slices.Clone(s.CompilerOpts)
	s.Dependencies = slices.Clone(s.Dependencies)
	return s
}

// Spec returns a detached copy of all construction data.
func (a Artifact) Spec() ArtifactSpec {
	if a.data == nil {
		return ArtifactSpec{}
	}
	return copyArtifactSpec(a.data.spec)
}

// Kind returns the artifact format, or zero for an invalid artifact.
func (a Artifact) Kind() Kind {
	if a.data == nil {
		return 0
	}
	return a.data.spec.Kind
}

// Digest returns SHA256 of the raw bytes, not of the descriptor metadata.
func (a Artifact) Digest() string {
	if a.data == nil {
		return ""
	}
	return a.data.digest
}

// EntryPoint returns the kernel symbol.
func (a Artifact) EntryPoint() string {
	if a.data == nil {
		return ""
	}
	return a.data.spec.EntryPoint
}

// Target returns the target architecture.
func (a Artifact) Target() string {
	if a.data == nil {
		return ""
	}
	return a.data.spec.Target
}

// Resources returns the resource requirements by value.
func (a Artifact) Resources() ResourceRequirements {
	if a.data == nil {
		return ResourceRequirements{}
	}
	return a.data.spec.Resources
}

// JITOptions returns the driver load options by value.
func (a Artifact) JITOptions() JITOptions {
	if a.data == nil {
		return JITOptions{}
	}
	return a.data.spec.JITOptions
}

// Toolchain returns the recorded compiler identity.
func (a Artifact) Toolchain() string {
	if a.data == nil {
		return ""
	}
	return a.data.spec.Toolchain
}

// Schema returns a detached parameter schema.
func (a Artifact) Schema() Schema {
	if a.data == nil {
		return Schema{}
	}
	s := a.data.spec.Schema
	s.Params = slices.Clone(s.Params)
	return s
}

// CompilerOptions returns a detached compilation-option slice.
func (a Artifact) CompilerOptions() []string {
	if a.data == nil {
		return nil
	}
	return slices.Clone(a.data.spec.CompilerOpts)
}

// Dependencies returns a detached external-input slice.
func (a Artifact) Dependencies() []Dependency {
	if a.data == nil {
		return nil
	}
	return slices.Clone(a.data.spec.Dependencies)
}

// Pack packs arguments against the artifact's immutable schema.
func (a *Artifact) Pack(args ...any) ([]byte, error) {
	if a == nil || a.data == nil {
		return nil, errors.New("cudakernel: invalid artifact")
	}
	owned := *a
	return owned.data.spec.Schema.Pack(args...)
}

// Bytes returns a safe copy of the raw artifact bytes.
func (a *Artifact) Bytes() []byte {
	if a == nil || a.data == nil || len(a.data.spec.Raw) == 0 {
		return nil
	}
	return append([]byte(nil), a.data.spec.Raw...)
}

// RequiresLink reports whether this artifact must be linked prior to execution.
func (a *Artifact) RequiresLink() bool {
	if a == nil {
		return false
	}
	return a.Kind().RequiresLink()
}

// Validate verifies artifact integrity, schema parameters, ELF format (for CUBIN),
// and target compatibility.
func (a *Artifact) Validate() error {
	if a == nil || a.data == nil {
		return errors.New("cudakernel: nil artifact")
	}
	s := a.data.spec
	if err := s.JITOptions.Validate(); err != nil {
		return err
	}
	if len(s.Raw) == 0 {
		return errors.New("cudakernel: empty raw bytes")
	}

	// Verify cryptographic digest.
	sum := sha256.Sum256(s.Raw)
	computedDigest := hex.EncodeToString(sum[:])
	if a.data.digest != computedDigest {
		return fmt.Errorf("cudakernel: corrupted artifact: digest mismatch (want %s, got %s)", a.data.digest, computedDigest)
	}

	if s.EntryPoint == "" {
		return errors.New("cudakernel: missing entry point")
	}
	if strings.ContainsRune(s.EntryPoint, 0) {
		return errors.New("cudakernel: invalid entry point")
	}

	if err := s.Schema.Validate(); err != nil {
		return fmt.Errorf("cudakernel: invalid schema: %w", err)
	}

	if err := validateWorkspaceSchema(s.Resources.Workspace, s.Schema); err != nil {
		return err
	}

	if s.Target == "" {
		return errors.New("cudakernel: missing target architecture")
	}

	// Validate target compatibility rules.
	if err := ValidateTargetArch(s.Target); err != nil {
		return fmt.Errorf("cudakernel: invalid target: %w", err)
	}

	// For CUBIN artifacts, validate ELF structure.
	if s.Kind == KindCUBIN {
		info, err := ParseCUBIN(s.Raw)
		if err != nil {
			return fmt.Errorf("cudakernel: invalid cubin elf: %w", err)
		}
		// Match the exact lookup token when raw entry symbols are available.
		if len(info.Symbols) > 0 {
			found := false
			for _, sym := range info.Symbols {
				if sym == s.EntryPoint {
					found = true
					break
				}
			}
			if !found {
				return fmt.Errorf("cudakernel: entry point %q not found in CUBIN symbols", s.EntryPoint)
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
	targetInfo, err := TargetSM(s.Target)
	if err == nil && !targetInfo.Estimated {
		if s.Resources.StaticSharedMemory > targetInfo.MaxSharedMemoryBytes {
			return fmt.Errorf("cudakernel: static shared memory (%d B) exceeds target %s limit (%d B)",
				s.Resources.StaticSharedMemory, s.Target, targetInfo.MaxSharedMemoryBytes)
		}
		if s.Resources.TMEMBytes > 0 && !targetInfo.SupportsTMEM {
			return fmt.Errorf("cudakernel: target %s does not support TMEM (requested %d B)", s.Target, s.Resources.TMEMBytes)
		}
	}

	return nil
}

// ValidateTarget checks whether the artifact can execute on a device targeting deviceArch.
func (a *Artifact) ValidateTarget(deviceArch string) error {
	if a == nil {
		return errors.New("cudakernel: nil artifact")
	}
	return ValidateTargetCompatibility(a.Target(), deviceArch)
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
	image, err := a.prepareLoad()
	if err != nil {
		return 0, err
	}
	if dev == nil {
		return 0, errors.New("cudakernel: nil device passed to Load")
	}

	mod, err := loadInContext(dev.WithContext, func() (driver.Module, error) {
		var mod driver.Module
		var loadErr error
		switch image.kind {
		case KindPTX:
			if len(image.options) > 0 {
				mod, loadErr = driver.ModuleLoadDataEx(image.raw, uint32(len(image.options)), &image.options[0], &image.values[0])
			} else {
				mod, loadErr = driver.ModuleLoadData(image.raw)
			}
		case KindCUBIN, KindTileIR:
			mod, loadErr = driver.ModuleLoadData(image.raw)
		}
		runtime.KeepAlive(image)
		return mod, loadErr
	})
	if err != nil {
		return mod, fmt.Errorf("cudakernel: load module on %s: %w", image.target, err)
	}
	return mod, nil
}

type loadImage struct {
	kind    Kind
	target  string
	raw     []byte
	options []driver.JITOption
	values  []uintptr
}

// prepareLoad captures one immutable handle and owns every driver input.
func (a Artifact) prepareLoad() (loadImage, error) {
	if a.data == nil {
		return loadImage{}, errors.New("cudakernel: invalid artifact")
	}
	s := a.data.spec
	if err := s.JITOptions.Validate(); err != nil {
		return loadImage{}, err
	}
	if s.Kind.RequiresLink() {
		return loadImage{}, fmt.Errorf("cudakernel: artifact kind %s requires linking before loading", s.Kind)
	}
	image := loadImage{kind: s.Kind, target: s.Target, raw: slices.Clone(s.Raw)}
	switch s.Kind {
	case KindPTX:
		image.raw = append(bytes.TrimRight(image.raw, "\x00"), 0)
		image.options, image.values = s.JITOptions.DriverOptions()
	case KindCUBIN, KindTileIR:
	default:
		return loadImage{}, fmt.Errorf("cudakernel: artifact kind %s cannot be loaded directly", s.Kind)
	}
	return image, nil
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
func (a Artifact) MarshalJSON() ([]byte, error) {
	if a.data == nil {
		return nil, errors.New("cudakernel: invalid artifact")
	}
	return json.Marshal(newLegacyArtifactWire(a.data.spec, a.data.digest))
}

type artifactWire struct {
	ArtifactSpec
	Digest string `json:"digest"`
}

// legacyArtifactWire is the JSON form MarshalJSON writes and ParseArtifact
// reads. Its fields and tags keep the order of the former exported Artifact
// struct, so persisted bytes and the keys derived from them do not change.
type legacyArtifactWire struct {
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

func newLegacyArtifactWire(spec ArtifactSpec, digest string) legacyArtifactWire {
	return legacyArtifactWire{
		Kind:         spec.Kind,
		Digest:       digest,
		Raw:          spec.Raw,
		EntryPoint:   spec.EntryPoint,
		Schema:       spec.Schema,
		Target:       spec.Target,
		Resources:    spec.Resources,
		JITOptions:   spec.JITOptions,
		Toolchain:    spec.Toolchain,
		CompilerOpts: spec.CompilerOpts,
		Dependencies: spec.Dependencies,
	}
}

func (w legacyArtifactWire) spec() ArtifactSpec {
	return ArtifactSpec{
		Kind:         w.Kind,
		Raw:          w.Raw,
		EntryPoint:   w.EntryPoint,
		Schema:       w.Schema,
		Target:       w.Target,
		Resources:    w.Resources,
		JITOptions:   w.JITOptions,
		Toolchain:    w.Toolchain,
		CompilerOpts: w.CompilerOpts,
		Dependencies: w.Dependencies,
	}
}

// UnmarshalJSON validates detached wire data before replacing this handle.
// A failure leaves the receiver unchanged; previously captured handle copies
// are unchanged even on success.
func (a *Artifact) UnmarshalJSON(data []byte) error {
	if a == nil {
		return errors.New("cudakernel: nil artifact")
	}
	var wire artifactWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return fmt.Errorf("cudakernel: unmarshal artifact: %w", err)
	}
	decoded := Artifact{data: &artifactData{spec: copyArtifactSpec(wire.ArtifactSpec), digest: wire.Digest}}
	if err := decoded.Validate(); err != nil {
		return err
	}
	*a = decoded
	return nil
}

// ParseArtifact decodes and validates one artifact in the JSON form written
// by MarshalJSON, including its recorded digest. Unlike UnmarshalJSON, it
// rejects unknown fields and any data after the artifact.
func ParseArtifact(data []byte) (*Artifact, error) {
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	var wire legacyArtifactWire
	if err := d.Decode(&wire); err != nil {
		return nil, fmt.Errorf("cudakernel: parse artifact: %w", err)
	}
	if _, err := d.Token(); err != io.EOF {
		return nil, errors.New("cudakernel: parse artifact: data after artifact")
	}
	a := &Artifact{data: &artifactData{spec: wire.spec(), digest: wire.Digest}}
	if err := a.Validate(); err != nil {
		return nil, err
	}
	return a, nil
}

// Size returns the length of the raw artifact bytes without copying them,
// or zero for an invalid artifact.
func (a *Artifact) Size() int {
	if a == nil || a.data == nil {
		return 0
	}
	return len(a.data.spec.Raw)
}
