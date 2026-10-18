package cudainspect

import (
	"bytes"
	"context"
	"crypto/sha256"
	"debug/elf"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

// ReportSchemaVersion is the schema version written by [Inspector.Report].
const ReportSchemaVersion = 2

// Input bounds enforced by [Inspector.Report] and [ReadReport].
const (
	DefaultMaxInputBytes = 64 << 20 // also the largest accepted Request.MaxInputBytes
	MaxReportBytes       = 64 << 20
	maxSymbolBytes       = 4096
	maxFeatureNames      = 64
	maxFeatureNameBytes  = 64
)

// Feature states. Only observed and inferred features carry a value.
const (
	StateObserved    = "observed"
	StateInferred    = "inferred"
	StateUnknown     = "unknown"
	StateUnsupported = "unsupported"
	StateError       = "error"
)

// Report results.
const (
	ResultComplete = "complete" // every requested feature has a value
	ResultPartial  = "partial"  // every required feature has a value; an optional one does not
	ResultFailed   = "failed"   // a required feature has no value
)

// Errors returned by [Inspector.Report] and [ReadReport], for use with
// [errors.Is]. Every one but ErrRequiredFeature is a refusal: no report is
// returned with it.
var (
	ErrRequest         = errors.New("cudainspect: invalid request")
	ErrInputTooLarge   = errors.New("cudainspect: input exceeds limit")
	ErrNotCUDA         = errors.New("cudainspect: input is not a CUDA ELF image")
	ErrELFVersion      = errors.New("cudainspect: unsupported CUDA ELF version")
	ErrNoCode          = errors.New("cudainspect: image has no code symbols")
	ErrSymbol          = errors.New("cudainspect: symbol is not exactly one code symbol")
	ErrSASS            = errors.New("cudainspect: supplied SASS does not describe the selection")
	ErrRequiredFeature = errors.New("cudainspect: required feature unavailable")
	ErrInvalidReport   = errors.New("cudainspect: invalid report")
)

// Request selects one kernel and the features a caller depends on.
//
// Required and Optional name features. A required feature without a value
// fails the report; an optional one makes it partial. A name the inspector
// does not implement is reported as unsupported, not dropped. MaxInputBytes
// lowers the image bound; zero means [DefaultMaxInputBytes].
//
// SASS is optional cuobjdump -sass or nvdisasm output for the image, at most
// [MaxSASSBytes]. It is parsed, never produced: Report runs no tools.
type Request struct {
	Symbol        string   `json:"symbol"`
	Required      []string `json:"required"`
	Optional      []string `json:"optional"`
	MaxInputBytes int64    `json:"max_input_bytes"`
	SASS          string   `json:"-"`
}

// Report is a validated, schema-versioned inspection of one kernel in one
// CUDA ELF image: Request.Symbol names exactly one .text section of the
// image. Every feature has a state; a feature without a value has an explicit
// null value and a diagnostic saying why.
//
// The features are:
//
//	allocated_registers   registers per thread from the image's allocation attribute
//	static_shared_bytes   size of the kernel's .nv.shared section
//	dynamic_shared_bytes  always unsupported: it requires a launch configuration
//	target                SM number from the ELF header flags
//	feature_suffix        a or f suffix declared by the supplied SASS target;
//	                      never established by the numeric ELF target
//	toolchain_stamp       tool versions from .note.nv.tkinfo
//	section_inventory     ELF section headers, as [SectionInfo]
//	opcode_histogram      mnemonic counts of the selected SASS function
//	memory_access_widths  global, shared and local load and store counts by width
//	local_memory_ops      local loads and stores, which are not classified as spills
//
// The last three require Request.SASS. They are observed when the listing's
// encodings are exactly the image's .text section and inferred otherwise.
// An instruction with a malformed opcode or ambiguous width is quarantined:
// it is reported in a diagnostic and never counted.
type Report struct {
	SchemaVersion int          `json:"schema_version"`
	Artifact      Digest       `json:"artifact"`
	SASS          *Digest      `json:"sass"`
	Request       Request      `json:"request"`
	Limits        Limits       `json:"limits"`
	Result        string       `json:"result"`
	Features      []Feature    `json:"features"`
	Diagnostics   []Diagnostic `json:"diagnostics"`
}

// Digest identifies supplied bytes.
type Digest struct {
	SHA256    string `json:"sha256"`
	SizeBytes int64  `json:"size_bytes"`
}

// Limits records the bounds applied to the inputs. ToolOutput is always
// [StateUnsupported]: Report runs no tools, so it bounds no tool output.
type Limits struct {
	InputBytes int64  `json:"input_bytes"`
	SASSBytes  int64  `json:"sass_bytes"`
	ToolOutput string `json:"tool_output"`
}

// Feature is one fact about the selected kernel.
// Value is JSON null unless State is observed or inferred.
type Feature struct {
	Name     string          `json:"name"`
	State    string          `json:"state"`
	Value    json.RawMessage `json:"value"`
	Unit     string          `json:"unit,omitempty"`
	Evidence []string        `json:"evidence,omitempty"`
}

// Diagnostic explains a feature's missing value or a quarantined input.
type Diagnostic struct {
	Feature string `json:"feature"`
	Message string `json:"message"`
}

// Report inspects exactly req.Symbol in cubin and returns a validated report.
//
// Unlike [Inspector.Inspect], which describes whatever bytes it is given,
// Report refuses input it cannot describe exactly: an image larger than the
// bound, an image that is not an EM_CUDA ELF file, an EM_CUDA image whose ELF
// version is not 1, an image without .text.* code sections, a symbol that is
// not exactly one of them, and SASS that does not list that symbol exactly
// once or names another target. Each refusal wraps one of the package's Err
// values. When a required feature has no value, Report returns the report
// together with [ErrRequiredFeature].
func (i *Inspector) Report(ctx context.Context, cubin []byte, req Request) (*Report, error) {
	if i == nil {
		return nil, fmt.Errorf("%w: nil inspector", ErrRequest)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	limit, err := req.check()
	if err != nil {
		return nil, err
	}
	if int64(len(cubin)) > limit {
		return nil, fmt.Errorf("%w: image is %d bytes, limit %d", ErrInputTooLarge, len(cubin), limit)
	}
	if len(req.SASS) > MaxSASSBytes {
		return nil, fmt.Errorf("%w: SASS is %d bytes, limit %d", ErrInputTooLarge, len(req.SASS), MaxSASSBytes)
	}
	if len(cubin) == 0 {
		return nil, fmt.Errorf("%w: empty image", ErrNotCUDA)
	}
	base, err := i.Inspect(cubin, req.Symbol)
	if err != nil {
		return nil, err
	}
	if !base.ELFParsed {
		return nil, elfRefusal(cubin, base.Unknowns)
	}
	if base.ELFMachine != elf.EM_CUDA.String() {
		return nil, fmt.Errorf("%w: machine %q", ErrNotCUDA, base.ELFMachine)
	}
	if len(base.CodeSymbols) == 0 {
		return nil, ErrNoCode
	}
	if n := countSections(base.SectionDetails, ".text."+req.Symbol); n != 1 {
		return nil, fmt.Errorf("%w: %q matches %d code sections", ErrSymbol, req.Symbol, n)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	h := sha256.Sum256(cubin)
	r := &Report{
		SchemaVersion: ReportSchemaVersion,
		Artifact:      Digest{SHA256: hex.EncodeToString(h[:]), SizeBytes: int64(len(cubin))},
		Request:       req.echo(),
		Limits:        Limits{InputBytes: limit, SASSBytes: MaxSASSBytes, ToolOutput: StateUnsupported},
		Features:      []Feature{},
		Diagnostics:   []Diagnostic{},
	}
	r.addELF(base)
	if err := r.addSASS(ctx, req.SASS, cubin, base.SectionDetails); err != nil {
		return nil, err
	}
	r.addRequested()
	r.Result = r.outcome()
	if err := r.Validate(); err != nil {
		return nil, err
	}
	if r.Result == ResultFailed {
		var missing []string
		for _, name := range r.Request.Required {
			if !r.Feature(name).Known() {
				missing = append(missing, name)
			}
		}
		return r, fmt.Errorf("%w: %s", ErrRequiredFeature, strings.Join(missing, ", "))
	}
	return r, nil
}

// elfRefusal explains why debug/elf could not read image. An EM_CUDA header
// with an ELF version other than 1 is refused with [ErrELFVersion]; anything
// else is not a CUDA ELF image.
func elfRefusal(image []byte, unknowns []string) error {
	if len(image) >= 24 && string(image[:4]) == "\x7fELF" {
		var order binary.ByteOrder = binary.LittleEndian
		if elf.Data(image[elf.EI_DATA]) == elf.ELFDATA2MSB {
			order = binary.BigEndian
		}
		ident, version := image[elf.EI_VERSION], order.Uint32(image[20:])
		if elf.Machine(order.Uint16(image[18:])) == elf.EM_CUDA && (ident != 1 || version != 1) {
			return fmt.Errorf("%w: EI_VERSION %d, e_version %d; only version 1 is supported", ErrELFVersion, ident, version)
		}
	}
	reason := "invalid ELF image"
	for _, u := range unknowns {
		if r, ok := strings.CutPrefix(u, "section inventory unavailable: "); ok {
			reason = r
		}
	}
	return fmt.Errorf("%w: ELF parse failed: %s", ErrNotCUDA, reason)
}

// check validates the request and returns the effective image bound.
func (req Request) check() (int64, error) {
	if req.Symbol == "" || len(req.Symbol) > maxSymbolBytes {
		return 0, fmt.Errorf("%w: symbol must be 1 to %d bytes", ErrSymbol, maxSymbolBytes)
	}
	if req.MaxInputBytes < 0 || req.MaxInputBytes > DefaultMaxInputBytes {
		return 0, fmt.Errorf("%w: max input bytes %d outside 0..%d", ErrRequest, req.MaxInputBytes, DefaultMaxInputBytes)
	}
	if len(req.Required)+len(req.Optional) > maxFeatureNames {
		return 0, fmt.Errorf("%w: more than %d feature names", ErrRequest, maxFeatureNames)
	}
	seen := make(map[string]bool)
	for _, name := range append(append([]string(nil), req.Required...), req.Optional...) {
		if name == "" || len(name) > maxFeatureNameBytes {
			return 0, fmt.Errorf("%w: feature name must be 1 to %d bytes", ErrRequest, maxFeatureNameBytes)
		}
		if seen[name] {
			return 0, fmt.Errorf("%w: feature %q requested twice", ErrRequest, name)
		}
		seen[name] = true
	}
	if req.MaxInputBytes == 0 {
		return DefaultMaxInputBytes, nil
	}
	return req.MaxInputBytes, nil
}

// echo returns req with non-nil feature lists, so they encode as [], and
// without the SASS text, which the report identifies by digest.
func (req Request) echo() Request {
	req.SASS = ""
	req.Required = append([]string{}, req.Required...)
	req.Optional = append([]string{}, req.Optional...)
	return req
}

func countSections(sections []SectionInfo, name string) int {
	n := 0
	for _, s := range sections {
		if s.Name == name {
			n++
		}
	}
	return n
}

// addELF records the features the existing ELF owners establish.
func (r *Report) addELF(base *ReportSchemaV1) {
	sym := r.Request.Symbol
	switch info := ".nv.info." + sym; {
	case countSections(base.SectionDetails, info) > 1:
		r.unavailable("allocated_registers", StateError, "duplicate "+info+" sections")
	case base.RegCountKnown:
		r.set("allocated_registers", StateObserved, base.RegCount, "registers", info+" EIATTR_REGCOUNT")
	default:
		r.unavailable("allocated_registers", StateUnknown, "no nonzero allocation attribute for the symbol")
	}
	switch shared := ".nv.shared." + sym; {
	case countSections(base.SectionDetails, shared) > 1:
		r.unavailable("static_shared_bytes", StateError, "duplicate "+shared+" sections")
	case base.SharedMemoryKnown:
		r.set("static_shared_bytes", StateObserved, base.SharedMemory, "bytes", shared+" section size")
	default:
		r.unavailable("static_shared_bytes", StateUnknown, "no "+shared+" section; absence does not establish zero")
	}
	r.unavailable("dynamic_shared_bytes", StateUnsupported, "requires launch configuration")
	if base.Architecture == "" {
		r.unavailable("target", StateUnknown, "ELF header carries no SM number for its ABI version")
	} else if major, minor, _, err := cudakernel.ParseArch(base.Architecture); err != nil {
		r.unavailable("target", StateError, err.Error())
	} else {
		v := struct {
			Architecture string `json:"architecture"`
			Major        int    `json:"major"`
			Minor        int    `json:"minor"`
		}{base.Architecture, major, minor}
		r.set("target", StateObserved, v, "", "ELF e_flags SM field")
	}
	if base.Toolchain == "" {
		r.unavailable("toolchain_stamp", StateUnknown, "no readable .note.nv.tkinfo")
	} else {
		r.set("toolchain_stamp", StateObserved, base.Toolchain, "", ".note.nv.tkinfo")
	}
	r.set("section_inventory", StateObserved, base.SectionDetails, "", "ELF section headers")
}

// addRequested reports each requested name the inspector does not implement.
func (r *Report) addRequested() {
	for _, name := range append(append([]string(nil), r.Request.Required...), r.Request.Optional...) {
		if r.Feature(name) == nil {
			r.unavailable(name, StateUnsupported, "feature not implemented")
		}
	}
}

func (r *Report) set(name, state string, value any, unit string, evidence ...string) {
	b, err := json.Marshal(value)
	if err != nil {
		r.unavailable(name, StateError, err.Error())
		return
	}
	r.Features = append(r.Features, Feature{Name: name, State: state, Value: b, Unit: unit, Evidence: evidence})
}

func (r *Report) unavailable(name, state, reason string) {
	r.Features = append(r.Features, Feature{Name: name, State: state})
	r.Diagnostics = append(r.Diagnostics, Diagnostic{Feature: name, Message: reason})
}

// Feature returns the named feature, or nil if the report does not name it.
func (r *Report) Feature(name string) *Feature {
	for n := range r.Features {
		if r.Features[n].Name == name {
			return &r.Features[n]
		}
	}
	return nil
}

// Known reports whether f carries a value.
func (f *Feature) Known() bool {
	return f != nil && (f.State == StateObserved || f.State == StateInferred)
}

func (r *Report) outcome() string {
	for _, name := range r.Request.Required {
		if !r.Feature(name).Known() {
			return ResultFailed
		}
	}
	for _, name := range r.Request.Optional {
		if !r.Feature(name).Known() {
			return ResultPartial
		}
	}
	return ResultComplete
}

// A featureRule is what Validate accepts for one implemented feature: the
// states Report can give it, whether only supplied SASS can establish it, and
// the shape of its value.
type featureRule struct {
	states []string
	sass   bool
	check  func(json.RawMessage) error
}

var (
	allStates  = []string{StateObserved, StateInferred, StateUnknown, StateUnsupported, StateError}
	elfStates  = []string{StateObserved, StateUnknown, StateError}
	sassStates = []string{StateObserved, StateInferred, StateUnknown}
)

// featureRules holds every feature Report implements.
var featureRules = map[string]featureRule{
	"allocated_registers":  {elfStates, false, checkRegisters},
	"static_shared_bytes":  {elfStates, false, checkValue[uint32]},
	"dynamic_shared_bytes": {[]string{StateUnsupported}, false, nil},
	"target":               {elfStates, false, checkTarget},
	"feature_suffix":       {[]string{StateInferred, StateUnknown}, true, checkSuffix},
	"toolchain_stamp":      {[]string{StateObserved, StateUnknown}, false, checkStamp},
	"section_inventory":    {[]string{StateObserved}, false, checkSections},
	"opcode_histogram":     {sassStates, true, checkHistogram},
	"memory_access_widths": {sassStates, true, checkWidths},
	"local_memory_ops":     {sassStates, true, checkValue[localMemoryOps]},
}

// decodeValue decodes v into dst, refusing unknown fields and trailing data.
func decodeValue(v json.RawMessage, dst any) error {
	dec := json.NewDecoder(bytes.NewReader(v))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return err
	}
	if _, err := dec.Token(); err != io.EOF {
		return errors.New("trailing data")
	}
	return nil
}

func hasValue[K, V comparable](m map[K]V, v V) bool {
	for _, x := range m {
		if x == v {
			return true
		}
	}
	return false
}

func checkValue[T any](v json.RawMessage) error {
	var x T
	return decodeValue(v, &x)
}

func checkRegisters(v json.RawMessage) error {
	var n uint32
	if err := decodeValue(v, &n); err != nil {
		return err
	}
	if n == 0 {
		return errors.New("zero registers")
	}
	return nil
}

func checkTarget(v json.RawMessage) error {
	var t struct {
		Architecture string `json:"architecture"`
		Major        int    `json:"major"`
		Minor        int    `json:"minor"`
	}
	if err := decodeValue(v, &t); err != nil {
		return err
	}
	major, minor, suffix, err := cudakernel.ParseArch(t.Architecture)
	if err != nil || !strings.HasPrefix(t.Architecture, "sm_") || suffix != "" || major != t.Major || minor != t.Minor {
		return fmt.Errorf("architecture %q is not sm_%d%d", t.Architecture, t.Major, t.Minor)
	}
	return nil
}

func checkSuffix(v json.RawMessage) error {
	var s string
	if err := decodeValue(v, &s); err != nil {
		return err
	}
	if !featureSuffixes[s] {
		return fmt.Errorf("suffix %q, want a or f", s)
	}
	return nil
}

func checkStamp(v json.RawMessage) error {
	var s string
	if err := decodeValue(v, &s); err != nil {
		return err
	}
	if s == "" {
		return errors.New("empty stamp")
	}
	return nil
}

func checkSections(v json.RawMessage) error {
	var s []SectionInfo
	if err := decodeValue(v, &s); err != nil {
		return err
	}
	if len(s) == 0 {
		return errors.New("no sections")
	}
	return nil
}

func checkHistogram(v json.RawMessage) error {
	var h opcodeHistogram
	if err := decodeValue(v, &h); err != nil {
		return err
	}
	if h.Opcodes == nil {
		return errors.New("no opcodes")
	}
	var sum uint64
	for op, n := range h.Opcodes {
		if !sassMnemonic.MatchString(op) || n == 0 {
			return fmt.Errorf("opcode %q count %d", op, n)
		}
		if n > ^uint64(0)-sum {
			return errors.New("opcode counts overflow")
		}
		sum += n
	}
	if sum != h.Counted {
		return fmt.Errorf("opcodes sum to %d, counted %d", sum, h.Counted)
	}
	return nil
}

func checkWidths(v json.RawMessage) error {
	var w map[string]map[string]uint64
	if err := decodeValue(v, &w); err != nil {
		return err
	}
	if w == nil {
		return errors.New("null widths")
	}
	for class, counts := range w {
		if !hasValue(memoryClasses, class) || len(counts) == 0 {
			return fmt.Errorf("access class %q", class)
		}
		for bits, n := range counts {
			b, err := strconv.Atoi(bits)
			if err != nil || !hasValue(memoryWidths, b) || n == 0 {
				return fmt.Errorf("%s width %q count %d", class, bits, n)
			}
		}
	}
	return nil
}

func isNull(v json.RawMessage) bool {
	return len(v) == 0 || string(bytes.TrimSpace(v)) == "null"
}

// Validate checks the report's internal consistency. It rejects an unknown
// schema version or state, a value on a feature without one, a missing value
// or evidence on a feature with one, a value of the wrong shape, a state
// Report never gives the feature, a missing diagnostic for a feature without
// a value, a missing or duplicated feature, an invalid request or one whose
// image bound differs from the recorded limit, and a result that does not
// follow from the requested features.
func (r *Report) Validate() error {
	if r == nil {
		return fmt.Errorf("%w: nil report", ErrInvalidReport)
	}
	bad := func(format string, args ...any) error {
		return fmt.Errorf("%w: %s", ErrInvalidReport, fmt.Sprintf(format, args...))
	}
	if r.SchemaVersion != ReportSchemaVersion {
		return bad("schema version %d, want %d", r.SchemaVersion, ReportSchemaVersion)
	}
	if b, err := hex.DecodeString(r.Artifact.SHA256); err != nil || len(b) != sha256.Size {
		return bad("artifact digest %q", r.Artifact.SHA256)
	}
	limit, err := r.Request.check()
	if err != nil {
		return bad("request: %v", err)
	}
	if r.Limits.InputBytes != limit {
		return bad("input limit %d, request gives %d", r.Limits.InputBytes, limit)
	}
	if r.Artifact.SizeBytes <= 0 || r.Artifact.SizeBytes > limit {
		return bad("artifact size %d outside limit %d", r.Artifact.SizeBytes, limit)
	}
	if r.Limits.ToolOutput != StateUnsupported {
		return bad("tool output limit %q", r.Limits.ToolOutput)
	}
	if r.Limits.SASSBytes != MaxSASSBytes {
		return bad("SASS limit %d, want %d", r.Limits.SASSBytes, MaxSASSBytes)
	}
	if r.SASS != nil {
		if b, err := hex.DecodeString(r.SASS.SHA256); err != nil || len(b) != sha256.Size || r.SASS.SizeBytes <= 0 || r.SASS.SizeBytes > r.Limits.SASSBytes {
			return bad("SASS digest %+v", *r.SASS)
		}
	}
	explained := make(map[string]bool)
	for _, d := range r.Diagnostics {
		explained[d.Feature] = true
	}
	seen := make(map[string]bool)
	for _, f := range r.Features {
		if f.Name == "" || seen[f.Name] {
			return bad("feature %q empty or duplicated", f.Name)
		}
		seen[f.Name] = true
		rule, implemented := featureRules[f.Name]
		if !implemented {
			rule.states = []string{StateUnsupported}
		}
		if slices.Contains(allStates, f.State) && !slices.Contains(rule.states, f.State) {
			return bad("feature %q cannot be %s", f.Name, f.State)
		}
		switch f.State {
		case StateObserved, StateInferred:
			if isNull(f.Value) || len(f.Evidence) == 0 {
				return bad("feature %q is %s without value and evidence", f.Name, f.State)
			}
			if r.SASS == nil && rule.sass {
				return bad("feature %q is %s without SASS", f.Name, f.State)
			}
			if err := rule.check(f.Value); err != nil {
				return bad("feature %q value %s: %v", f.Name, f.Value, err)
			}
		case StateUnknown, StateUnsupported, StateError:
			if !isNull(f.Value) {
				return bad("feature %q is %s with value %s", f.Name, f.State, f.Value)
			}
			if !explained[f.Name] {
				return bad("feature %q is %s without diagnostic", f.Name, f.State)
			}
		default:
			return bad("feature %q has state %q", f.Name, f.State)
		}
	}
	for _, name := range append(append([]string(nil), r.Request.Required...), r.Request.Optional...) {
		if !seen[name] {
			return bad("requested feature %q absent", name)
		}
	}
	for name := range featureRules {
		if !seen[name] {
			return bad("feature %q absent", name)
		}
	}
	if want := r.outcome(); r.Result != want {
		return bad("result %q, features give %q", r.Result, want)
	}
	return nil
}

// WriteJSON writes the report as indented JSON.
func (r *Report) WriteJSON(w io.Writer) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(r)
}

// ReadReport decodes and validates one report. It refuses unknown fields,
// trailing data, and input over [MaxReportBytes].
func ReadReport(rd io.Reader) (*Report, error) {
	data, err := io.ReadAll(io.LimitReader(rd, MaxReportBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read report: %w", err)
	}
	if len(data) > MaxReportBytes {
		return nil, fmt.Errorf("%w: report over %d bytes", ErrInputTooLarge, MaxReportBytes)
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var r Report
	if err := dec.Decode(&r); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidReport, err)
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, fmt.Errorf("%w: trailing data", ErrInvalidReport)
	}
	if err := r.Validate(); err != nil {
		return nil, err
	}
	return &r, nil
}
