package cudainspect

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// ResourceFact is a nonnegative resource count with its evidence category.
// A nil Value means unavailable, including when an owner cannot distinguish
// absent metadata from a genuine zero. Category is not runtime occupancy.
type ResourceFact struct {
	Value    *uint64 `json:"value"`
	Category string  `json:"category"`
	Reason   string  `json:"reason,omitempty"`
}

// CompiledResourceInput supplies exact compilation and disassembly receipts.
// DiagnosticImage and SASSImage identify the bytes read by those tool runs.
// These are producer receipts, not tool authenticity or driver attestations.
// Peak life ranges are optional nvdisasm count-mode output for the same image.
type CompiledResourceInput struct {
	Source              []byte
	CompilerVersion     string
	CompilerOptions     []string
	Diagnostics         string
	DiagnosticImage     string
	SASS                string
	SASSImage           string
	DisassemblerVersion string
	LifeRanges          string
	RegisterCap         *uint64
	ModelRegisters      *uint64
}

// CompiledResources keeps allocated, live, modeled and capped registers apart.
// Local operations are observations, not proof that every operation is a spill.
type CompiledResources struct {
	SchemaVersion       string       `json:"schema_version"`
	SourceDigest        string       `json:"source_digest"`
	ImageDigest         string       `json:"image_digest"`
	Entry               string       `json:"entry"`
	Target              string       `json:"target"`
	CompilerVersion     string       `json:"compiler_version"`
	CompilerOptions     []string     `json:"compiler_options"`
	DisassemblerVersion string       `json:"disassembler_version"`
	Allocated           ResourceFact `json:"allocated"`
	PeakLive            ResourceFact `json:"peak_live"`
	RegisterCap         ResourceFact `json:"register_cap"`
	ModelRegisters      ResourceFact `json:"model_registers"`
	StaticShared        ResourceFact `json:"static_shared"`
	LocalMemory         ResourceFact `json:"local_memory"`
	StackBytes          ResourceFact `json:"stack_bytes"`
	SpillStoreBytes     ResourceFact `json:"spill_store_bytes"`
	SpillLoadBytes      ResourceFact `json:"spill_load_bytes"`
	LocalLoads          []uint64     `json:"local_load_pcs"`
	LocalStores         []uint64     `json:"local_store_pcs"`
	InstructionsKnown   bool         `json:"instructions_known"`
	InstructionsReason  string       `json:"instructions_reason,omitempty"`
	Diagnostics         string       `json:"diagnostics"`
	SASS                string       `json:"sass"`
}

func resourceFact(value uint64, category string) ResourceFact {
	return ResourceFact{Value: &value, Category: category}
}

func optionalResource(value *uint64, category string) ResourceFact {
	if value == nil {
		return ResourceFact{Category: category, Reason: "not supplied"}
	}
	return resourceFact(*value, category)
}

var compiledEntry = regexp.MustCompile(`Compiling entry function '([^']+)' for '([^']+)'`)
var compiledProperties = regexp.MustCompile(`Function properties for\s+(.+?)\s*$`)
var compiledSpills = regexp.MustCompile(`([0-9]+) bytes stack frame, ([0-9]+) bytes spill stores, ([0-9]+) bytes spill loads`)

// InspectCompiledResources inspects a validated CUBIN using the existing ELF
// and SASS owners. Inputs are bounded to 64 MiB of image and 16 MiB of textual
// receipts. Only the exact entry's unambiguous ptxas summary is admitted.
func (i *Inspector) InspectCompiledResources(a *cudakernel.Artifact, in CompiledResourceInput) (*CompiledResources, error) {
	if i == nil || a == nil {
		return nil, fmt.Errorf("compiled resources: inspector and artifact required")
	}
	if len(a.Raw) > 64<<20 || len(in.Source)+len(in.Diagnostics)+len(in.SASS)+len(in.LifeRanges) > 16<<20 {
		return nil, fmt.Errorf("compiled resources: input budget exceeded")
	}
	if len(in.CompilerVersion) > 4096 || len(in.DisassemblerVersion) > 4096 || len(in.CompilerOptions) > 128 {
		return nil, fmt.Errorf("compiled resources: tool metadata budget exceeded")
	}
	if err := a.Validate(); err != nil {
		return nil, err
	}
	if a.Kind != cudakernel.KindCUBIN || len(in.Source) == 0 || strings.TrimSpace(in.CompilerVersion) == "" {
		return nil, fmt.Errorf("compiled resources: CUBIN, source and compiler version required")
	}
	if in.DiagnosticImage != a.Digest || in.SASSImage != a.Digest || strings.TrimSpace(in.DisassemblerVersion) == "" {
		return nil, fmt.Errorf("compiled resources: receipt image or tool identity mismatch")
	}
	if (a.Toolchain != "" && a.Toolchain != in.CompilerVersion) || (len(a.CompilerOpts) > 0 && !slices.Equal(a.CompilerOpts, in.CompilerOptions)) {
		return nil, fmt.Errorf("compiled resources: artifact compiler receipt mismatch")
	}
	var cap *uint64
	for n, opt := range in.CompilerOptions {
		if len(opt) > 4096 || len(in.CompilerOptions) > 128 {
			return nil, fmt.Errorf("compiled resources: compiler option budget exceeded")
		}
		value, ok := strings.CutPrefix(opt, "--maxrregcount=")
		if !ok {
			value, ok = strings.CutPrefix(opt, "-maxrregcount=")
		}
		if opt == "--maxrregcount" || opt == "-maxrregcount" {
			if n+1 >= len(in.CompilerOptions) {
				return nil, fmt.Errorf("compiled resources: missing cap argument")
			}
			value = in.CompilerOptions[n+1]
			ok = true
		}
		if ok {
			v, err := strconv.ParseUint(value, 10, 64)
			if err != nil || v == 0 || cap != nil {
				return nil, fmt.Errorf("compiled resources: invalid or duplicate cap")
			}
			cap = &v
		}
	}
	if (cap == nil) != (in.RegisterCap == nil) || (cap != nil && *cap != *in.RegisterCap) {
		return nil, fmt.Errorf("compiled resources: register cap differs from compiler invocation")
	}
	base, err := i.Inspect(a.Raw, a.EntryPoint)
	if err != nil {
		return nil, err
	}
	if !base.ELFParsed || base.ELFMachine != "EM_CUDA" || base.Architecture != a.Target {
		return nil, fmt.Errorf("compiled resources: decoded image target mismatch")
	}
	found := false
	for _, name := range base.CodeSymbols {
		if name == a.EntryPoint {
			found = true
		}
	}
	if !found {
		return nil, fmt.Errorf("compiled resources: entry absent from image")
	}
	h := sha256.Sum256(in.Source)
	r := &CompiledResources{
		SchemaVersion: "compiled-resources/v1",
		SourceDigest:  hex.EncodeToString(h[:]), ImageDigest: a.Digest, Entry: a.EntryPoint, Target: a.Target,
		CompilerVersion: in.CompilerVersion, CompilerOptions: append([]string(nil), in.CompilerOptions...),
		DisassemblerVersion: in.DisassemblerVersion,
		Allocated:           ResourceFact{Category: "decoded", Reason: "allocation metadata unavailable"},
		StaticShared:        ResourceFact{Category: "decoded", Reason: "static shared metadata unavailable"},
		PeakLive:            ResourceFact{Category: "derived", Reason: "matched life ranges unavailable"},
		RegisterCap:         optionalResource(in.RegisterCap, "compiler condition"),
		ModelRegisters:      optionalResource(in.ModelRegisters, "modeled"),
		LocalMemory:         ResourceFact{Category: "decoded", Reason: "existing inspector does not expose per-thread local memory bytes"},
		StackBytes:          ResourceFact{Category: "compiler reported", Reason: "matched summary unavailable"},
		SpillStoreBytes:     ResourceFact{Category: "compiler reported", Reason: "matched summary unavailable"},
		SpillLoadBytes:      ResourceFact{Category: "compiler reported", Reason: "matched summary unavailable"},
		Diagnostics:         in.Diagnostics, SASS: in.SASS,
	}
	if base.RegCountKnown {
		r.Allocated = resourceFact(uint64(base.RegCount), "decoded")
	}
	if base.SharedMemoryKnown {
		r.StaticShared = resourceFact(uint64(base.SharedMemory), "decoded")
	}
	selected := false
	summaries := 0
	for _, line := range strings.Split(in.Diagnostics, "\n") {
		if m := compiledEntry.FindStringSubmatch(line); m != nil {
			selected = m[1] == a.EntryPoint && m[2] == a.Target
			continue
		}
		if m := compiledProperties.FindStringSubmatch(line); m != nil {
			selected = selected && m[1] == a.EntryPoint
			continue
		}
		if m := compiledSpills.FindStringSubmatch(line); m != nil && selected {
			summaries++
			if summaries > 1 {
				return nil, fmt.Errorf("compiled resources: duplicate entry summary")
			}
			v := [3]uint64{}
			for n := range v {
				v[n], err = strconv.ParseUint(m[n+1], 10, 64)
				if err != nil {
					return nil, err
				}
			}
			r.StackBytes = resourceFact(v[0], "compiler reported")
			r.SpillStoreBytes = resourceFact(v[1], "compiler reported")
			r.SpillLoadBytes = resourceFact(v[2], "compiler reported")
		}
	}
	d, err := cudasass.Parse(strings.NewReader(in.SASS))
	if err != nil {
		return nil, fmt.Errorf("compiled resources: parse SASS: %w", err)
	}
	var fn *cudasass.Function
	for _, f := range d.Functions {
		if f.Name == a.EntryPoint {
			if fn != nil {
				return nil, fmt.Errorf("compiled resources: duplicate SASS entry")
			}
			fn = f
		}
	}
	r.InstructionsReason = "matched SASS entry unavailable"
	if fn != nil && len(fn.Instructions) > 0 && d.Target == a.Target {
		r.InstructionsReason = compiledInstructionCoverage(base.SectionDetails, a.Raw, a.EntryPoint, fn)
		r.InstructionsKnown = r.InstructionsReason == ""
		for _, op := range fn.Instructions {
			name := strings.Split(op.Opcode, ".")[0]
			if name == "LDL" {
				r.LocalLoads = append(r.LocalLoads, op.Address)
			}
			if name == "STL" {
				r.LocalStores = append(r.LocalStores, op.Address)
			}
		}
	}
	if d.Target != a.Target {
		r.InstructionsReason = "SASS target unavailable or inconsistent"
	}
	if in.LifeRanges != "" && d.Target == a.Target {
		p, err := cudasass.ParseLifeRangeCount(bytes.NewBufferString(in.LifeRanges))
		if err != nil {
			return nil, err
		}
		// Require exact names before the owner's join: its single-function fallback
		// must not turn a different kernel's pressure into this entry's evidence.
		if p[a.EntryPoint] == nil || fn == nil {
			return nil, fmt.Errorf("compiled resources: unmatched life range entry")
		}
		r.PeakLive = compiledPeakLive(r.InstructionsKnown, fn, p[a.EntryPoint])
	}
	return r, nil
}

// compiledPeakLive requires complete instruction and pressure coverage. A
// maximum over a partial listing is not the whole kernel's peak live count.
func compiledPeakLive(complete bool, fn *cudasass.Function, pressure map[uint64]cudasass.RegisterPressure) ResourceFact {
	if !complete {
		return ResourceFact{Category: "derived", Reason: "complete artifact instruction coverage unavailable"}
	}
	if fn == nil || len(pressure) != len(fn.Instructions) {
		return ResourceFact{Category: "derived", Reason: "complete pressure coverage unavailable"}
	}
	for _, op := range fn.Instructions {
		if _, ok := pressure[op.Address]; !ok {
			return ResourceFact{Category: "derived", Reason: "complete pressure coverage unavailable"}
		}
	}
	only := &cudasass.Disassembly{Functions: []*cudasass.Function{fn}}
	if err := cudasass.JoinPressure(only, map[string]map[uint64]cudasass.RegisterPressure{fn.Name: pressure}); err != nil {
		return ResourceFact{Category: "derived", Reason: "pressure join unavailable: " + err.Error()}
	}
	var peak uint64
	for _, op := range fn.Instructions {
		if op.LiveGPR < 0 {
			return ResourceFact{Category: "derived", Reason: "negative pressure"}
		}
		if uint64(op.LiveGPR) > peak {
			peak = uint64(op.LiveGPR)
		}
	}
	return resourceFact(peak, "derived")
}

// compiledInstructionCoverage requires an exact encoded listing of the entire
// owner-inventoried text section. A terminator alone cannot establish coverage:
// omitted instructions before it, and instructions after it, remain unknown.
func compiledInstructionCoverage(sections []SectionInfo, image []byte, entry string, fn *cudasass.Function) string {
	var section *SectionInfo
	for n := range sections {
		if sections[n].Name == ".text."+entry {
			if section != nil {
				return "ambiguous text section"
			}
			section = &sections[n]
		}
	}
	if section == nil || section.Size == 0 || section.Offset > uint64(len(image)) || section.Size > uint64(len(image))-section.Offset {
		return "text section unavailable"
	}
	// Encoding.Bytes is the SASS owner's 16-byte instruction representation.
	if section.Size%16 != 0 || uint64(len(fn.Instructions)) != section.Size/16 {
		return "instruction listing does not cover text section"
	}
	for n, op := range fn.Instructions {
		pc := uint64(n) * 16
		if op.Address != pc || !op.Encoding.Present() {
			return "instruction address or encoding unavailable"
		}
		encoded := op.Encoding.Bytes()
		start := section.Offset + pc
		if !bytes.Equal(encoded[:], image[start:start+16]) {
			return "instruction encoding differs from image"
		}
	}
	return ""
}

// CorroboratedSpills reports matched compiler spill bytes and corresponding
// local load/store instructions. Explicit local arrays alone return false.
func (r *CompiledResources) CorroboratedSpills() bool {
	return r != nil && ((r.SpillLoadBytes.Category == "compiler reported" && r.SpillLoadBytes.Value != nil && *r.SpillLoadBytes.Value > 0 && len(r.LocalLoads) > 0) || (r.SpillStoreBytes.Category == "compiler reported" && r.SpillStoreBytes.Value != nil && *r.SpillStoreBytes.Value > 0 && len(r.LocalStores) > 0))
}

// LoadedResources records attributes queried from the function of an exactly
// loaded image. Image identity is the loader's receipt, not inferred by name.
type LoadedResources struct {
	ImageDigest  string
	Entry        string
	Target       string
	Registers    ResourceFact
	StaticShared ResourceFact
	LocalBytes   ResourceFact
}

// RegisterResidencyClaim is a strict no-spill claim for one compiled entry.
// This supported checker requires zero compiler stack/spill bytes, no decoded
// local operations, and equal allocated-register metadata and loaded attributes.
// It does not claim measured runtime occupancy or physical register contents.
type RegisterResidencyClaim struct{ SourceDigest, ImageDigest, Entry, Target string }

// CheckRegisterResidency rejects mismatched, absent or spilling evidence.
// A refusal is independent of numerical correctness.
func (r *CompiledResources) CheckRegisterResidency(c RegisterResidencyClaim, l LoadedResources) error {
	for _, digest := range []string{c.SourceDigest, c.ImageDigest} {
		b, err := hex.DecodeString(digest)
		if err != nil || len(b) != sha256.Size {
			return fmt.Errorf("register residency: invalid identity")
		}
	}
	if r == nil || c.SourceDigest != r.SourceDigest || c.ImageDigest != r.ImageDigest || c.Entry != r.Entry || c.Target != r.Target || l.ImageDigest != r.ImageDigest || l.Entry != r.Entry || l.Target != r.Target {
		return fmt.Errorf("register residency: identity mismatch")
	}
	if r.SchemaVersion != "compiled-resources/v1" {
		return fmt.Errorf("register residency: unsupported resource schema")
	}
	if r.Allocated.Value == nil || r.Allocated.Category != "decoded" || l.Registers.Value == nil || l.Registers.Category != "runtime attribute" || *r.Allocated.Value != *l.Registers.Value {
		return fmt.Errorf("register residency: allocated register evidence unavailable or inconsistent")
	}
	for _, f := range []ResourceFact{r.StackBytes, r.SpillLoadBytes, r.SpillStoreBytes} {
		if f.Category != "compiler reported" {
			return fmt.Errorf("register residency: wrong evidence category")
		}
	}
	if l.LocalBytes.Category != "runtime attribute" {
		return fmt.Errorf("register residency: wrong local evidence category")
	}
	for _, f := range []ResourceFact{r.StackBytes, r.SpillLoadBytes, r.SpillStoreBytes, l.LocalBytes} {
		if f.Value == nil {
			return fmt.Errorf("register residency: local or spill metadata unavailable")
		}
		if *f.Value != 0 {
			return fmt.Errorf("register residency: stack, local memory or compiler spills present")
		}
	}
	if !r.InstructionsKnown || len(r.LocalLoads)+len(r.LocalStores) > 0 {
		return fmt.Errorf("register residency: local instructions present or unavailable")
	}
	return nil
}
