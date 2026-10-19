package cudainspect

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// MaxSASSBytes bounds Request.SASS.
const MaxSASSBytes = 16 << 20

// maxQuarantineDiagnostics bounds the per-instruction quarantine diagnostics;
// the rest are counted in one summary diagnostic.
const maxQuarantineDiagnostics = 16

var (
	sassMnemonic = regexp.MustCompile(`^[A-Z][A-Z0-9_]*$`)
	sassModifier = regexp.MustCompile(`^[A-Z0-9_]+$`)
	// sassWidth matches modifiers spelled like a size. One missing from
	// memoryWidths, such as the 256 in LDG.E.ENL2.256, is ambiguous.
	sassWidth = regexp.MustCompile(`^[US]?[0-9]+$`)
)

// memoryClasses names the direction of the memory opcodes whose access width
// is reported. Other memory opcodes appear only in the histogram.
var memoryClasses = map[string]string{
	"LDG": "global_load", "STG": "global_store",
	"LDS": "shared_load", "STS": "shared_store",
	"LDL": "local_load", "STL": "local_store",
}

// memoryWidths maps explicit size modifiers to bits. A memory opcode with no
// size modifier is a 32-bit access, as nvdisasm writes LDG.E for one. A memory
// opcode with any other size-like modifier is quarantined.
var memoryWidths = map[string]int{
	"U8": 8, "S8": 8, "U16": 16, "S16": 16, "U32": 32, "S32": 32,
	"32": 32, "64": 64, "128": 128,
}

type opcodeHistogram struct {
	Counted     uint64            `json:"counted"`
	Quarantined uint64            `json:"quarantined"`
	Opcodes     map[string]uint64 `json:"opcodes"`
}

type localMemoryOps struct {
	Loads  uint64 `json:"loads"`
	Stores uint64 `json:"stores"`
}

// addSASS records the instruction features of the selected function in the
// supplied SASS listing. Without a listing they are unknown. The listing
// must name the selected function exactly once and, when it declares a
// target, the same SM number as the image. Instruction features are observed
// only when the listing's encodings are exactly the image's .text section;
// otherwise they are inferred from text the caller supplied.
func (r *Report) addSASS(ctx context.Context, sass string, image []byte, sections []SectionInfo) error {
	names := []string{"opcode_histogram", "memory_access_widths", "local_memory_ops"}
	if sass == "" {
		for _, name := range names {
			r.unavailable(name, StateUnknown, "no SASS supplied")
		}
		if r.requested("precision_attribution") {
			r.unavailable("precision_attribution", StateUnknown, "no SASS supplied")
		}
		r.unavailable("feature_suffix", StateUnknown, "numeric ELF target does not establish a or f suffix features")
		return nil
	}
	names = append(names, "precision_attribution")
	h := sha256.Sum256([]byte(sass))
	r.SASS = &Digest{SHA256: hex.EncodeToString(h[:]), SizeBytes: int64(len(sass))}
	d, err := cudasass.ParseString(sass)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrSASS, err)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	sym := r.Request.Symbol
	var fn *cudasass.Function
	for _, f := range d.Functions {
		if f.Name != sym {
			continue
		}
		if fn != nil {
			return fmt.Errorf("%w: function %q listed twice", ErrSASS, sym)
		}
		fn = f
	}
	if fn == nil || len(fn.Instructions) == 0 {
		return fmt.Errorf("%w: no instructions for function %q", ErrSASS, sym)
	}
	if err := r.addSuffix(d.Target); err != nil {
		return err
	}

	state, evidence := StateObserved, "SASS encodings equal .text."+sym+" bytes"
	if reason := compiledInstructionCoverage(sections, image, sym, fn); reason != "" {
		state, evidence = StateInferred, "supplied SASS sha256:"+r.SASS.SHA256
		for _, name := range names {
			r.Diagnostics = append(r.Diagnostics, Diagnostic{Feature: name, Message: "SASS not bound to image bytes: " + reason})
		}
	}
	hist := opcodeHistogram{Opcodes: map[string]uint64{}}
	widths := map[string]map[string]uint64{}
	var local localMemoryOps
	for n, inst := range fn.Instructions {
		if n%4096 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		base, class, bits, reason := classifyOpcode(inst.Opcode)
		if reason != "" {
			hist.Quarantined++
			if hist.Quarantined <= maxQuarantineDiagnostics {
				r.Diagnostics = append(r.Diagnostics, Diagnostic{Feature: "opcode_histogram", Message: fmt.Sprintf("quarantined instruction at %#x: %s", inst.Address, reason)})
			}
			continue
		}
		hist.Counted++
		hist.Opcodes[base]++
		if class == "" {
			continue
		}
		if widths[class] == nil {
			widths[class] = map[string]uint64{}
		}
		widths[class][strconv.Itoa(bits)]++
		switch class {
		case "local_load":
			local.Loads++
		case "local_store":
			local.Stores++
		}
	}
	if extra := hist.Quarantined - min(hist.Quarantined, maxQuarantineDiagnostics); extra > 0 {
		r.Diagnostics = append(r.Diagnostics, Diagnostic{Feature: "opcode_histogram", Message: fmt.Sprintf("%d more quarantined instructions", extra)})
	}
	r.set("opcode_histogram", state, hist, "instructions", evidence)
	if hist.Quarantined > 0 {
		reason := fmt.Sprintf("%d quarantined instructions may be memory operations", hist.Quarantined)
		r.unavailable("memory_access_widths", StateUnknown, reason)
		r.unavailable("local_memory_ops", StateUnknown, reason)
		r.addPrecisionAttribution(fn, state, evidence, hist.Quarantined)
		return nil
	}
	r.set("memory_access_widths", state, widths, "instructions by width in bits", evidence)
	r.set("local_memory_ops", state, local, "instructions", evidence)
	r.Diagnostics = append(r.Diagnostics, Diagnostic{Feature: "local_memory_ops", Message: "local memory operations are observations; none is classified as a spill"})
	r.addPrecisionAttribution(fn, state, evidence, 0)
	return nil
}

// featureSuffixes are the target suffixes feature_suffix may report.
var featureSuffixes = map[string]bool{"a": true, "f": true}

// addSuffix records the a/f feature suffix the listing's declared target
// names. It refuses a listing whose SM number differs from the image's or
// whose suffix is neither a nor f.
func (r *Report) addSuffix(target string) error {
	if target == "" {
		r.unavailable("feature_suffix", StateUnknown, "SASS declares no target; numeric ELF target does not establish a or f suffix features")
		return nil
	}
	major, minor, suffix, err := cudakernel.ParseArch(target)
	if err != nil {
		return fmt.Errorf("%w: target %q: %v", ErrSASS, target, err)
	}
	if t := r.Feature("target"); t.Known() {
		var v struct{ Major, Minor int }
		if err := json.Unmarshal(t.Value, &v); err != nil || v.Major != major || v.Minor != minor {
			return fmt.Errorf("%w: target %q differs from image target %s", ErrSASS, target, t.Value)
		}
	}
	if suffix == "" {
		r.unavailable("feature_suffix", StateUnknown, "SASS target "+target+" names no suffix; absence is not established")
		return nil
	}
	if !featureSuffixes[suffix] {
		return fmt.Errorf("%w: target %q has suffix %q, want a or f", ErrSASS, target, suffix)
	}
	r.set("feature_suffix", StateInferred, suffix, "", "supplied SASS .target "+target)
	return nil
}

// classifyOpcode returns the opcode's mnemonic, and for the memory opcodes in
// memoryClasses its direction and width. A nonempty reason quarantines the
// instruction: it is malformed or its width is ambiguous.
func classifyOpcode(opcode string) (base, class string, bits int, reason string) {
	fields := strings.Split(opcode, ".")
	base = fields[0]
	if !sassMnemonic.MatchString(base) {
		return "", "", 0, fmt.Sprintf("opcode %q is not a mnemonic", opcode)
	}
	for _, m := range fields[1:] {
		if !sassModifier.MatchString(m) {
			return "", "", 0, fmt.Sprintf("opcode %q has malformed modifier %q", opcode, m)
		}
	}
	class = memoryClasses[base]
	if class == "" {
		return base, "", 0, ""
	}
	bits = 32
	explicit := 0
	for _, m := range fields[1:] {
		if w, ok := memoryWidths[m]; ok {
			explicit++
			bits = w
		} else if sassWidth.MatchString(m) {
			return "", "", 0, fmt.Sprintf("opcode %q has unrecognized size modifier %q", opcode, m)
		}
	}
	if explicit > 1 {
		return "", "", 0, fmt.Sprintf("opcode %q has %d size modifiers", opcode, explicit)
	}
	return base, class, bits, ""
}
