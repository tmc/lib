package cudainspect

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudareg"
)

// RegisterResources converts the register facts of this selected compiled
// entry for cudareg. It checks this package's schema and producer identities;
// it does not authenticate the compiler, disassembler or supplied receipts.
// Source and tool provenance stay in r, which the caller should retain.
//
// Missing values stay unavailable, category mismatches or caveats stay
// unverified, and counts outside uint32 are refused. The declared target and
// numeric decoded target remain separate: numeric agreement proves no suffix.
func (r *CompiledResources) RegisterResources() (cudareg.Resources, error) {
	if r == nil || r.SchemaVersion != "compiled-resources/v1" {
		return cudareg.Resources{}, fmt.Errorf("register resources: compiled-resources/v1 report required")
	}
	for _, digest := range []string{r.SourceDigest, r.ImageDigest} {
		if b, err := hex.DecodeString(digest); err != nil || len(b) != sha256.Size {
			return cudareg.Resources{}, fmt.Errorf("register resources: invalid producer digest")
		}
	}
	if strings.TrimSpace(r.CompilerVersion) == "" || strings.TrimSpace(r.DisassemblerVersion) == "" {
		return cudareg.Resources{}, fmt.Errorf("register resources: tool provenance unavailable")
	}
	if len(r.CompilerVersion) > 4096 || len(r.DisassemblerVersion) > 4096 || len(r.CompilerOptions) > 128 {
		return cudareg.Resources{}, fmt.Errorf("register resources: tool metadata budget exceeded")
	}
	for _, option := range r.CompilerOptions {
		if len(option) > 4096 {
			return cudareg.Resources{}, fmt.Errorf("register resources: compiler option budget exceeded")
		}
	}
	resources := cudareg.Resources{ImageDigest: r.ImageDigest, Entry: r.Entry, Target: r.Target, DecodedTarget: r.DecodedTarget}
	for _, f := range []struct {
		name, category string
		fact           ResourceFact
		dst            *cudareg.Fact
	}{
		{"allocated", "decoded", r.Allocated, &resources.Allocated},
		{"peak live", "derived", r.PeakLive, &resources.PeakLive},
		{"cap", "compiler condition", r.RegisterCap, &resources.Cap},
		{"modeled", "modeled", r.ModelRegisters, &resources.Modeled},
	} {
		reason := f.fact.Reason
		if f.fact.Value == nil {
			if strings.TrimSpace(reason) == "" {
				reason = "not reported"
			}
			*f.dst = cudareg.Fact{Status: cudareg.FactUnavailable, Reason: reason}
			continue
		}
		if *f.fact.Value > uint64(^uint32(0)) {
			return cudareg.Resources{}, fmt.Errorf("register resources: %s exceeds uint32", f.name)
		}
		if *f.fact.Value == 0 && f.fact.Category == f.category && (f.name == "allocated" || f.name == "cap") {
			return cudareg.Resources{}, fmt.Errorf("register resources: zero %s", f.name)
		}
		*f.dst = cudareg.Fact{Count: uint32(*f.fact.Value), Status: cudareg.FactUnverified, Reason: reason}
		if f.fact.Category == f.category && reason == "" {
			f.dst.Status = cudareg.FactAvailable
		} else if strings.TrimSpace(reason) == "" {
			f.dst.Reason = fmt.Sprintf("category %q, want %q", f.fact.Category, f.category)
		}
	}
	if err := resources.Validate(); err != nil {
		return cudareg.Resources{}, err
	}
	return resources, nil
}
