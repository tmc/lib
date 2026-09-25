package capturefiles

import (
	"bytes"
	"errors"
	"fmt"
	"strings"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/tileiras"
)

// PTX returns the PTX listing a capture carries and its file name: the file
// the capture's code names, else kernels.ptx, else any .ptx file. It returns
// nil data when there is none; the error is always nil.
func PTX(cap *cudacapture.Capture) ([]byte, string, error) {
	if cap == nil || len(cap.Files) == 0 {
		return nil, "", nil
	}
	if cap.Code.PTX != "" {
		if data, ok := cap.Files[cap.Code.PTX]; ok && len(data) > 0 {
			return data, cap.Code.PTX, nil
		}
	}
	if data, ok := cap.Files["kernels.ptx"]; ok && len(data) > 0 {
		return data, "kernels.ptx", nil
	}
	for name, data := range cap.Files {
		if strings.HasSuffix(name, ".ptx") && len(data) > 0 {
			return data, name, nil
		}
	}
	return nil, "", nil
}

// Pressure parses the capture's SASS listing and joins to it the register
// life-range counts found in a separate file or in the listing itself.
func Pressure(cap *cudacapture.Capture) (*cudasass.Disassembly, error) {
	sassData, ok := SASS(cap)
	if !ok || len(sassData) == 0 {
		return nil, errors.New("no SASS file found")
	}

	// Look for a separate life-range count file in cap.Files.
	var lrmData []byte
	candidates := []string{"kernels.lrm", "kernels.pressure", "life_ranges.txt", "life_ranges.count"}
	for _, name := range candidates {
		if d, ok := cap.Files[name]; ok && len(d) > 0 {
			lrmData = d
			break
		}
	}
	if len(lrmData) == 0 {
		for name, d := range cap.Files {
			if (strings.HasSuffix(name, ".lrm") || strings.HasSuffix(name, ".pressure") || strings.HasSuffix(name, ".count")) && len(d) > 0 {
				lrmData = d
				break
			}
		}
	}

	var pMap map[string]map[uint64]cudasass.RegisterPressure
	var err error
	if len(lrmData) > 0 {
		pMap, err = cudasass.ParseLifeRangeCount(bytes.NewReader(lrmData))
	}
	if len(pMap) == 0 {
		// Try parsing life ranges from the SASS file itself (if disassembled with --life-range-mode count).
		pMap, err = cudasass.ParseLifeRangeCount(bytes.NewReader(sassData))
	}
	if err != nil || len(pMap) == 0 {
		return nil, errors.New("no register life ranges found in capture files")
	}

	disasm, err := cudasass.Parse(bytes.NewReader(sassData))
	if err != nil {
		return nil, fmt.Errorf("parse SASS: %w", err)
	}

	if err := cudasass.JoinPressure(disasm, pMap); err != nil {
		return nil, fmt.Errorf("join pressure: %w", err)
	}

	return disasm, nil
}

// SASS returns the SASS listing a capture carries: the file the capture's
// code names, else kernels.sass, else any .sass file.
func SASS(cap *cudacapture.Capture) ([]byte, bool) {
	if cap.Code.SASS != "" {
		if data, ok := cap.Files[cap.Code.SASS]; ok && len(data) > 0 {
			return data, true
		}
	}
	if data, ok := cap.Files["kernels.sass"]; ok && len(data) > 0 {
		return data, true
	}
	for name, data := range cap.Files {
		if strings.HasSuffix(name, ".sass") && len(data) > 0 {
			return data, true
		}
	}
	return nil, false
}

// LocationPC resolves a sampled location to the kernel it is in and its PC
// offset from that kernel's start, using code or, failing that, disasm to
// place the kernel.
func LocationPC(loc *profile.Location, s *profile.Sample, code *cudacapture.Code, disasm *cudasass.Disassembly) (string, uint64) {
	var kName string
	for _, line := range loc.Line {
		if line.Function != nil && line.Function.Name != "" {
			name := line.Function.Name
			name = strings.TrimSuffix(name, ".SASS")
			name = strings.TrimSuffix(name, ".PTX")
			name = strings.TrimSuffix(name, ".TILEIR")
			kName = name
			break
		}
	}
	if kName == "" && s.Label != nil && len(s.Label["kernel"]) > 0 {
		kName = s.Label["kernel"][0]
	}
	if kName == "" && disasm != nil && len(disasm.Functions) == 1 {
		kName = disasm.Functions[0].Name
	}

	base := cudaprof.DefaultMappingStart
	if loc.Mapping != nil && loc.Mapping.Start > 0 {
		base = loc.Mapping.Start
	}

	if code != nil {
		if k := code.Kernel(kName); k != nil {
			if loc.Address >= base+k.Offset && (k.Size == 0 || loc.Address < base+k.Offset+k.Size) {
				return kName, loc.Address - (base + k.Offset)
			}
			if loc.Address >= k.Offset && (k.Size == 0 || loc.Address < k.Offset+k.Size) {
				return kName, loc.Address - k.Offset
			}
			if loc.Address < k.Size {
				return kName, loc.Address
			}
		}
	}

	if disasm != nil {
		fn := disasm.Function(kName)
		if fn == nil && len(disasm.Functions) == 1 {
			fn = disasm.Functions[0]
		}
		if fn != nil {
			if loc.Address >= base {
				addr := loc.Address - base
				if fn.BaseAddress > 0 && addr >= fn.BaseAddress {
					return kName, addr - fn.BaseAddress
				}
				return kName, addr
			}
			if fn.BaseAddress > 0 && loc.Address >= fn.BaseAddress {
				return kName, loc.Address - fn.BaseAddress
			}
			return kName, loc.Address
		}
	}

	if loc.Address >= base {
		return kName, loc.Address - base
	}

	return kName, loc.Address
}

// Instruction formats inst as one line of SASS.
func Instruction(inst cudasass.Instruction) string {
	if inst.Opcode == "" {
		return strings.TrimSpace(inst.Raw)
	}
	var sb strings.Builder
	if inst.Predicate != "" {
		sb.WriteString(inst.Predicate)
		sb.WriteString(" ")
	}
	sb.WriteString(inst.Opcode)
	if inst.Operands != "" {
		sb.WriteString(" ")
		sb.WriteString(strings.TrimSuffix(inst.Operands, ";"))
	}
	sb.WriteString(";")
	return sb.String()
}

// LoadCode loads a capture's code tiers and separates a source-fusion
// failure from every other load failure.
//
// Since [cudacapture.LoadCode] began returning tiers rather than all or nothing, a
// source-fusion failure comes back alongside a usable Code whose source
// attribution has been zeroed: every instruction's file is empty and its line
// is 0 because the binding between the PTX and the SASS was refused, not
// because the compiler located nothing. Those two states are indistinguishable
// once rendered, so a caller that discards the error prints the refusal as if
// it were the answer. The failure is returned separately here so each caller
// states it in whatever channel its own output has.
//
// A nonnil error is every other load failure. Callers that already tolerate a
// nil Code keep doing so; the error is returned rather than dropped so they can
// say which tier they are rendering without.
func LoadCode(cap *cudacapture.Capture) (*cudacapture.Code, *cudacapture.SourceFusionError, error) {
	code, err := cudacapture.LoadCode(cap)
	if err == nil {
		return code, nil, nil
	}
	var fusion *cudacapture.SourceFusionError
	if errors.As(err, &fusion) && code != nil {
		return code, fusion, nil
	}
	return code, nil, err
}

// Remarks returns the tileiras remarks a capture carries and the name of the
// file they were read from, or nil when it carries none.
func Remarks(cap *cudacapture.Capture) ([]tileiras.Remark, string, error) {
	if cap == nil || len(cap.Files) == 0 {
		return nil, "", nil
	}
	candidateNames := []string{"tileiras-remarks.yaml", "remarks.yaml", "remarks.yml"}
	for _, name := range candidateNames {
		if d, ok := cap.Files[name]; ok {
			rems, err := tileiras.ParseRemarksYAML(d)
			if err == nil && len(rems) > 0 {
				return rems, name, nil
			}
		}
	}
	for name, d := range cap.Files {
		if strings.HasSuffix(name, ".yaml") || strings.HasSuffix(name, ".yml") {
			if strings.Contains(string(d), "--- !") || strings.Contains(string(d), "Remark") {
				rems, err := tileiras.ParseRemarksYAML(d)
				if err == nil && len(rems) > 0 {
					return rems, name, nil
				}
			}
		}
	}
	return nil, "", nil
}
