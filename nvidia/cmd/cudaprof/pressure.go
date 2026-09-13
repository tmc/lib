package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

func runPressureCommand(rawArgs []string) error {
	args, limits, err := parseReadLimits(rawArgs)
	if err != nil {
		return err
	}
	if len(args) == 0 {
		return errors.New("usage: cudaprof pressure [kernel] capture.pprof")
	}

	cmd := args[0]
	var kernel string
	switch {
	case cmd == "pressure" || cmd == "-pressure" || cmd == "--pressure":
		args = args[1:]
	case strings.HasPrefix(cmd, "-pressure="):
		kernel = strings.TrimPrefix(cmd, "-pressure=")
		args = args[1:]
	case strings.HasPrefix(cmd, "--pressure="):
		kernel = strings.TrimPrefix(cmd, "--pressure=")
		args = args[1:]
	}

	var capturePath string
	switch len(args) {
	case 1:
		capturePath = args[0]
	case 2:
		if kernel != "" {
			return errors.New("kernel specified multiple times")
		}
		kernel = args[0]
		capturePath = args[1]
	default:
		return errors.New("usage: cudaprof pressure [kernel] capture.pprof")
	}

	path, err := filepath.Abs(capturePath)
	if err != nil {
		return err
	}
	cap, err := readCapturePathLimits(path, limits)
	if err != nil {
		return fmt.Errorf("read capture: %w", err)
	}
	return printCapturePressure(os.Stdout, cap, kernel)
}

func printCapturePressure(w io.Writer, cap *cudacapture.Capture, kernel string) error {
	disasm, err := loadDisassemblyWithPressure(cap)
	if err != nil {
		return err
	}

	var targets []*cudasass.Function
	if kernel != "" {
		fn := disasm.Function(kernel)
		if fn == nil {
			for _, f := range disasm.Functions {
				if strings.EqualFold(f.Name, kernel) || strings.Contains(f.Name, kernel) {
					fn = f
					break
				}
			}
		}
		if fn == nil {
			return fmt.Errorf("kernel %q not found in disassembly", kernel)
		}
		targets = []*cudasass.Function{fn}
	} else {
		targets = disasm.Functions
	}

	if len(targets) == 0 {
		return errors.New("no functions found in disassembly")
	}

	code, _ := cudacapture.LoadCode(cap)
	stalls := extractStallMetrics(cap, code, disasm)

	for i, fn := range targets {
		if i > 0 {
			fmt.Fprintln(w)
		}
		fmt.Fprintf(w, "Kernel: %s\n", fn.Name)

		tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
		if stalls.hasStalls {
			fmt.Fprintln(tw, "PC\tSTALL%\tGPR\tPRED\tUGPR\tUPRED\tINSTRUCTION")
		} else {
			fmt.Fprintln(tw, "PC\tGPR\tPRED\tUGPR\tUPRED\tINSTRUCTION")
		}

		kernelTotal := stalls.byKernel[fn.Name]
		if kernelTotal == 0 && len(disasm.Functions) == 1 {
			for _, v := range stalls.byKernel {
				kernelTotal = v
				break
			}
		}

		for _, inst := range fn.Instructions {
			instText := formatInstruction(inst)
			if stalls.hasStalls {
				var instStall int64
				if pcMap, ok := stalls.byPC[fn.Name]; ok {
					instStall = pcMap[inst.Address]
				} else if len(disasm.Functions) == 1 {
					for _, pcMap := range stalls.byPC {
						instStall = pcMap[inst.Address]
						break
					}
				}
				var pct float64
				if kernelTotal > 0 {
					pct = float64(instStall) / float64(kernelTotal) * 100.0
				}
				fmt.Fprintf(tw, "0x%04x\t%5.1f%%\t%d\t%d\t%d\t%d\t%s\n",
					inst.Address, pct, inst.LiveGPR, inst.LivePred, inst.LiveUGPR, inst.LiveUPred, instText)
			} else {
				fmt.Fprintf(tw, "0x%04x\t%d\t%d\t%d\t%d\t%s\n",
					inst.Address, inst.LiveGPR, inst.LivePred, inst.LiveUGPR, inst.LiveUPred, instText)
			}
		}
		if err := tw.Flush(); err != nil {
			return err
		}
	}

	return nil
}

func loadDisassemblyWithPressure(cap *cudacapture.Capture) (*cudasass.Disassembly, error) {
	if cap == nil {
		return nil, errors.New("nil capture")
	}

	// 1. Try reading pre-joined pressure from capture files if present.
	if disasm, err := extractPreJoinedPressure(cap); err == nil && disasm != nil {
		return disasm, nil
	}

	// 2. If pre-joined pressure is not present, extract CUBIN and run DisassembleWithPressure.
	cubinData, ok := findCUBIN(cap)
	if ok && len(cubinData) > 0 {
		disasm, err := cudasass.DisassembleWithPressure(cubinData)
		if err != nil {
			return nil, fmt.Errorf("disassemble with pressure: %w", err)
		}
		return disasm, nil
	}

	return nil, errors.New("capture contains neither CUBIN nor SASS disassembly with register pressure")
}

func extractPreJoinedPressure(cap *cudacapture.Capture) (*cudasass.Disassembly, error) {
	sassData, ok := findSASS(cap)
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

func findCUBIN(cap *cudacapture.Capture) ([]byte, bool) {
	if cap.Code.CUBIN != "" {
		if data, ok := cap.Files[cap.Code.CUBIN]; ok && len(data) > 0 {
			return data, true
		}
	}
	if data, ok := cap.Files["kernels.cubin"]; ok && len(data) > 0 {
		return data, true
	}
	for name, data := range cap.Files {
		if strings.HasSuffix(name, ".cubin") && len(data) > 0 {
			return data, true
		}
	}
	return nil, false
}

func findSASS(cap *cudacapture.Capture) ([]byte, bool) {
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

type kernelStalls struct {
	hasStalls bool
	byPC      map[string]map[uint64]int64
	byKernel  map[string]int64
}

func extractStallMetrics(cap *cudacapture.Capture, code *cudacapture.Code, disasm *cudasass.Disassembly) kernelStalls {
	if cap == nil || cap.Profile == nil || len(cap.Profile.Sample) == 0 {
		return kernelStalls{}
	}

	var stallIndices []int
	for i, st := range cap.Profile.SampleType {
		if strings.HasPrefix(st.Type, "stall_") {
			stallIndices = append(stallIndices, i)
		}
	}
	if len(stallIndices) == 0 {
		return kernelStalls{}
	}

	res := kernelStalls{
		byPC:     make(map[string]map[uint64]int64),
		byKernel: make(map[string]int64),
	}

	for _, s := range cap.Profile.Sample {
		var stalls int64
		for _, idx := range stallIndices {
			stalls += s.Value[idx]
		}
		if stalls == 0 || len(s.Location) == 0 {
			continue
		}
		loc := s.Location[0]
		kName, pc := resolveLocationPC(loc, s, code, disasm)
		if kName == "" {
			continue
		}
		if res.byPC[kName] == nil {
			res.byPC[kName] = make(map[uint64]int64)
		}
		res.byPC[kName][pc] += stalls
		res.byKernel[kName] += stalls
		res.hasStalls = true
	}

	return res
}

func resolveLocationPC(loc *profile.Location, s *profile.Sample, code *cudacapture.Code, disasm *cudasass.Disassembly) (string, uint64) {
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

	return kName, loc.Address
}

func formatInstruction(inst cudasass.Instruction) string {
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
