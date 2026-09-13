package cudasass

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// RegisterPressure records live register counts at a specific instruction.
type RegisterPressure struct {
	GPR   int // Live General Purpose Registers (GPR)
	Pred  int // Live Predicate registers (PRED)
	UGPR  int // Live Uniform GPRs (UGPR, sm_70+)
	UPred int // Live Uniform Predicate registers (UPRED, sm_70+)
}

// ParseLifeRangeCount parses nvdisasm output formatted with --life-range-mode count.
// It returns a map of kernel name to instruction address to RegisterPressure.
//
// In count mode, nvdisasm formats trailing table columns:
//
//	/*0000*/ LDC R1, c[0x0][0x37c] ; // |   1   |        |        |         |
//
// Column order is GPR, PRED, UGPR, UPRED. A blank cell represents zero.
func ParseLifeRangeCount(r io.Reader) (map[string]map[uint64]RegisterPressure, error) {
	scanner := bufio.NewScanner(r)
	res := make(map[string]map[uint64]RegisterPressure)

	var curKernel string
	for scanner.Scan() {
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}

		if fnName := extractFunctionName(trimmed); fnName != "" {
			curKernel = fnName
			if res[curKernel] == nil {
				res[curKernel] = make(map[uint64]RegisterPressure)
			}
			continue
		}

		// Look for instruction offset /*hhhh*/ or /* 0xhhhh */
		if !strings.HasPrefix(trimmed, "/*") {
			continue
		}
		closeIdx := strings.Index(trimmed, "*/")
		if closeIdx == -1 {
			continue
		}
		addrStr := strings.TrimSpace(trimmed[2:closeIdx])
		addrStr = strings.TrimPrefix(addrStr, "0x")
		addr, err := strconv.ParseUint(addrStr, 16, 64)
		if err != nil {
			continue
		}

		// Look for comment delimiter // followed by table columns | col1 | col2 | ...
		commentIdx := strings.Index(line, "//")
		if commentIdx == -1 {
			continue
		}
		comment := strings.TrimSpace(line[commentIdx+2:])
		if !strings.HasPrefix(comment, "|") {
			continue
		}

		// Split columns: | GPR | PRED | UGPR | UPRED |
		cols := strings.Split(comment, "|")
		// cols[0] is empty (before first |).
		// col 1: GPR, col 2: PRED, col 3: UGPR, col 4: UPRED.
		var press RegisterPressure
		if len(cols) > 1 {
			press.GPR = parseCountCell(cols[1])
		}
		if len(cols) > 2 {
			press.Pred = parseCountCell(cols[2])
		}
		if len(cols) > 3 {
			press.UGPR = parseCountCell(cols[3])
		}
		if len(cols) > 4 {
			press.UPred = parseCountCell(cols[4])
		}

		kernelKey := curKernel
		if kernelKey == "" {
			kernelKey = "default"
		}
		if res[kernelKey] == nil {
			res[kernelKey] = make(map[uint64]RegisterPressure)
		}
		res[kernelKey][addr] = press
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("cudasass: scan life range: %w", err)
	}
	return res, nil
}

func parseCountCell(s string) int {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0
	}
	v, err := strconv.Atoi(s)
	if err != nil {
		return 0
	}
	return v
}

// JoinPressure performs a total join between a parsed Disassembly (Pass 1) and
// a register pressure map from --life-range-mode count (Pass 2).
//
// Every instruction present in Pass 2 must match an instruction in Pass 1.
// Every instruction in Pass 1 must match an entry in Pass 2, with the sole
// exception of alignment NOP padding at the end of a section, which is given 0 pressure.
func JoinPressure(disasm *Disassembly, pressureMap map[string]map[uint64]RegisterPressure) error {
	if disasm == nil {
		return fmt.Errorf("cudasass: nil disassembly")
	}

	for _, fn := range disasm.Functions {
		kmap := pressureMap[fn.Name]
		if kmap == nil && len(disasm.Functions) == 1 && len(pressureMap) == 1 {
			// Single-function fallback if names differ slightly (e.g. default vs mangled)
			for _, m := range pressureMap {
				kmap = m
				break
			}
		}
		if kmap == nil {
			return fmt.Errorf("cudasass: kernel %q found in pass 1 but missing in pass 2 register life ranges", fn.Name)
		}

		// Make a copy of keys so we can track visited addresses and enforce total join
		unmatchedPass2 := make(map[uint64]RegisterPressure, len(kmap))
		for k, v := range kmap {
			unmatchedPass2[k] = v
		}

		for i := range fn.Instructions {
			inst := &fn.Instructions[i]
			if press, ok := kmap[inst.Address]; ok {
				inst.LiveGPR = press.GPR
				inst.LivePred = press.Pred
				inst.LiveUGPR = press.UGPR
				inst.LiveUPred = press.UPred
				delete(unmatchedPass2, inst.Address)
			} else {
				// nvdisasm register dataflow analysis stops at kernel exit (e.g. EXIT / RET),
				// omitting trailing .align 128 NOP padding. Trailing NOPs have 0 live registers.
				if inst.Opcode == "NOP" {
					inst.LiveGPR = 0
					inst.LivePred = 0
					inst.LiveUGPR = 0
					inst.LiveUPred = 0
					continue
				}
				return fmt.Errorf("cudasass: total join failed: kernel %q instruction at 0x%x (%s) missing from pass 2 register life ranges", fn.Name, inst.Address, inst.Opcode)
			}
		}

		if len(unmatchedPass2) > 0 {
			var orphanAddr uint64
			for addr := range unmatchedPass2 {
				orphanAddr = addr
				break
			}
			return fmt.Errorf("cudasass: total join failed: kernel %q has pass 2 register life range at 0x%x not present in pass 1", fn.Name, orphanAddr)
		}
	}

	return nil
}

// DisassembleWithPressure executes a dual-pass disassembly on a CUBIN ELF image:
// Pass 1 disassembles instructions with source, inline, and PTX line information.
// Pass 2 collects live register pressure (--life-range-mode count).
// The passes are joined on instruction offset, populating LiveGPR, LivePred,
// LiveUGPR, and LiveUPred on each instruction.
func (d Disassembler) DisassembleWithPressure(ctx context.Context, cubin []byte) (*Disassembly, error) {
	// Pass 1: Line and instruction structure (-c -gi -gp)
	p1Out, err := d.Disassemble(ctx, cubin, Options{
		Mode:        Text,
		PrintCode:   true,
		SourceLines: true,
		InlineLines: true,
		PTXLines:    true,
	})
	if err != nil {
		return nil, fmt.Errorf("cudasass: pass 1 line info disassembly: %w", err)
	}

	disasm, err := Parse(bytes.NewReader(p1Out.Stdout))
	if err != nil {
		return nil, fmt.Errorf("cudasass: parse pass 1 disassembly: %w", err)
	}

	// Pass 2: Register life-range counts (-c -plr -lrm count)
	p2Out, err := d.Disassemble(ctx, cubin, Options{
		Mode:          RegisterLife,
		LifeRangeMode: "count",
		PrintCode:     true,
	})
	if err != nil {
		return nil, fmt.Errorf("cudasass: pass 2 register life disassembly: %w", err)
	}

	pressureMap, err := ParseLifeRangeCount(bytes.NewReader(p2Out.Stdout))
	if err != nil {
		return nil, fmt.Errorf("cudasass: parse pass 2 life ranges: %w", err)
	}

	if err := JoinPressure(disasm, pressureMap); err != nil {
		return nil, err
	}

	return disasm, nil
}

// DisassembleWithPressure invokes nvdisasm twice on cubin bytes to produce a
// structured Disassembly with line info and live register pressure joined.
func DisassembleWithPressure(cubin []byte) (*Disassembly, error) {
	return (Disassembler{}).DisassembleWithPressure(context.Background(), cubin)
}
