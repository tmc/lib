package main

import (
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// SASS is what the machine runs, so it answers questions PTX cannot: PTX uses
// unlimited virtual registers and says nothing about the register pressure
// that decides occupancy.
//
// What is reported here is a floor, not the launch's register count. The exact
// figure is a function attribute the driver reports for a loaded kernel, which
// this program has no API for; the highest register index a function's
// operands mention is a lower bound on what ptxas allocated, and it excludes
// uniform and predicate registers entirely. It is reported as such.

// sassKernelStat summarizes one disassembled function.
type sassKernelStat struct {
	Name         string
	Instructions int
	MinRegisters int            // lower bound: highest Rn mentioned, plus one
	Opcodes      map[string]int // base opcode to count, e.g. "LDG" to 12
}

// sassKernelStats parses nvdisasm output and summarizes every function in it.
func sassKernelStats(disasm string) ([]sassKernelStat, error) {
	d, err := cudasass.ParseString(disasm)
	if err != nil {
		return nil, fmt.Errorf("parse sass: %w", err)
	}
	return summarizeSASS(d), nil
}

// summarizeSASS summarizes each function in a parsed disassembly.
func summarizeSASS(d *cudasass.Disassembly) []sassKernelStat {
	stats := make([]sassKernelStat, 0, len(d.Functions))
	for _, fn := range d.Functions {
		s := sassKernelStat{Name: fn.Name, Instructions: len(fn.Instructions), Opcodes: map[string]int{}}
		for _, inst := range fn.Instructions {
			s.Opcodes[baseOpcode(inst.Opcode)]++
			if n := maxRegisterIndex(inst.Operands); n+1 > s.MinRegisters {
				s.MinRegisters = n + 1
			}
		}
		stats = append(stats, s)
	}
	return stats
}

// baseOpcode strips the modifiers from a SASS mnemonic: LDG.E.CONSTANT.F32
// becomes LDG, so a histogram groups the loads together.
func baseOpcode(opcode string) string {
	if i := strings.IndexByte(opcode, '.'); i >= 0 {
		return opcode[:i]
	}
	return opcode
}

// maxRegisterIndex returns the highest general-purpose register index the
// operands mention, or -1 when they mention none.
//
// RZ is the zero register and is not an allocation, so it does not count. UR
// and P registers are separate files and are not counted either: including
// them would report a number that is neither the GPR count nor any other
// single quantity.
func maxRegisterIndex(operands string) int {
	max := -1
	for i := 0; i < len(operands); i++ {
		if operands[i] != 'R' {
			continue
		}
		// A register reference starts a token: anything but a delimiter before
		// it means this is the tail of another name, such as UR4 or SR_TID.
		if i > 0 && !isOperandDelim(operands[i-1]) {
			continue
		}
		j := i + 1
		for j < len(operands) && operands[j] >= '0' && operands[j] <= '9' {
			j++
		}
		if j == i+1 {
			continue // RZ, or a bare R
		}
		n, err := strconv.Atoi(operands[i+1 : j])
		if err == nil && n > max {
			max = n
		}
	}
	return max
}

func isOperandDelim(c byte) bool {
	switch c {
	case ' ', ',', '[', '(', '\t', '|', '-', '!', '{':
		return true
	}
	return false
}

// writeSASSReport writes one line per function: instruction count, the
// register floor, and the four most frequent opcodes.
func writeSASSReport(w io.Writer, disasm string) error {
	stats, err := sassKernelStats(disasm)
	if err != nil {
		return err
	}
	return writeSASSStats(w, stats)
}

// writeSASSStats writes summarized functions, sorted by name so the report is
// stable across runs.
func writeSASSStats(w io.Writer, stats []sassKernelStat) error {
	sort.Slice(stats, func(i, j int) bool { return stats[i].Name < stats[j].Name })
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "KERNEL\tINSTRUCTIONS\tREGISTERS>=\tTOP OPCODES")
	for _, s := range stats {
		fmt.Fprintf(tw, "%s\t%d\t%d\t%s\n", s.Name, s.Instructions, s.MinRegisters, topMnemonics(s.Opcodes, 4))
	}
	return tw.Flush()
}

// writeSASSDisassembly writes the report for an already parsed disassembly.
func writeSASSDisassembly(w io.Writer, d *cudasass.Disassembly) error {
	return writeSASSStats(w, summarizeSASS(d))
}
