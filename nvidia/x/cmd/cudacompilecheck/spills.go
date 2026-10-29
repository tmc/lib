package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"path/filepath"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// A localAccess is one STL or LDL instruction.
type localAccess struct {
	PC       uint64 `json:"pc"`
	Opcode   string `json:"opcode"`
	Operands string `json:"operands"`
	Spill    bool   `json:"spill"` // nvdisasm marks it SpillRefill; otherwise a local array or stack
	File     string `json:"file,omitempty"`
	Line     int    `json:"line,omitempty"`
	Loops    int    `json:"loops"` // natural loops that hold PC
}

// localAccesses returns fn's local-memory accesses, each with the number
// of natural loops that hold it (see [cudasass.CFG.Loops]). Incomplete
// reports that fn has an indirect branch, whose missing edges can hide a
// loop, or a cycle that is not a natural loop.
func localAccesses(fn *cudasass.Function) (acc []localAccess, incomplete bool, err error) {
	g, err := cudasass.NewCFG(fn)
	if err != nil {
		return nil, false, err
	}
	depth := make([]int, len(g.Blocks))
	for _, l := range g.Loops() {
		for _, b := range l.Blocks {
			depth[b]++
		}
	}
	for bi, b := range g.Blocks {
		for _, in := range b.Instructions {
			op := cudasass.BaseOpcode(in.Opcode)
			if op != "STL" && op != "LDL" {
				continue
			}
			operands, _, spill := strings.Cut(in.Operands, `(*"SpillRefill"*)`)
			acc = append(acc, localAccess{
				PC:       in.Address,
				Opcode:   in.Opcode,
				Operands: strings.TrimSpace(operands),
				Spill:    spill,
				File:     in.SourceFile,
				Line:     in.SourceLine,
				Loops:    depth[bi],
			})
		}
	}
	return acc, g.IsDegenerate || !g.Reducible(), nil
}

// disassembleLines extracts the arch cubin from the fatbin in outDir and
// disassembles it with line information, on the compiling host.
func disassembleLines(ctx context.Context, c *config, outDir, arch string) (*cudasass.Disassembly, error) {
	if text, err := c.compile(ctx, "sh", "-c", `cd "$1" && "$2" -xelf all out.fatbin`, "sh", outDir, c.cuobjdump); err != nil {
		return nil, fmt.Errorf("cuobjdump -xelf: %v%s\n%s", err, c.hint(err, c.cuobjdump, "-cuobjdump"), text)
	}
	text, err := c.compile(ctx, "sh", "-c", `cd "$1" && "$2" -g -c *."$3".cubin`, "sh", outDir, c.nvdisasm, arch)
	if err != nil {
		return nil, fmt.Errorf("nvdisasm: %v%s\n%s", err, c.hint(err, c.nvdisasm, "-nvdisasm"), text)
	}
	d, err := cudasass.Parse(bytes.NewReader(text))
	if err != nil {
		return nil, fmt.Errorf("nvdisasm %s: %w", arch, err)
	}
	return d, nil
}

// printLocals summarizes a function's local accesses by source line:
// spills in loops first, the ones that cost every iteration.
func printLocals(w io.Writer, arch, name string, acc []localAccess) {
	if len(acc) == 0 {
		return
	}
	type key struct {
		file  string
		line  int
		loops int
		spill bool
	}
	type group struct {
		key
		st, ld int
		first  uint64
	}
	groups := map[key]*group{}
	inLoop := 0
	for _, a := range acc {
		k := key{filepath.Base(a.File), a.Line, a.Loops, a.Spill}
		g := groups[k]
		if g == nil {
			g = &group{key: k, first: a.PC}
			groups[k] = g
		}
		if strings.HasPrefix(a.Opcode, "STL") {
			g.st++
		} else {
			g.ld++
		}
		if a.Loops > 0 {
			inLoop++
		}
	}
	gs := make([]*group, 0, len(groups))
	for _, g := range groups {
		gs = append(gs, g)
	}
	sort.Slice(gs, func(i, j int) bool {
		if gs[i].loops != gs[j].loops {
			return gs[i].loops > gs[j].loops
		}
		return gs[i].first < gs[j].first
	})
	fmt.Fprintf(w, "  local %s %s: %d accesses, %d inside loops\n", arch, name, len(acc), inLoop)
	for _, g := range gs {
		where := "outside loops"
		if g.loops > 0 {
			where = fmt.Sprintf("loop depth %d", g.loops)
		}
		kind := "spill"
		if !g.spill {
			kind = "local"
		}
		loc := "?"
		if g.file != "" {
			loc = fmt.Sprintf("%s:%d", g.file, g.line)
		}
		fmt.Fprintf(w, "    %s %s, %s: %d STL, %d LDL (first at %#04x)\n", loc, kind, where, g.st, g.ld, g.first)
	}
}
