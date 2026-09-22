package cudaview

import (
	"path"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// Annotate resolves instruction and source identities from retained text.
// Missing source file identities remain empty; a line number alone is not a match.
func Annotate(tier *PipelineTier) {
	var text strings.Builder
	for i, l := range tier.Lines {
		if i > 0 {
			text.WriteByte('\n')
		}
		text.WriteString(l.Text)
	}
	switch tier.Language {
	case "go", "cuda":
		for i := range tier.Lines {
			tier.Lines[i].SourceFile = sourceName(tier.FileName)
			tier.Lines[i].SourceLine = tier.Lines[i].Num
		}
	case "ptx":
		module, err := cudaptx.ParseString(text.String())
		if err != nil {
			return
		}
		for _, kernel := range module.Kernels {
			for _, inst := range kernel.Instructions {
				if inst.Line < 1 || inst.Line > len(tier.Lines) {
					continue
				}
				line := &tier.Lines[inst.Line-1]
				line.SourceFile = sourceName(inst.Loc.File)
				line.SourceLine = inst.Loc.Line
				line.Opcode = inst.Opcode
				line.Predicate = inst.Predicate
				line.Operands = inst.Operands
				line.Kernel = kernel.Name
			}
		}
	case "sass":
		disasm, err := cudasass.ParseString(text.String())
		if err != nil {
			return
		}
		type entry struct {
			kernel string
			inst   cudasass.Instruction
		}
		entries := make(map[string][]entry)
		for _, fn := range disasm.Functions {
			for _, inst := range fn.Instructions {
				key := strings.TrimSpace(inst.Raw)
				entries[key] = append(entries[key], entry{fn.Name, inst})
			}
		}
		for i := range tier.Lines {
			line := &tier.Lines[i]
			key := strings.TrimSpace(line.Text)
			queue := entries[key]
			if len(queue) == 0 {
				continue
			}
			e := queue[0]
			entries[key] = queue[1:]
			line.SourceFile = sourceName(e.inst.SourceFile)
			if e.inst.SourceFile != "" {
				line.SourceLine = e.inst.SourceLine
			}
			line.Kernel = e.kernel
			line.Opcode = e.inst.Opcode
			line.Predicate = e.inst.Predicate
			line.Operands = e.inst.Operands
			if e.inst.Encoding.Present() {
				line.Encoding = e.inst.Encoding.String()
			}
		}
	}
}

// sourceName is the name by which every tier refers to a source file. The
// PTX and the disassembly record the path the compiler was given, which is a
// build-host path; the listings name the file as the capture bundles it. Only
// the base name is common to all of them.
func sourceName(file string) string {
	if file == "" {
		return ""
	}
	return path.Base(file)
}
