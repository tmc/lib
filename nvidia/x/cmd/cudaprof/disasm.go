package main

import (
	"fmt"
	"io"
	"regexp"
	"slices"
	"strings"

	"github.com/google/pprof/profile"
)

// retainedDisasm provides code navigation when the profile has no addressed
// samples for pprof to annotate. It reuses the object tool's parsed artifacts;
// matching a symbol here makes no claim that a region executed that kernel.
func retainedDisasm(w io.Writer, args []string, p *profile.Profile, tool *objectTool) (bool, error) {
	expression := ""
	var rest []string
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if strings.HasPrefix(arg, "-disasm=") || strings.HasPrefix(arg, "--disasm=") {
			expression = strings.SplitN(arg, "=", 2)[1]
		} else if arg == "-disasm" || arg == "--disasm" {
			if i+1 == len(args) {
				return true, fmt.Errorf("missing disassembly expression")
			}
			i++
			expression = args[i]
		} else {
			rest = append(rest, arg)
		}
	}
	if expression == "" {
		return false, nil
	}
	for _, sample := range p.Sample {
		for _, loc := range sample.Location {
			if loc.Address != 0 && loc.Mapping != nil {
				return false, nil
			}
		}
	}
	// Do not silently ignore pprof filtering, diffing, or output flags. Those
	// options apply to measurements, whereas this fallback prints retained code.
	if len(rest) != 1 || strings.HasPrefix(rest[0], "-") {
		return true, fmt.Errorf("unannotated retained disassembly requires only -disasm=regexp and a capture path")
	}
	re, err := regexp.Compile(expression)
	if err != nil {
		return true, err
	}
	if err := tool.loadRetainedImages(); err != nil {
		return true, err
	}
	var functions []objectFunction
	for _, image := range tool.images {
		for _, fn := range image.functions {
			if re.MatchString(fn.name) && len(fn.instructions) > 0 {
				functions = append(functions, fn)
			}
		}
	}
	if len(functions) == 0 {
		return true, fmt.Errorf("no retained code matches regexp %s", expression)
	}
	slices.SortFunc(functions, func(a, b objectFunction) int {
		if n := strings.Compare(a.name, b.name); n != 0 {
			return n
		}
		return strings.Compare(a.artifact, b.artifact)
	})
	if _, err := fmt.Fprintln(w, "Retained code (unannotated): no mapped profile addresses.\n"+
		"No instruction costs or associations with profile regions are inferred.\n"+
		"SASS/PTX use artifact file offsets; Tile IR uses source lines. These are not device PCs."); err != nil {
		return true, err
	}
	for _, fn := range functions {
		if _, err := fmt.Fprintf(w, "\n%s\n", fn.name); err != nil {
			return true, err
		}
		if fn.artifact != "" {
			if _, err := fmt.Fprintf(w, "  artifact: %s\n", fn.artifact); err != nil {
				return true, err
			}
		}
		for _, in := range fn.instructions {
			if _, err := fmt.Fprintf(w, "  %#x  %s\n", in.Addr, in.Text); err != nil {
				return true, err
			}
		}
	}
	return true, nil
}
