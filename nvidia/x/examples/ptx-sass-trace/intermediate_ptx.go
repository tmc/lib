package main

import (
	"bufio"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/tileiras"
)

var tileIREntryRe = regexp.MustCompile(`cuda_tile\.entry\s+@([a-zA-Z0-9_]+)`)

// kernelTileIRLines parses kernels.tileir text and returns a map of kernel name
// to 1-based line number in kernels.tileir.
func kernelTileIRLines(tileIRText string) map[string]int {
	m := make(map[string]int)
	scanner := bufio.NewScanner(strings.NewReader(tileIRText))
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := scanner.Text()
		if matches := tileIREntryRe.FindStringSubmatch(line); len(matches) > 1 {
			m[matches[1]] = lineNum
		}
	}
	return m
}

// injectTileIRLocations enhances generated PTX with PTX ISA 9.4 .file 2 "kernels.tileir"
// and .loc_intermediate directives linking each kernel back to its CUDA Tile IR definition.
func injectTileIRLocations(ptx, tileIRText string) string {
	tileLines := kernelTileIRLines(tileIRText)
	if len(tileLines) == 0 {
		return ptx
	}

	lines := strings.Split(ptx, "\n")
	var out []string
	fileDirectiveInserted := false

	for i := 0; i < len(lines); i++ {
		line := lines[i]
		out = append(out, line)

		// Insert .file 2 directive right after the first .file directive
		if !fileDirectiveInserted && strings.HasPrefix(strings.TrimSpace(line), ".file") {
			out = append(out, "\t.file\t2 \"kernels.tileir\"")
			fileDirectiveInserted = true
		}

		// When entering a kernel definition, attach .loc_intermediate
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, ".visible .entry ") || strings.HasPrefix(trimmed, ".entry ") {
			// Find kernel name
			for kname, tileLine := range tileLines {
				if strings.Contains(trimmed, kname) {
					// Scan ahead to find opening brace or first instruction
					for j := i + 1; j < len(lines) && j <= i+10; j++ {
						if strings.Contains(lines[j], "{") {
							// Append the remaining lines up to {
							for k := i + 1; k <= j; k++ {
								out = append(out, lines[k])
							}
							i = j
							out = append(out, fmt.Sprintf("\t.loc_intermediate 2 %d 1", tileLine))
							break
						}
					}
					break
				}
			}
		}
	}

	// Fallback if no .file directive was in the PTX
	if !fileDirectiveInserted {
		var header []string
		header = append(header, "\t.file\t2 \"kernels.tileir\"")
		out = append(header, out...)
	}

	return strings.Join(out, "\n")
}

// tileIRRemarkSummary renders the tileiras optimization remarks against the
// bundled tile IR. Every remark names the function it fired on, so each one is
// resolved to its cuda_tile.entry line in kernels.tileir: a remark naming a
// function this bundle does not define describes some other compilation, and a
// reader has nothing to check it against. That mismatch is an error rather than
// an omitted line, because a silently dropped remark and a kernel the compiler
// found nothing to say about render identically.
//
// The kernels that drew no remark are listed too. "No remark" on a kernel the
// tile IR does define is a fact about the compilation; on a kernel the remarks
// were generated from a different source, it is an artifact of the mismatch,
// and only the roster tells the two apart.
func tileIRRemarkSummary(remarksYAML, tileIRText string) (string, error) {
	remarks, err := tileiras.ParseRemarksYAML([]byte(remarksYAML))
	if err != nil {
		return "", fmt.Errorf("parse tileiras remarks: %w", err)
	}
	if len(remarks) == 0 {
		return "", errors.New("parse tileiras remarks: no remarks in file")
	}
	tileLines := kernelTileIRLines(tileIRText)
	if len(tileLines) == 0 {
		return "", errors.New("parse tileiras remarks: no cuda_tile.entry in tile ir")
	}

	var b strings.Builder
	remarked := make(map[string]bool)
	for _, r := range remarks {
		line, ok := tileLines[r.Function]
		if !ok {
			return "", fmt.Errorf("tileiras remark %q names function %q, absent from %s", r.Name, r.Function, tileIRArtifact)
		}
		remarked[r.Function] = true
		// The remark text arrives under Args["Remark"] from the compiler and
		// under Args["String"] from the LLVM-style emitter; both spellings are
		// the message a reader wants.
		msg := r.Remark
		if msg == "" {
			msg = r.Args["String"]
		}
		fmt.Fprintf(&b, "[%s/%s] %s at %s:%d\n", r.Pass, r.Kind, r.Name, tileIRArtifact, line)
		fmt.Fprintf(&b, "\t%s: %s\n", r.Function, msg)
	}

	var silent []string
	for name := range tileLines {
		if !remarked[name] {
			silent = append(silent, name)
		}
	}
	slices.Sort(silent)
	fmt.Fprintf(&b, "\n%d remarks over %d of %d tile IR kernels\n", len(remarks), len(remarked), len(tileLines))
	if len(silent) > 0 {
		fmt.Fprintf(&b, "no remarks: %s\n", strings.Join(silent, " "))
	}
	return b.String(), nil
}
