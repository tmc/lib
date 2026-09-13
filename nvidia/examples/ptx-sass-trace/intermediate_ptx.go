package main

import (
	"bufio"
	"fmt"
	"regexp"
	"strings"
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
