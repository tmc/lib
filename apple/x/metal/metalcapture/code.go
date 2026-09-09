package metalcapture

import (
	"bufio"
	"fmt"
	"strings"
)

// Code holds parsed function and instruction information from captured code files.
type Code struct {
	Functions []*Function
	funcMap   map[string]*Function
}

// Function represents a GPU compute kernel function with optional instructions.
type Function struct {
	Name         string
	StartLine    int
	EndLine      int
	Instructions []Instruction
}

// Instruction represents a line or instruction within an IR/disassembly representation.
type Instruction struct {
	Line   int    // 1-indexed line number in the source or IR file
	Offset uint64 // Logical offset or address
	Text   string // Text of the instruction
}

// LoadCode parses the attached source and/or IR files from the capture.
func LoadCode(c *Capture) (*Code, error) {
	if c == nil {
		return nil, fmt.Errorf("metalcapture: nil capture")
	}

	code := &Code{
		funcMap: make(map[string]*Function),
	}

	// Parse MSL source file
	srcBytes, ok := c.Files[c.Code.Source]
	if !ok {
		return nil, fmt.Errorf("metalcapture: source file %q not found in capture", c.Code.Source)
	}

	scanner := bufio.NewScanner(strings.NewReader(string(srcBytes)))
	lineNum := 0
	var curFn *Function

	for scanner.Scan() {
		lineNum++
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)

		if strings.HasPrefix(trimmed, "kernel ") || strings.Contains(trimmed, "kernel void ") {
			// Extract function name
			parts := strings.Fields(trimmed)
			for i, p := range parts {
				if p == "void" && i+1 < len(parts) {
					fnName := parts[i+1]
					if parenIdx := strings.Index(fnName, "("); parenIdx != -1 {
						fnName = fnName[:parenIdx]
					}
					curFn = &Function{
						Name:      fnName,
						StartLine: lineNum,
					}
					code.Functions = append(code.Functions, curFn)
					code.funcMap[fnName] = curFn
					break
				}
			}
		}

		if curFn != nil {
			if strings.Contains(line, ";") || strings.Contains(line, "=") {
				curFn.Instructions = append(curFn.Instructions, Instruction{
					Line: lineNum,
					Text: trimmed,
				})
			}
			if strings.HasPrefix(trimmed, "}") {
				curFn.EndLine = lineNum
				curFn = nil
			}
		}
	}

	return code, nil
}
