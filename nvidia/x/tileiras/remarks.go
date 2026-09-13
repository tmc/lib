package tileiras

import (
	"bufio"
	"bytes"
	"fmt"
	"strconv"
	"strings"
)

// Remark represents a compiler optimization remark emitted by tileiras.
type Remark struct {
	Kind        string            // "Passed", "Failed", "Missed", "Analysis"
	Pass        string            // e.g. "Memory", "Tensor-core", "Hints"
	Name        string            // e.g. "RemarkMemoryTmaLoadFailed", "RemarkMemoryLoadInstructionSelected"
	File        string            // source file path from DebugLoc
	Line        int               // source line number from DebugLoc
	Column      int               // source column number from DebugLoc
	Function    string            // enclosing function or kernel name
	RemarkId    int               // internal remark identifier
	Remark      string            // descriptive message text
	Instruction string            // instruction type, e.g. "TMA Load instruction", "FMA"
	Shape       string            // shape descriptor, e.g. "[16, 16]"
	Reason      string            // failure or missed reason if reported
	Args        map[string]string // raw key-value arguments
}

// IsTMAFailure reports whether this remark records a failed TMA load or store selection.
func (r Remark) IsTMAFailure() bool {
	if r.Kind != "Failed" && r.Kind != "Failure" {
		return false
	}
	if strings.Contains(r.Name, "Tma") || strings.Contains(r.Name, "TMA") {
		return true
	}
	msg := strings.ToLower(r.Remark)
	return strings.Contains(msg, "tma") && strings.Contains(msg, "failed")
}

// IsTensorCoreMMA reports whether an MMA operation was optimized to use Tensor Cores.
func (r Remark) IsTensorCoreMMA() bool {
	return r.Kind == "Passed" && (strings.Contains(r.Name, "TensorCore") || strings.Contains(r.Remark, "Tensor Cores"))
}

// IsTensorCoreFailure reports whether an MMA operation failed to use Tensor Cores and fell back to FMA.
func (r Remark) IsTensorCoreFailure() bool {
	if r.Kind != "Failed" && r.Kind != "Failure" {
		return false
	}
	return strings.Contains(r.Name, "TensorCore") || strings.Contains(r.Remark, "FMA") || strings.Contains(r.Instruction, "FMA")
}

func (r Remark) String() string {
	loc := ""
	if r.File != "" && r.Line > 0 {
		loc = fmt.Sprintf(" at %s:%d:%d", r.File, r.Line, r.Column)
	}
	desc := r.Remark
	if desc == "" {
		desc = r.Name
	}
	if r.Instruction != "" {
		desc += fmt.Sprintf(" (%s)", r.Instruction)
	}
	return fmt.Sprintf("[%s/%s]%s %s", r.Pass, r.Kind, loc, desc)
}

// ParseRemarksYAML parses YAML-formatted remarks emitted by tileiras --remark-format=yaml.
func ParseRemarksYAML(data []byte) ([]Remark, error) {
	if len(data) == 0 {
		return nil, nil
	}

	var remarks []Remark
	scanner := bufio.NewScanner(bytes.NewReader(data))

	var cur *Remark
	inArgs := false

	flush := func() {
		if cur != nil {
			// Normalize Kind
			if cur.Kind == "Failure" {
				cur.Kind = "Failed"
			}
			remarks = append(remarks, *cur)
			cur = nil
			inArgs = false
		}
	}

	for scanner.Scan() {
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)

		// Document header: "--- !Passed", "--- !Failure", etc.
		if strings.HasPrefix(trimmed, "---") {
			flush()
			kind := ""
			if idx := strings.Index(trimmed, "!"); idx >= 0 {
				kind = strings.TrimSpace(trimmed[idx+1:])
			}
			cur = &Remark{
				Kind: kind,
				Args: make(map[string]string),
			}
			continue
		}

		if trimmed == "..." {
			flush()
			continue
		}

		if cur == nil {
			continue
		}

		// Check for top-level keys vs Args items
		if strings.HasPrefix(line, "Pass:") {
			cur.Pass = strings.TrimSpace(strings.TrimPrefix(line, "Pass:"))
			inArgs = false
		} else if strings.HasPrefix(line, "Name:") {
			cur.Name = strings.TrimSpace(strings.TrimPrefix(line, "Name:"))
			inArgs = false
		} else if strings.HasPrefix(line, "Function:") {
			cur.Function = strings.TrimSpace(strings.TrimPrefix(line, "Function:"))
			inArgs = false
		} else if strings.HasPrefix(line, "DebugLoc:") {
			parseDebugLoc(line, cur)
			inArgs = false
		} else if strings.HasPrefix(line, "Args:") {
			inArgs = true
		} else if inArgs && strings.HasPrefix(trimmed, "-") {
			// Argument item e.g. "- RemarkId: 4" or "- Remark: Load instruction selected"
			argLine := strings.TrimSpace(strings.TrimPrefix(trimmed, "-"))
			if k, v, ok := parseKeyValue(argLine); ok {
				cur.Args[k] = v
				switch k {
				case "RemarkId":
					cur.RemarkId, _ = strconv.Atoi(v)
				case "Remark":
					cur.Remark = v
				case "Instruction":
					cur.Instruction = v
				case "Shape":
					cur.Shape = v
				case "Reason":
					cur.Reason = v
				}
			}
		} else if inArgs && strings.Contains(trimmed, ":") {
			// Continuation argument key: value
			if k, v, ok := parseKeyValue(trimmed); ok {
				cur.Args[k] = v
				switch k {
				case "RemarkId":
					cur.RemarkId, _ = strconv.Atoi(v)
				case "Remark":
					cur.Remark = v
				case "Instruction":
					cur.Instruction = v
				case "Shape":
					cur.Shape = v
				case "Reason":
					cur.Reason = v
				}
			}
		}
	}

	flush()
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("reading remarks yaml: %w", err)
	}

	return remarks, nil
}

func parseDebugLoc(line string, r *Remark) {
	// DebugLoc: { File: '/tmp/tile_matrix.cu', Line: 10, Column: 37 }
	idx := strings.Index(line, "{")
	if idx < 0 {
		return
	}
	content := line[idx+1:]
	if end := strings.Index(content, "}"); end >= 0 {
		content = content[:end]
	}
	parts := strings.Split(content, ",")
	for _, p := range parts {
		k, v, ok := parseKeyValue(strings.TrimSpace(p))
		if !ok {
			continue
		}
		switch k {
		case "File":
			r.File = strings.Trim(v, "'\"")
		case "Line":
			r.Line, _ = strconv.Atoi(v)
		case "Column":
			r.Column, _ = strconv.Atoi(v)
		}
	}
}

func parseKeyValue(s string) (key, value string, ok bool) {
	idx := strings.Index(s, ":")
	if idx < 0 {
		return "", "", false
	}
	key = strings.TrimSpace(s[:idx])
	value = strings.TrimSpace(s[idx+1:])
	value = strings.Trim(value, "'\"")
	return key, value, true
}
