package tileir

import (
	"path/filepath"
	"strings"

	"github.com/tmc/lib/nvidia/x/tileiras"
)

// AttachRemarks correlates compiler optimization remarks emitted by tileiras
// to Operation and Function AST nodes by matching DebugLoc line/file coordinates
// and function symbol names.
func (m *Module) AttachRemarks(remarks []tileiras.Remark) {
	if m == nil || len(remarks) == 0 {
		return
	}

	for _, r := range remarks {
		// Target candidates within matching function, or across all functions
		var targetFns []*Function
		if r.Function != "" {
			cleanName := strings.TrimPrefix(r.Function, "@")
			if fn := m.FindFunction(cleanName); fn != nil {
				targetFns = []*Function{fn}
			}
		}
		if len(targetFns) == 0 {
			targetFns = m.Functions
		}

		matchedOp := false
		for _, fn := range targetFns {
			for _, op := range fn.Operations {
				if matchLoc(op.Loc, op.Line, r) {
					op.Remarks = append(op.Remarks, r)
					matchedOp = true
				}
			}
			if matchedOp || matchFnLoc(fn.Loc, fn.StartLine, r) {
				fn.Remarks = append(fn.Remarks, r)
			}
		}
	}
}

func matchLoc(loc Location, textLine int, r tileiras.Remark) bool {
	if r.Line <= 0 {
		return false
	}

	// First try matching source debug location
	if loc.Line > 0 && loc.Line == r.Line {
		if r.File == "" || loc.File == "" {
			return true
		}
		if loc.File == r.File || filepath.Base(loc.File) == filepath.Base(r.File) {
			return true
		}
	}

	// Fallback to text line number in tileir if remark references the IR file
	if textLine > 0 && textLine == r.Line {
		if r.File == "" || strings.HasSuffix(r.File, ".tileir") || strings.HasSuffix(r.File, ".tilebc") {
			return true
		}
	}

	return false
}

func matchFnLoc(loc Location, startLine int, r tileiras.Remark) bool {
	if r.Line <= 0 {
		return false
	}
	if loc.Line > 0 && loc.Line == r.Line {
		if r.File == "" || loc.File == "" || loc.File == r.File || filepath.Base(loc.File) == filepath.Base(r.File) {
			return true
		}
	}
	if startLine > 0 && startLine == r.Line {
		return true
	}
	return false
}
