package tileiras

import (
	"fmt"
	"io"
)

// EmitRemarksYAML writes a slice of Remarks as YAML into w, following the format
// parsed by ParseRemarksYAML.
func EmitRemarksYAML(w io.Writer, remarks []Remark) error {
	for _, r := range remarks {
		kind := r.Kind
		if kind == "" {
			kind = "Passed"
		}
		if _, err := fmt.Fprintf(w, "--- !%s\n", kind); err != nil {
			return err
		}
		if r.Pass != "" {
			if _, err := fmt.Fprintf(w, "Pass:            %s\n", r.Pass); err != nil {
				return err
			}
		}
		if r.Name != "" {
			if _, err := fmt.Fprintf(w, "Name:            %s\n", r.Name); err != nil {
				return err
			}
		}
		if r.File != "" || r.Line > 0 || r.Column > 0 {
			if _, err := fmt.Fprintf(w, "DebugLoc:        { File: '%s', Line: %d, Column: %d }\n", r.File, r.Line, r.Column); err != nil {
				return err
			}
		}
		if r.Function != "" {
			if _, err := fmt.Fprintf(w, "Function:        %s\n", r.Function); err != nil {
				return err
			}
		}
		hasArgs := len(r.Args) > 0 || r.Remark != "" || r.Instruction != "" || r.Shape != "" || r.Reason != "" || r.RemarkId > 0
		if hasArgs {
			if _, err := fmt.Fprintf(w, "Args:\n"); err != nil {
				return err
			}
			if r.RemarkId > 0 {
				if _, err := fmt.Fprintf(w, "  - RemarkId:    %d\n", r.RemarkId); err != nil {
					return err
				}
			}
			if r.Remark != "" {
				if _, err := fmt.Fprintf(w, "  - Remark:      %s\n", r.Remark); err != nil {
					return err
				}
			}
			if r.Instruction != "" {
				if _, err := fmt.Fprintf(w, "  - Instruction: %s\n", r.Instruction); err != nil {
					return err
				}
			}
			if r.Shape != "" {
				if _, err := fmt.Fprintf(w, "  - Shape:       '%s'\n", r.Shape); err != nil {
					return err
				}
			}
			if r.Reason != "" {
				if _, err := fmt.Fprintf(w, "  - Reason:      %s\n", r.Reason); err != nil {
					return err
				}
			}
			for k, v := range r.Args {
				if k == "RemarkId" || k == "Remark" || k == "Instruction" || k == "Shape" || k == "Reason" {
					continue
				}
				if _, err := fmt.Fprintf(w, "  - %s:    %s\n", k, v); err != nil {
					return err
				}
			}
		}
		if _, err := fmt.Fprintf(w, "...\n"); err != nil {
			return err
		}
	}
	return nil
}
