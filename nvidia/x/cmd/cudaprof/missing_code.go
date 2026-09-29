package main

import (
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func budgetOmittedCubins(cap *cudacapture.Capture) []captureOmittedFile {
	if cap == nil {
		return nil
	}
	var status struct {
		Status struct {
			OmittedFiles []captureOmittedFile `json:"omitted_files"`
		} `json:"status"`
	}
	if json.Unmarshal(cap.Files["capture-status.json"], &status) != nil {
		return nil
	}
	var omitted []captureOmittedFile
	for _, file := range status.Status.OmittedFiles {
		if strings.EqualFold(filepath.Ext(file.Name), ".cubin") && strings.Contains(file.Reason, "budget") {
			omitted = append(omitted, file)
		}
	}
	return omitted
}

func reportMissingModuleCode(w io.Writer, cap *cudacapture.Capture) {
	files := budgetOmittedCubins(cap)
	if len(files) == 0 {
		return
	}
	items := make([]string, 0, len(files))
	for _, file := range files {
		items = append(items, fmt.Sprintf("%s (%s)", filepath.Base(file.Name), file.Reason))
	}
	fmt.Fprintf(w, "cudaprof: code not captured (module over budget): %s; pressure and code-based advice cannot inspect those modules\n", strings.Join(items, ", "))
}
