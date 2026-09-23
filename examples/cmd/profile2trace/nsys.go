package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"path/filepath"
)

// sqlite uses the read-only CLI to avoid adding a SQLite driver dependency.
func sqlite(path, query string, dst any) error {
	path, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	cmd := exec.Command("sqlite3", "-readonly", "-json", path, query)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	data, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("sqlite3: %w: %s", err, stderr.String())
	}
	if len(bytes.TrimSpace(data)) == 0 {
		data = []byte("[]")
	}
	if err := json.Unmarshal(data, dst); err != nil {
		return fmt.Errorf("sqlite3 JSON: %w", err)
	}
	return nil
}

func readNSYS(path string, w window, max int, log io.Writer) ([]slice, error) {
	var tables []struct{ Name string }
	if err := sqlite(path, "SELECT name FROM sqlite_master WHERE type='table'", &tables); err != nil {
		return nil, err
	}
	has := make(map[string]bool)
	for _, table := range tables {
		has[table.Name] = true
	}
	// Diagnostics describe the capture, including events the profiler may
	// have lost. Keep them visible even when selecting a smaller window.
	if has["DIAGNOSTIC_EVENT"] {
		query := `SELECT cast(severity AS TEXT) AS Severity, text AS Message FROM DIAGNOSTIC_EVENT ORDER BY timestamp`
		if has["ENUM_DIAGNOSTIC_SEVERITY_LEVEL"] {
			query = `SELECT coalesce(s.label, 'severity ' || d.severity) AS Severity, d.text AS Message
 FROM DIAGNOSTIC_EVENT d LEFT JOIN ENUM_DIAGNOSTIC_SEVERITY_LEVEL s ON s.id=d.severity
 WHERE s.name NOT IN ('Info', 'Verbose') OR s.name IS NULL ORDER BY d.timestamp`
		}
		var diagnostics []struct{ Severity, Message string }
		if err := sqlite(path, query, &diagnostics); err != nil {
			return nil, fmt.Errorf("read Nsight diagnostics: %w", err)
		}
		for _, d := range diagnostics {
			fmt.Fprintf(log, "Nsight diagnostic (%s): %s\n", d.Severity, d.Message)
		}
	}
	if !has["CUPTI_ACTIVITY_KIND_KERNEL"] {
		return nil, errors.New("Nsight export has no CUPTI_ACTIVITY_KIND_KERNEL table; check the capture mode and profiler version")
	}
	if !has["StringIds"] {
		return nil, errors.New("Nsight export has no StringIds table")
	}
	// Use LEFT JOIN so missing name strings cannot silently discard kernels.
	var kernels []struct {
		Start, End                       *int64
		Device, Context, Stream, Process *int64
		Correlation                      *int64
		Name                             string
	}
	var counts []struct{ Total, Selected int }
	query := fmt.Sprintf(`SELECT count(*) AS Total, sum(start >= %d AND start < %d) AS Selected
 FROM CUPTI_ACTIVITY_KIND_KERNEL`, w.lo, w.hi)
	if err := sqlite(path, query, &counts); err != nil {
		return nil, err
	}
	if len(counts) != 1 || counts[0].Total == 0 {
		return nil, errors.New("Nsight kernel table is empty")
	}
	if counts[0].Selected > max {
		return nil, fmt.Errorf("window %s selects %d of %d kernel rows, over the -max-slices limit of %d; narrow -nsys-window or raise -max-slices", w, counts[0].Selected, counts[0].Total, max)
	}
	query = fmt.Sprintf(`SELECT k.start AS Start, k.end AS End, k.deviceId AS Device,
 k.contextId AS Context, k.streamId AS Stream, k.globalPid AS Process,
 k.correlationId AS Correlation, coalesce(s.value, 'kernel name unavailable') AS Name
 FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON s.id=k.demangledName
 WHERE k.start >= %d AND k.start < %d
 ORDER BY k.start, k.end`, w.lo, w.hi)
	if err := sqlite(path, query, &kernels); err != nil {
		return nil, err
	}
	if len(kernels) == 0 {
		return nil, fmt.Errorf("window %s selects none of the %d kernel rows", w, counts[0].Total)
	}
	var rows []slice
	for i, k := range kernels {
		if k.Start == nil || k.End == nil || *k.Start < 0 || *k.End < *k.Start || k.Device == nil || k.Context == nil || k.Stream == nil || k.Process == nil {
			return nil, fmt.Errorf("kernel row %d: invalid interval or lane identity", i)
		}
		attrs := map[string]any{"global_pid": *k.Process, "device_id": *k.Device, "context_id": *k.Context, "stream_id": *k.Stream, "clock_domain": "nsys_ns"}
		if k.Correlation != nil {
			attrs["correlation_id"] = *k.Correlation
		}
		rows = append(rows, slice{k.Name, "cuda_kernel", fmt.Sprintf("CUDA device %d process %d", *k.Device, *k.Process), fmt.Sprintf("context %d stream %d", *k.Context, *k.Stream), *k.Start, *k.End - *k.Start, attrs})
	}
	if w.unbounded() {
		fmt.Fprintf(log, "CUDA device slices: %d (all kernel rows)\n", len(kernels))
	} else {
		fmt.Fprintf(log, "CUDA device slices: %d of %d kernel rows; window %s omits %d\n", len(kernels), counts[0].Total, w, counts[0].Total-len(kernels))
	}
	if !has["NVTX_EVENTS"] {
		fmt.Fprintln(log, "NVTX table absent")
		return rows, nil
	}
	var ranges []struct {
		Start, End, Thread, EndThread *int64
		Name                          string
	}
	query = fmt.Sprintf(`SELECT n.start AS Start, n.end AS End, n.globalTid AS Thread,
 n.endGlobalTid AS EndThread, coalesce(n.text, s.value, 'NVTX name unavailable') AS Name
 FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId
 WHERE n.start >= %d AND n.start < %d ORDER BY n.start, n.end`, w.lo, w.hi)
	if err := sqlite(path, query, &ranges); err != nil {
		return nil, err
	}
	skipped := 0
	for i, n := range ranges {
		if n.End == nil {
			skipped++
			continue
		} // Marks and unfinished ranges are not intervals.
		if n.Start == nil || n.Thread == nil || *n.Start < 0 || *n.End < *n.Start {
			return nil, fmt.Errorf("NVTX row %d: invalid interval or thread", i)
		}
		attrs := map[string]any{"global_tid": *n.Thread, "clock_domain": "nsys_ns"}
		if n.EndThread != nil {
			attrs["end_global_tid"] = *n.EndThread
		}
		rows = append(rows, slice{n.Name, "nvtx", "Nsight host", fmt.Sprintf("global thread %d", *n.Thread), *n.Start, *n.End - *n.Start, attrs})
	}
	fmt.Fprintf(log, "NVTX slices: %d; non-interval records omitted: %d\n", len(ranges)-skipped, skipped)
	return rows, nil
}
