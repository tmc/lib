package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
)

// runCaptureCommand is the command-level adapter for the launcher foundation.
// It always emits a self-contained status recording when -o is supplied,
// including no-CUDA and partial target outcomes.
func runCaptureCommand(args []string) error {
	if len(args) == 0 {
		printCaptureHelp()
		return nil
	}
	opts, err := parseCaptureArgs(args)
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			printCaptureHelp()
			return nil
		}
		return err
	}
	if opts.Inject == "" && !opts.NoInject {
		fmt.Fprintln(os.Stderr, "cudaprof: error: no injection library specified (--inject /path/to/agent.so, CUDAPROF_INJECT, or CUDAPROF_AGENT); target will not run")
	}
	result, runErr := runCapture(context.Background(), opts)
	if opts.Output != "" {
		if err := writeCaptureStatus(opts.Output, result, captureArtifactBudget{FileBytes: opts.CaptureFileBytes, TotalBytes: opts.CaptureTotalBytes}); err != nil {
			return fmt.Errorf("write capture: %w", err)
		}
	}
	status := compactCaptureStatus(result)
	if opts.JSON || !isTerminal(os.Stdout) {
		if err := json.NewEncoder(os.Stdout).Encode(status); err != nil {
			return fmt.Errorf("write capture status: %w", err)
		}
	} else {
		if result.CaptureSuccess {
			fmt.Fprint(os.Stdout, captureHeadline(result))
			if opts.Output != "" {
				if fi, err := os.Stat(opts.Output); err == nil {
					fmt.Fprintf(os.Stdout, "wrote %s (%s)\n", opts.Output, formatByteSize(fi.Size()))
				} else {
					fmt.Fprintf(os.Stdout, "wrote %s\n", opts.Output)
				}
			}
		} else {
			fmt.Fprintf(os.Stdout, "capture incomplete: %s (outcome: %s)\n", result.CaptureError, result.Outcome)
		}
	}
	if runErr != nil {
		return runErr
	}
	if result.TargetSignal != "" {
		return &exitCodeError{Code: 128, Err: fmt.Errorf("target terminated by %s", result.TargetSignal)}
	}
	if result.TargetExitCode != 0 {
		return &exitCodeError{Code: result.TargetExitCode, Err: fmt.Errorf("target exited with status %d", result.TargetExitCode)}
	}
	if !result.CaptureSuccess {
		if opts.NoInject && result.TargetExitCode == 0 {
			return nil
		}
		msg := "capture incomplete"
		if result.CaptureError != "" {
			msg = fmt.Sprintf("capture incomplete: %s", result.CaptureError)
		}
		return &exitCodeError{Code: 2, Err: errors.New(msg)}
	}
	return nil
}

// captureHeadline is the one line a caller reads after a capture. It must not
// call a run complete that is not: CaptureSuccess qualifies the target
// generation alone, and a run that finalized cleanly while losing 28.5 million
// records to producer backpressure satisfies it. Printing "capture complete"
// there is what let a 0.2%-retained capture read as a good one, and two such
// captures of the same workload retained wildly different fractions while both
// announcing success. The word follows RunCompleteness, which is the field
// that answers the question the word asks.
func captureHeadline(result captureResult) string {
	if result.RunCompleteness == runComplete {
		head := fmt.Sprintf("capture complete: %d records, 0 dropped, run_completeness: %s\n",
			result.ActivityRecords, result.RunCompleteness)
		// A run that dropped nothing can still produce an artifact with no raw
		// records in it, because the journal did not fit the budget. Every
		// count above reports that capture as healthy, and it is unusable for
		// any question about the device timeline.
		if missing := missingJournals(result.OmittedFiles); len(missing) != 0 {
			head += fmt.Sprintf("  but the journal was NOT attached: %s exceeded the artifact budget, so this capture carries no raw records and cudaprof gaps will refuse it%s\n",
				strings.Join(missing, ", "), budgetRemedy(result.OmittedFiles))
		}
		return head
	}
	var b strings.Builder
	fmt.Fprintf(&b, "capture TRUNCATED: %d records retained, %d dropped, run_completeness: %s\n",
		result.ActivityRecords, result.DroppedRecords, result.RunCompleteness)
	if result.RecordLoss != nil {
		if pct, ok := result.RecordLoss.retainedFraction(); ok {
			fmt.Fprintf(&b, "  retained %.2f%% of the %d records the producer offered\n", pct, result.RecordLoss.Offered)
		}
		if why := result.RecordLoss.dominant(result.Capability); why != "" {
			fmt.Fprintf(&b, "  %s\n", why)
		}
	}
	b.WriteString("  totals from this capture are a floor, not a run total, and two captures of one workload can retain different fractions\n")
	b.WriteString("  these records were never written, so --max-records does not apply: it bounds how many already-written frames the reader decodes, and a journal that exceeds it fails outright rather than dropping silently\n")
	b.WriteString("  analyses needing the whole device timeline (cudaprof gaps) refuse this capture; pass --require-complete to make truncation fail the capture itself\n")
	if missing := missingJournals(result.OmittedFiles); len(missing) != 0 {
		fmt.Fprintf(&b, "  the journal was also NOT attached: %s exceeded the artifact budget%s\n",
			strings.Join(missing, ", "), budgetRemedy(result.OmittedFiles))
	}
	return b.String()
}

// captureStatus is the compact projection published as capture-status.json.
// It keeps everything a consumer needs to judge the capture -- outcome,
// identity, counts and totals, worker dispositions, capability, losses and
// omissions -- and deliberately excludes the per-record Activities and
// APIRecords arrays.
//
// Those arrays are O(journal bytes): embedding them made the written artifact
// grow to roughly 1.66x the journal bytes it already contained, and a serving
// run then failed the total-size bound at the write, after the whole workload
// had been served. The per-record detail is not lost; it is projected into
// gpu.pprof and retained verbatim in the attached raw journals, and
// DetailedRecords says exactly where it is for this capture.
type captureStatus struct {
	captureResult
	// ActivitySamples and APIRecordCount replace the omitted arrays with their
	// lengths, so a reader can still tell how much detail exists.
	ActivitySamples int `json:"activity_samples"`
	APIRecordCount  int `json:"api_record_count"`
	// Devices and ActivityKinds are the bounded aggregates that coverage
	// reporting previously recomputed from the per-record array.
	Devices         []uint32          `json:"devices,omitempty"`
	ActivityKinds   map[string]uint64 `json:"activity_kinds,omitempty"`
	DetailedRecords string            `json:"detailed_records"`
}

func compactCaptureStatus(result captureResult) captureStatus {
	status := captureStatus{ActivitySamples: len(result.Activities), APIRecordCount: len(result.APIRecords)}
	seen := make(map[uint32]bool)
	kinds := make(map[string]uint64)
	for _, activity := range result.Activities {
		if !seen[activity.Device] {
			seen[activity.Device] = true
			status.Devices = append(status.Devices, activity.Device)
		}
		kinds[activity.Subtype]++
	}
	sort.Slice(status.Devices, func(i, j int) bool { return status.Devices[i] < status.Devices[j] })
	if len(kinds) != 0 {
		status.ActivityKinds = kinds
	}
	status.DetailedRecords = detailedRecordsDisposition(result)
	// Both fields carry omitempty, so clearing them removes the arrays from
	// the published JSON rather than writing an empty list.
	result.Activities, result.APIRecords = nil, nil
	status.captureResult = result
	return status
}

// budgetRemedy names the flag that would have kept an omitted artifact, and
// bounds the value it needs. The budget is the one refusal in this tool with a
// remedy the caller can actually reach -- unlike a producer-side drop, nothing
// was lost here, the journal simply did not fit -- so saying nothing wastes the
// one case where advice is possible.
//
// The number is stated as a bound rather than a target on purpose. A journal is
// measured against the budget compressed, but CompressJournal abandons the
// encode the moment it passes the limit, so the compressed size is never
// learned and the size reported in omitted_files is the raw file. Naming that
// figure as the value to set would overshoot by the compression ratio; naming
// it as a ceiling is true.
func budgetRemedy(omitted []captureOmittedFile) string {
	file, total := false, false
	var largest int64
	for _, f := range omitted {
		if !cudacapture.IsJournalName(f.Name) {
			continue
		}
		switch {
		case strings.Contains(f.Reason, "total budget"):
			total = true
		case strings.Contains(f.Reason, "file budget"):
			file = true
		}
		if f.Bytes > largest {
			largest = f.Bytes
		}
	}
	switch {
	case file && total:
		return fmt.Sprintf("; raise both --capture-file-bytes (default %d) and --capture-total-bytes (default %d) on the capture that writes it", cudacapture.MaxCaptureFileBytes, cudacapture.MaxCaptureTotalBytes)
	case total:
		return fmt.Sprintf("; raise --capture-total-bytes (default %d) on the capture that writes it, to at most %d for this journal", cudacapture.MaxCaptureTotalBytes, largest)
	case file:
		return fmt.Sprintf("; raise --capture-file-bytes (default %d) on the capture that writes it -- the journal is compared against the budget compressed, so it needs more than the budget and at most %d, the uncompressed size reported above", cudacapture.MaxCaptureFileBytes, largest)
	}
	return ""
}

// detailedRecordsDisposition states where this capture's per-record detail
// lives, including when a journal holding it was omitted.
func detailedRecordsDisposition(result captureResult) string {
	const projected = "per-record activity and API arrays are not embedded in this status; activity samples are projected into gpu.pprof"
	journals := 0
	for _, f := range result.OmittedFiles {
		if cudacapture.IsJournalName(f.Name) {
			journals++
		}
	}
	if journals == 0 {
		return projected + " and the raw records remain in the attached capture journals, embedded gzip-compressed under a " + cudacapture.JournalGzipSuffix + " member name"
	}
	// The remedy deliberately does not go here. This string is embedded in
	// capture-status.json, which is generated, always written, and governed by
	// the same per-file budget -- padding it with advice makes the status more
	// likely to exceed the budget in exactly the case the advice is about.
	// omitted_files already carries name, size, sha256 and reason for a
	// consumer; the remedy belongs on the surfaces a person reads.
	return fmt.Sprintf("%s; %d capture journal(s) exceeded the artifact budget and were not attached, so their raw records are not in this artifact (see omitted_files for exact size and sha256)", projected, journals)
}

func sortedOmissions(omitted []captureOmittedFile) []captureOmittedFile {
	if len(omitted) == 0 {
		return nil
	}
	out := append([]captureOmittedFile(nil), omitted...)
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// captureAttachment is one raw artifact considered for embedding, identified
// before it is read so an oversized file is never held in memory.
type captureAttachment struct {
	// name is the member name inside the capture; path is the file on disk.
	name, path string
	// size is the embedded size: the encoded length of data when the member is
	// encoded, otherwise the file's own size.
	size int64
	// data holds the member when it is not a byte copy of path, as for a
	// gzip-compressed journal. rawName and rawSize then describe the file on
	// disk, which is what an omission reports.
	data    []byte
	rawName string
	rawSize int64
}

// raw describes the artifact as it exists on disk, which stays the identity
// reported for an omission even when the member would have been encoded.
func (a captureAttachment) raw() (string, int64) {
	if a.data != nil {
		return a.rawName, a.rawSize
	}
	return a.name, a.size
}

// compressJournal encodes the journal at a for embedding. Journals are CDW1
// JSON payloads and compress by better than an order of magnitude, which is
// what keeps a long run's raw records inside the per-file budget instead of
// being dropped from the capture entirely.
func compressJournal(a captureAttachment, limit int64) (captureAttachment, error) {
	f, err := os.Open(a.path)
	if err != nil {
		return a, err
	}
	defer f.Close()
	data, err := cudacapture.CompressJournal(f, limit)
	if err != nil {
		return a, err
	}
	return captureAttachment{
		name:    a.name + cudacapture.JournalGzipSuffix,
		path:    a.path,
		size:    int64(len(data)),
		data:    data,
		rawName: a.name,
		rawSize: a.size,
	}, nil
}

// scratchAttachments lists the raw artifacts of a finished capture in a
// deterministic order.
func scratchAttachments(dir string) []captureAttachment {
	var found []captureAttachment
	add := func(name, path string) {
		info, err := os.Lstat(path)
		if err != nil || !info.Mode().IsRegular() {
			return
		}
		found = append(found, captureAttachment{name: name, path: path, size: info.Size()})
	}
	for _, name := range []string{"capture.journal", "handshake.json", "python-profile.json", "python-result.json"} {
		add(name, filepath.Join(dir, name))
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return found
	}
	for _, entry := range entries {
		if !entry.IsDir() || !strings.HasPrefix(entry.Name(), "process-") {
			continue
		}
		for _, name := range []string{"capture.journal", "handshake.json"} {
			// Capture files use flat names. Keep the process/generation
			// namespace in the basename so archives remain self-contained.
			add("worker-"+entry.Name()+"-"+name, filepath.Join(dir, entry.Name(), name))
		}
	}
	sort.Slice(found, func(i, j int) bool { return found[i].name < found[j].name })
	return found
}

// omitAttachment records a file that was not embedded. The digest is of the
// complete file on disk, so a separately retained copy stays identifiable.
func omitAttachment(a captureAttachment, reason string) captureOmittedFile {
	sum, err := fileSHA256(a.path)
	if err != nil {
		sum = ""
	}
	name, size := a.raw()
	return captureOmittedFile{Name: name, Bytes: size, SHA256: sum, Reason: reason}
}

// fileSHA256 digests a file without holding it in memory.
func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// writeCaptureStatus publishes the capture recording under budget. The budget
// is a policy applied to raw attachments before anything is written, so a
// capture whose workload has already run is never lost to a size rejection at
// the end: an artifact that does not fit is omitted with its exact size and
// digest instead.
func writeCaptureStatus(path string, result captureResult, budget captureArtifactBudget) error {
	// pprof sample values are signed int64. Keep the native uint64 in the
	// status/journal, but refuse to publish a misleading zero when the ordinary
	// projection cannot represent it.
	const maxInt64 = uint64(^uint64(0) >> 1)
	for i, activity := range result.Activities {
		if activity.Valid && activity.Subtype == "memcpy" && activity.Bytes > maxInt64 {
			return fmt.Errorf("activity %d copy bytes %d exceed pprof int64 range", i, activity.Bytes)
		}
	}
	budget = budget.effective()

	// Decide what is embedded before anything is read. Generated content --
	// the compact status and the provenance record -- is the capture itself
	// and is always written; the budget governs the raw attachments, which are
	// dropped one at a time, largest first, until the whole artifact fits.
	var kept []captureAttachment
	var omitted []captureOmittedFile
	if result.ScratchDir != "" {
		for _, a := range scratchAttachments(result.ScratchDir) {
			if cudacapture.IsJournalName(a.name) {
				// A journal is embedded compressed, so the budget is applied
				// to the compressed member rather than to the raw file.
				encoded, err := compressJournal(a, budget.FileBytes)
				if err != nil {
					reason := fmt.Sprintf("compress capture journal: %v", err)
					if errors.Is(err, cudacapture.ErrJournalTooLarge) {
						reason = fmt.Sprintf("compressed journal exceeds capture file budget of %d bytes", budget.FileBytes)
					}
					omitted = append(omitted, omitAttachment(a, reason))
					continue
				}
				kept = append(kept, encoded)
				continue
			}
			if a.size > budget.FileBytes {
				omitted = append(omitted, omitAttachment(a, fmt.Sprintf("exceeds capture file budget of %d bytes", budget.FileBytes)))
				continue
			}
			kept = append(kept, a)
		}
	}
	provData := determineCaptureProvenance(result.Capability)
	var metadata []byte
	for {
		result.OmittedFiles = sortedOmissions(omitted)
		var err error
		metadata, err = json.Marshal(struct {
			Status captureStatus `json:"status"`
		}{Status: compactCaptureStatus(result)})
		if err != nil {
			return err
		}
		total := int64(len(metadata)) + int64(len(provData))
		for _, a := range kept {
			total += a.size
		}
		if total <= budget.TotalBytes || len(kept) == 0 {
			break
		}
		largest := 0
		for i, a := range kept {
			if a.size > kept[largest].size {
				largest = i
			}
		}
		omitted = append(omitted, omitAttachment(kept[largest], fmt.Sprintf("exceeds capture total budget of %d bytes", budget.TotalBytes)))
		kept = append(kept[:largest], kept[largest+1:]...)
	}
	attachments := make(map[string][]byte, len(kept))
	for _, a := range kept {
		if a.data != nil {
			attachments[a.name] = a.data
			continue
		}
		if data, err := os.ReadFile(a.path); err == nil {
			attachments[a.name] = data
		}
	}
	limits := cudacapture.Limits{FileBytes: budget.FileBytes, TotalBytes: budget.TotalBytes}
	p := cudacapture.BuildActivityProfile(result.Session, uint64(result.TargetPID), result.Activities)
	c := &cudacapture.Capture{
		Profile: p,
		Files:   map[string][]byte{"capture-status.json": metadata},
		CodeUnavailable: map[string]string{
			"source": "source was not available in this capture",
			"ptx":    "PTX was not available in this capture",
			"cubin":  "CUBIN was not available in this capture",
			"sass":   "SASS was not available in this capture",
		},
	}
	for name, data := range attachments {
		c.Files[name] = data
	}
	if len(provData) > 0 {
		c.Files["capture-provenance.json"] = provData
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".cudaprof-status-")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	if err := cudacapture.WriteLimits(tmp, c, limits); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	// Link publishes the completed file without replacing an existing output.
	// Unlike rename-on-Unix, this keeps a concurrent capture from clobbering a
	// previously selected recording. The temporary file and destination stay
	// in one directory, so publication is atomic with respect to readers.
	if err := os.Link(tmpName, path); err != nil {
		_ = os.Remove(tmpName)
		return fmt.Errorf("publish capture: %w", err)
	}
	if err := os.Remove(tmpName); err != nil {
		return fmt.Errorf("remove temporary capture: %w", err)
	}
	return nil
}

func formatByteSize(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(b)/float64(div), "KMGTPE"[exp])
}

// printCaptureHelp documents the capture flags. The budget defaults are
// interpolated from the constants that enforce them rather than transcribed,
// so help that disagrees with behaviour cannot be written.
func printCaptureHelp() {
	fmt.Print(captureHelp())
}

// captureHelp is the help text itself, separated from printing it so a test can
// read it without swapping os.Stdout for a pipe.
func captureHelp() string {
	return fmt.Sprintf(`Usage: cudaprof capture [options] -- command [args...]

Options:
  --inject <path>               path to CUPTI agent shared library (or set CUDAPROF_INJECT)
  --no-inject                   run target uninstrumented without injection library
  --json                        write JSON status summary to stdout
  -o <path>                     write capture output to file or directory
  --max-records <N>             maximum journal records the reader decodes (default %d);
                                bounds decoding only -- it cannot recover a record the
                                producer never wrote, and a journal that exceeds it fails
                                the capture rather than being silently truncated
  --capture-file-bytes <N>      maximum bytes for one embedded artifact (default %d);
                                a journal is measured against this compressed, and one that
                                does not fit is omitted, which makes the capture unusable
                                for cudaprof gaps
  --capture-total-bytes <N>     maximum bytes for the whole capture artifact (default %d)
  --trace                       enable CUDA activity tracing (default true)
  --deep                        enable deep tracing (API calls and activities)
  --exhaustive                  enable exhaustive tracing
  --collect <metric>            collect additional CUPTI metrics
  --exclude <kernel>            exclude matching kernel names
  --timeout <duration>          execution timeout
  --drain-timeout <duration>    drain timeout after target completion
  --require-complete            require complete capture without dropped records
  --follow-children             follow spawned child processes
`, controllerMaxJournalRecords, cudacapture.MaxCaptureFileBytes, cudacapture.MaxCaptureTotalBytes)
}

type deviceProvenance struct {
	Device            string `json:"device,omitempty"`
	Arch              string `json:"arch,omitempty"`
	ComputeCapability string `json:"compute_capability,omitempty"`
	SMCount           int    `json:"sm_count,omitempty"`
	MaxWarpsPerSM     int    `json:"max_warps_per_sm,omitempty"`
	MaxBlocksPerSM    int    `json:"max_blocks_per_sm,omitempty"`
}

func queryLiveDeviceProvenanceJSON() []byte {
	defer func() {
		_ = recover()
	}()
	dev, err := cuda.NewDevice(0)
	if err != nil || dev == nil {
		return nil
	}
	defer dev.Close()

	name, _ := dev.Name()
	major, minor, _ := dev.ComputeCapability()
	smCount, _ := dev.MultiprocessorCount()

	arch := ""
	cc := ""
	if major > 0 {
		arch = fmt.Sprintf("sm_%d%d", major, minor)
		cc = fmt.Sprintf("%d.%d", major, minor)
	}

	maxWarps := 0
	maxBlocks := 0
	if arch != "" {
		if occDev, err := cudaoccupancy.LookupDevice(arch); err == nil {
			maxWarps = occDev.MaxWarpsPerSM
			maxBlocks = occDev.MaxBlocksPerSM
		}
	}
	if maxBlocks == 0 {
		if mb, err := dev.Attribute(driver.DevAttrMaxBlocksPerMultiprocessor); err == nil && mb > 0 {
			maxBlocks = mb
		}
	}
	if maxWarps == 0 {
		if mt, err := dev.Attribute(driver.DevAttrMaxThreadsPerMultiprocessor); err == nil && mt > 0 {
			maxWarps = mt / 32
		}
	}

	if arch == "" && smCount <= 0 && name == "" {
		return nil
	}

	prov := deviceProvenance{
		Device:            name,
		Arch:              arch,
		ComputeCapability: cc,
		SMCount:           smCount,
		MaxWarpsPerSM:     maxWarps,
		MaxBlocksPerSM:    maxBlocks,
	}
	data, err := json.MarshalIndent(prov, "", "  ")
	if err != nil {
		return nil
	}
	return data
}

func determineCaptureProvenance(agentCapability []byte) []byte {
	liveProv := queryLiveDeviceProvenanceJSON()
	if len(liveProv) == 0 {
		if len(agentCapability) > 0 {
			return agentCapability
		}
		return nil
	}
	if len(agentCapability) == 0 {
		return liveProv
	}
	var m map[string]any
	if err := json.Unmarshal(liveProv, &m); err != nil {
		return liveProv
	}
	var agentMap map[string]any
	if err := json.Unmarshal(agentCapability, &agentMap); err == nil {
		for k, v := range agentMap {
			if _, exists := m[k]; !exists {
				m[k] = v
			}
		}
		if merged, err := json.MarshalIndent(m, "", "  "); err == nil {
			return merged
		}
	}
	return liveProv
}
