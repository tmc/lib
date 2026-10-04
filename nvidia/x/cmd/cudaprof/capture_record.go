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
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudaprof/isolation"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
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
	// An output that holds no activity records does not block its own retry.
	// The removal happens before the target runs, not at publication time:
	// writeCaptureStatus publishes with os.Link precisely so a concurrent
	// capture cannot clobber a selected recording, and that link fails on an
	// existing destination -- so a run allowed to start and refused at publish
	// would cost a whole workload under GPU lock to learn what a stat call
	// says now.
	if opts.ReplaceEmptyOutput {
		fmt.Fprintf(os.Stderr, "cudaprof: replacing %s, which holds no activity records\n", opts.Output)
		if err := os.Remove(opts.Output); err != nil {
			return fmt.Errorf("replace empty capture output: %w", err)
		}
	}
	if opts.Inject == "" && !opts.NoInject {
		fmt.Fprintln(os.Stderr, "cudaprof: error: no injection library specified (--inject /path/to/agent.so, CUDAPROF_INJECT, or CUDAPROF_AGENT); target will not run")
	}
	result, runErr := runCaptureWithPlan(context.Background(), opts, runCapture)
	attachCaptureIsolation(&result, opts.IsolationReceipt)
	if opts.LaunchManifest {
		result.LaunchManifest = buildLaunchManifest(result)
	}
	budget := captureArtifactBudget{FileBytes: opts.CaptureFileBytes, TotalBytes: opts.CaptureTotalBytes}
	maxJournalRecords := opts.MaxJournalRecords
	if result.planBudget != nil {
		budget = budget.effective()
		budget.FileBytes = min(budget.FileBytes, int64(result.planBudget.MaxBytes))
		budget.TotalBytes = min(budget.TotalBytes, int64(result.planBudget.MaxBytes))
		if maxJournalRecords <= 0 || uint64(maxJournalRecords) > result.planBudget.MaxRecords {
			maxJournalRecords = int64(result.planBudget.MaxRecords)
		}
	}
	result.recordLimits(maxJournalRecords, budget)
	if opts.Output != "" {
		omitted, err := writeCapture(opts.Output, result, budget)
		if err != nil {
			if result.planBudget != nil {
				result.CapturePlanError = err.Error()
				result.CaptureSuccess = false
				result.CaptureError = "capture plan output failed: " + err.Error()
				if statusErr := json.NewEncoder(os.Stdout).Encode(compactCaptureStatus(result)); statusErr != nil {
					return fmt.Errorf("write capture diagnostic: %w", statusErr)
				}
			}
			return fmt.Errorf("write capture: %w", err)
		}
		result.OmittedFiles = omitted
	}
	status := compactCaptureStatus(result)
	if opts.JSON || !isTerminal(os.Stdout) {
		if err := json.NewEncoder(os.Stdout).Encode(status); err != nil {
			return fmt.Errorf("write capture status: %w", err)
		}
	} else {
		printIsolation(os.Stdout, *result.Isolation)
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
	if opts.RequireIsolation {
		if err := isolation.Require(*result.Isolation, true); err != nil {
			return &exitCodeError{Code: 2, Err: err}
		}
	}
	if runErr != nil {
		return runErr
	}
	partial := partialCapture(result)
	if partial != nil && opts.AllowPartial {
		fmt.Fprintf(os.Stderr, "cudaprof: %v\n", partial)
		partial = nil
	}
	if result.TargetSignal != "" {
		return &exitCodeError{Code: 128, Err: errors.New(targetFailureMessage(result))}
	}
	if result.TargetExitCode != 0 {
		return &exitCodeError{Code: result.TargetExitCode, Err: errors.New(targetFailureMessage(result))}
	}
	if partial != nil {
		return &exitCodeError{Code: 2, Err: partial}
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
	if opts.Iterations != nil {
		cap, err := readCapturePathLimits(opts.Output, cudacapture.Limits{FileBytes: opts.CaptureFileBytes, TotalBytes: opts.CaptureTotalBytes})
		if err != nil {
			return fmt.Errorf("iteration gate: read capture: %w", err)
		}
		return captureIterationGate(os.Stderr, cap, *opts.Iterations)
	}
	return nil
}

// captureIterationGate runs the iterations gate on a written capture and
// writes a one-line summary of it to w. The capture fails with status 2
// when the gate does.
func captureIterationGate(w io.Writer, cap *cudacapture.Capture, opts iterationsOptions) error {
	res, err := measureIterations(cap, regexp.MustCompile(opts.marker), nil, opts, nil)
	if err != nil {
		return &exitCodeError{Code: 2, Err: fmt.Errorf("iteration gate: %w", err)}
	}
	r := res.Report
	if !r.Gate.Pass {
		fmt.Fprintf(w, "cudaprof: iteration gate FAIL: %s\n", strings.Join(r.Gate.Failures, "; "))
		return &exitCodeError{Code: 2, Err: errors.New("iteration gate failed")}
	}
	s := r.Steady
	n := float64(s.Iterations) * 1e6
	fmt.Fprintf(w, "cudaprof: iteration gate pass: %d markers, %d steady iterations; per iteration window %.3f ms, busy %.3f ms, idle %.3f ms, %.0f launches\n",
		r.Markers, s.Iterations, float64(s.WindowNanos)/n, float64(s.BusyNanos)/n, float64(s.IdleNanos)/n, float64(s.Launches)/float64(s.Iterations))
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
	return captureHeadlineTotals(result) + wrapperNote(result) + projectionNotes(result)
}

// wrapperNote says whose records the totals are when they are not the
// target's. A shell or launcher runs no CUDA of its own, so every record in
// the capture came from a process it started, and a reader who assumes
// otherwise will look for the target's own name in a timeline that cannot
// hold it.
func wrapperNote(result captureResult) string {
	for _, w := range result.WorkerGenerations {
		if w.Process != uint64(result.TargetPID) || w.Disposition != captureWorkerNoCUDA {
			continue
		}
		return fmt.Sprintf("  the target process ran no CUDA; these totals are from the %d generation(s) it started\n", len(result.WorkerGenerations)-1)
	}
	return ""
}

// projectionNotes says how the projection differs from the journal totals the
// headline just printed. Both differences are the caller's own policy, and
// both are silent in every count above them: an excluded kernel and an
// unfollowed child generation each leave a capture whose pprof holds fewer
// records than its own activity_records, with nothing on the terminal to say
// why.
func projectionNotes(result captureResult) string {
	var b strings.Builder
	if result.ExcludedRecords > 0 {
		fmt.Fprintf(&b, "  excluded %d kernel record(s) (%s) matching %s from the projection; the totals above and the attached journals still hold them, so cudaprof gaps still sees them\n",
			result.ExcludedRecords, time.Duration(result.ExcludedNanos), strings.Join(result.ExcludedPatterns, ", "))
	}
	if n := uncheckedModules(result.Modules); n > 0 {
		fmt.Fprintf(&b, "  %d module load(s) arrived after the agent's table of seen modules was full and were kept without a duplicate check, so the module census may count one image more than once\n", n)
	}
	if !result.ChildrenFollowed {
		if generations, records := unfollowedChildren(result.WorkerGenerations); generations > 0 {
			fmt.Fprintf(&b, "  %d child generation(s) holding %d activity record(s) are not in these totals (--follow-children=false); their journals are attached and their dispositions are in worker_generations\n",
				generations, records)
		}
	}
	return b.String()
}

// uncheckedModules counts the module records the agent wrote without a
// duplicate check.
func uncheckedModules(modules []captureModule) int {
	n := 0
	for _, m := range modules {
		if m.Unchecked {
			n++
		}
	}
	return n
}

// unfollowedChildren counts the generations left out of the totals and the
// records they hold. The counts come from the generations themselves, which
// were decoded either way: not following a child never means not observing it.
func unfollowedChildren(workers []captureWorker) (generations, records uint64) {
	for _, w := range workers {
		if w.ActivityRecords == 0 {
			continue
		}
		generations++
		records += w.ActivityRecords
	}
	return generations, records
}

func captureHeadlineTotals(result captureResult) string {
	if result.RunCompleteness == runComplete {
		head := fmt.Sprintf("capture complete: %d records, 0 dropped, run_completeness: %s\n",
			result.ActivityRecords, result.RunCompleteness)
		// Complete answers "was anything dropped", and at zero records it
		// answers it correctly: nothing was dropped, because nothing was
		// produced. The word is right and the reading it invites is not, so
		// the count has to be spoken rather than left for the reader to
		// interpret -- a capture with no records reads exactly like a
		// workload that ran no such kernels, and the two are opposite
		// findings. This cost a published census that covered 12.6% of a
		// workload while reading as complete.
		if result.ActivityRecords == 0 {
			head += "  but this capture holds NO activity records: complete means nothing was dropped, and nothing was dropped because nothing was produced. No ranking, census or duty cycle taken from this file covers any part of the run\n"
		}
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
// omissions, and the cudaprof build that wrote it -- and deliberately
// excludes the per-record Activities and APIRecords arrays.
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
	// Cudaprof is the build that wrote the status.
	Cudaprof buildStamp `json:"cudaprof"`
}

func compactCaptureStatus(result captureResult) captureStatus {
	status := captureStatus{ActivitySamples: len(result.Activities), APIRecordCount: len(result.APIRecords), Cudaprof: currentBuild()}
	seen := make(map[uint32]bool)
	kinds := make(map[string]uint64)
	for _, activity := range result.Activities {
		// A marker is a host annotation whose Device field names no device it
		// ran on. An external-correlation record is a link carrying no
		// device at all. A CUPTI overhead names an object that is usually a
		// host thread, and a synchronization record carries no device field
		// either: on both, Device is zero because nothing filled it. Listing
		// any of them would report device 0 as a device the capture saw work
		// on, on every capture, including one that used no device 0. A graph
		// trace is not in this set -- it carries a real device id. Their
		// subtypes are still counted below.
		if activity.Subtype == "unified_memory" || activity.Subtype == "marker" || activity.Subtype == "external_correlation" || activity.Subtype == "overhead" || activity.Subtype == "sync" {
			kinds[activity.Subtype]++
			continue
		}
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
		return fmt.Sprintf("; raise both --capture-file-bytes (default %d) and --capture-total-bytes (default %d) on the capture that writes it", captureFileBytes, captureTotalBytes)
	case total:
		return fmt.Sprintf("; raise --capture-total-bytes (default %d) on the capture that writes it, to at most %d for this journal", captureTotalBytes, largest)
	case file:
		return fmt.Sprintf("; raise --capture-file-bytes (default %d) on the capture that writes it -- the journal is compared against the budget compressed, so it needs more than the budget and at most %d, the uncompressed size reported above", captureFileBytes, largest)
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
	if entries, err := os.ReadDir(filepath.Join(dir, "stacks")); err == nil {
		for _, entry := range entries {
			name := entry.Name()
			if strings.HasSuffix(name, ".jsonl") || strings.HasSuffix(name, ".json") || strings.HasSuffix(name, ".trace") || strings.HasSuffix(name, ".pprof") {
				add("stacks-"+name, filepath.Join(dir, "stacks", name))
			}
		}
	}
	// Units the target published for itself. They are not per generation:
	// a unit is what the program's code generator produced, and every
	// process of the capture that loaded it loaded the same one.
	if units, err := os.ReadDir(filepath.Join(dir, "units")); err == nil {
		for _, unit := range units {
			if name := unit.Name(); strings.HasSuffix(name, ".json") {
				add("unit-"+name, filepath.Join(dir, "units", name))
			}
		}
	}
	// Executed-instruction counts the target measured for itself, which
	// CUPTI only lets the process owning the context collect.
	if counts, err := os.ReadDir(filepath.Join(dir, "counts")); err == nil {
		for _, count := range counts {
			if name := count.Name(); strings.HasSuffix(name, ".json") {
				add("count-"+name, filepath.Join(dir, "counts", name))
			}
		}
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
		// Module images the agent wrote out of band. They are named under the
		// same worker namespace, which is what keeps two processes that each
		// loaded a first module from colliding on module-0000.
		images, err := os.ReadDir(filepath.Join(dir, entry.Name(), "modules"))
		if err != nil {
			continue
		}
		for _, image := range images {
			name := image.Name()
			if !strings.HasSuffix(name, ".cubin") {
				continue
			}
			add("worker-"+entry.Name()+"-"+name, filepath.Join(dir, entry.Name(), "modules", name))
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

// externalKindCounts counts external-correlation records by namespace. It is a
// fixed array rather than a map so it stays comparable and sums element-wise
// across journals.
type externalKindCounts [cupti.ExternalCorrelationKindSize]uint64

func (c *externalKindCounts) observe(kind cupti.ExternalCorrelationKind) {
	if kind > cupti.ExternalCorrelationKindInvalid && kind < cupti.ExternalCorrelationKindSize {
		c[kind]++
	}
}

func (c externalKindCounts) add(o externalKindCounts) externalKindCounts {
	for i := range c {
		c[i] += o[i]
	}
	return c
}

// chosen is the namespace that gets joined: the lowest-numbered one present,
// or Invalid when the target pushed nothing.
//
// The minimum is not an arbitrary tiebreak. MatchExternal joins one namespace
// per call and the profile holds one cupti.external.kind per sample, so one
// has to be picked; picking the lowest makes the choice a property of which
// namespaces the target used, not of which record the producer happened to
// write first. Two captures of the same workload must not attribute to
// different namespaces because a buffer flushed in a different order.
func (c externalKindCounts) chosen() cupti.ExternalCorrelationKind {
	for i, n := range c {
		if n > 0 {
			return cupti.ExternalCorrelationKind(i)
		}
	}
	return cupti.ExternalCorrelationKindInvalid
}

// note states what was left unjoined when a target pushed into more than one
// namespace, and is empty otherwise.
//
// It exists because the alternative is the fail-open shape: a reader who sees
// "66% of device time carries a framework scope" and no note concludes the
// remaining third had no scope, when part of it had a scope this tool chose
// not to join. The note separates "nothing was pushed there" from "we did not
// join it". It is conditional for the same reason: a caveat printed on every
// capture is read as boilerplate, and then it is not a warning.
func (c externalKindCounts) note(override ...cupti.ExternalCorrelationKind) string {
	var present []cupti.ExternalCorrelationKind
	for i, n := range c {
		if n > 0 {
			present = append(present, cupti.ExternalCorrelationKind(i))
		}
	}
	if len(present) <= 1 {
		return ""
	}
	names := make([]string, len(present))
	for i, k := range present {
		names[i] = externalKindName(k)
	}
	chosen := c.chosen()
	if len(override) != 0 {
		chosen = override[0]
	}
	var unjoined uint64
	var dropped []string
	for _, k := range present {
		if k != chosen {
			unjoined += c[k]
			dropped = append(dropped, externalKindName(k))
		}
	}
	return fmt.Sprintf("%d external namespaces present (%s); only %s was joined, %d record(s) in %s left unattributed",
		len(present), strings.Join(names, ", "), externalKindName(chosen), unjoined, strings.Join(dropped, ", "))
}

// externalKindName is the namespace without the enum's type prefix: Custom0
// rather than ExternalCorrelationKindCustom0, which is what the framework
// documentation calls it.
func externalKindName(k cupti.ExternalCorrelationKind) string {
	return strings.TrimPrefix(k.String(), "ExternalCorrelationKind")
}

// labelExternalCorrelation labels the samples of p with the native
// external-correlation link of the record each was projected from, so a kernel
// carries the ID of the framework scope -- a PyTorch operator, an NCCL
// collective -- that was on the stack when it was launched. The join itself is
// cuptiactivity.MatchExternal, reached through cudaprof.LabelExternal; this
// function only supplies it the two things the capture path has and it does
// not: the records, and which record each sample came from.
//
// A capture whose target pushed no external IDs is returned unlabelled rather
// than labelled not_observed everywhere: nothing was joined, and paying five
// labels per sample to say so on every capture ever taken is not worth it.
//
// One namespace is labelled, because the profile holds one
// cupti.external.kind per sample. A target that pushes into more than one --
// PyTorch uses two -- gets the lowest-numbered one unless a retained stack
// sidecar contains another observed namespace, in which case that namespace
// is preferred. The namespaces left unjoined are stated in profile comments.
//
// scoped identity: the capture session is both session and run, because one
// capture is one run and the artifact retains no separate run ID. Process is
// the process generation's PID. Two generations of one PID inside one capture
// would share a scope and could collide on a correlation ID; MatchExternal
// reports such a collision as ambiguous rather than picking one, so the
// weakness costs attribution, never a wrong attribution.
func labelExternalCorrelation(p *profile.Profile, session string, targetPID uint64, activities []captureActivity, sampleRecords []int, stackKinds map[uint32]bool) (*profile.Profile, error) {
	var counts externalKindCounts
	for _, a := range activities {
		if a.Subtype == "external_correlation" {
			counts.observe(cupti.ExternalCorrelationKind(a.ExternalKind))
		}
	}
	kind := counts.chosen()
	stackKind := cupti.ExternalCorrelationKindInvalid
	for candidate := range counts {
		if counts[candidate] != 0 && stackKinds[uint32(candidate)] && (stackKind == cupti.ExternalCorrelationKindInvalid || candidate < int(stackKind)) {
			stackKind = cupti.ExternalCorrelationKind(candidate)
		}
	}
	if stackKind != cupti.ExternalCorrelationKindInvalid {
		kind = stackKind
	}
	if kind == cupti.ExternalCorrelationKindInvalid {
		return p, nil
	}
	scoped := func(a captureActivity, r cuptiactivity.Record) cuptiactivity.ScopedRecord {
		pid := a.Process
		if pid == 0 {
			pid = targetPID
		}
		r.CorrelationID = a.Correlation
		r.DeviceID = a.Device
		return cuptiactivity.ScopedRecord{Session: session, Run: session, Process: strconv.FormatUint(pid, 10), Record: r}
	}
	records := make([]cuptiactivity.ScopedRecord, 0, len(sampleRecords))
	indices := make([]int, len(sampleRecords))
	for i, index := range sampleRecords {
		a := activities[index]
		var r cuptiactivity.Record
		switch a.Subtype {
		case "kernel":
			r.Kind = cupti.ActivityKindConcurrentKernel
		case "memcpy":
			r.Kind = cupti.ActivityKindMemcpy
			r.RuntimeCorrelationID = a.RuntimeCorrelation
		case "memset":
			r.Kind = cupti.ActivityKindMemset
		default:
			// Every subtype that becomes a sample must have a native kind, or
			// the join sees a sample it cannot account for. Refusing here
			// names the subtype; letting it through makes LabelExternal report
			// a sample index instead.
			return nil, fmt.Errorf("activity subtype %q has a sample but no native activity kind", a.Subtype)
		}
		r.Start, r.End, r.Bytes = a.Start, a.End, a.Bytes
		indices[i] = len(records)
		records = append(records, scoped(a, r))
	}
	for _, a := range activities {
		if a.Subtype != "external_correlation" {
			continue
		}
		records = append(records, scoped(a, cuptiactivity.Record{
			Kind:         cupti.ActivityKindExternalCorrelation,
			ExternalKind: cupti.ExternalCorrelationKind(a.ExternalKind),
			ExternalID:   a.ExternalID,
		}))
	}
	out, err := cudaprof.LabelExternal(p, records, indices, kind)
	if err != nil {
		return nil, err
	}
	// A namespace this capture held and did not join is stated in the
	// profile, where the labels it is missing from are. pprof shows comments
	// alongside the profile, so the reader who asks why a kernel has no
	// cupti.external.id has the answer in the same artifact.
	if note := counts.note(kind); note != "" {
		out.Comments = append(out.Comments, "cupti: "+note)
	}
	return out, nil
}

// writeCapture publishes the capture recording under budget and returns the
// attachments it omitted. The budget is a policy applied to raw attachments
// before anything is written, so a capture whose workload has already run is
// never lost to a size rejection at the end: an artifact that does not fit is
// omitted with its exact size and digest instead.
func writeCapture(path string, result captureResult, budget captureArtifactBudget) ([]captureOmittedFile, error) {
	// pprof sample values are signed int64. Keep the native uint64 in the
	// status/journal, but refuse to publish a misleading zero when the ordinary
	// projection cannot represent it.
	const maxInt64 = uint64(^uint64(0) >> 1)
	for i, activity := range result.Activities {
		if activity.Valid && activity.Subtype == "memcpy" && activity.Bytes > maxInt64 {
			return nil, fmt.Errorf("activity %d copy bytes %d exceed pprof int64 range", i, activity.Bytes)
		}
	}
	budget = budget.effective()
	var maxRecords int64
	if result.Limits != nil {
		maxRecords = result.Limits.MaxRecords
	}
	result.recordLimits(maxRecords, budget)

	// Decide what is embedded before anything is read. Generated content --
	// the compact status and the provenance record -- is the capture itself
	// and is always written; the budget governs the raw attachments, which are
	// dropped one at a time, largest first, until the whole artifact fits.
	var kept []captureAttachment
	var omitted []captureOmittedFile
	var stackSidecar stackSidecarRead
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
		var sidecarErr error
		stackPath := filepath.Join(result.ScratchDir, cudacapture.ExternalStacksName)
		stackSidecar, sidecarErr = readExternalStackSidecar(stackPath, uint64(result.TargetPID), result.Session)
		if stackSidecar.exists {
			attachment := captureAttachment{name: cudacapture.ExternalStacksName, path: stackPath, size: int64(len(stackSidecar.data)), data: stackSidecar.data, rawName: cudacapture.ExternalStacksName, rawSize: stackSidecar.rawSize}
			switch {
			case sidecarErr != nil:
				omitted = append(omitted, omitAttachment(attachment, "external stack sidecar damaged: "+sidecarErr.Error()))
			case int64(len(stackSidecar.data)) > budget.FileBytes:
				omitted = append(omitted, omitAttachment(attachment, fmt.Sprintf("external stack sidecar exceeds capture file budget of %d bytes", budget.FileBytes)))
			default:
				kept = append(kept, attachment)
			}
		}
	}
	// Retained producer capability is the only provenance available here.
	// Artifact publication must not initialize a CUDA context or substitute
	// properties of this host for the device that produced these records.
	provData := result.Capability
	var metadata []byte
	for {
		result.OmittedFiles = sortedOmissions(omitted)
		var err error
		metadata, err = json.Marshal(struct {
			Status captureStatus `json:"status"`
		}{Status: compactCaptureStatus(result)})
		if err != nil {
			return nil, err
		}
		total := int64(len(metadata)) + int64(len(provData)) + int64(len(result.isolationRaw))
		for _, a := range kept {
			total += a.size
		}
		if total <= budget.TotalBytes || len(kept) == 0 {
			break
		}
		// A journal holds the records every analysis reads, so the module
		// images and other attachments go first.
		largest := -1
		for i, a := range kept {
			if !cudacapture.IsJournalName(a.name) && (largest < 0 || a.size > kept[largest].size) {
				largest = i
			}
		}
		if largest < 0 {
			largest = 0
			for i, a := range kept {
				if a.size > kept[largest].size {
					largest = i
				}
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
	// The budget governs attachments. The status and provenance are written
	// whatever their size, as the selection above already assumes, so the
	// writer's limits are raised to admit them; otherwise a budget smaller
	// than the status refuses the capture that selection decided to write.
	limits := cudacapture.Limits{
		FileBytes:  max(budget.FileBytes, int64(len(metadata)), int64(len(provData))),
		TotalBytes: max(budget.TotalBytes, int64(len(metadata))+int64(len(provData))),
	}
	p, sampleRecords := cudacapture.BuildActivityProfileIndex(result.Session, uint64(result.TargetPID), result.Activities, result.APIRecords)
	var err error
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
	if len(result.isolationRaw) != 0 {
		c.Files[isolation.Attachment] = result.isolationRaw
	}
	stackData := attachments[cudacapture.ExternalStacksName]
	if len(stackData) != 0 {
		c.Profile, err = labelExternalCorrelation(c.Profile, result.Session, uint64(result.TargetPID), result.Activities, sampleRecords, externalStackKinds(stackData))
		if err != nil {
			return nil, err
		}
		c.Profile, err = labelExternalStacks(c.Profile, result.Activities, sampleRecords, stackData, uint64(result.TargetPID))
		if err != nil {
			return nil, err
		}
		if stackSidecar.damaged {
			detail := stackSidecar.damageNote
			if len(stackSidecar.damageFiles) != 0 {
				detail += "; files: " + strings.Join(stackSidecar.damageFiles, ", ")
			}
			c.Profile.Comments = append(c.Profile.Comments, "cudaprof: external stack sidecar damaged; "+detail)
		}
	} else {
		c.Profile, err = labelExternalCorrelation(c.Profile, result.Session, uint64(result.TargetPID), result.Activities, sampleRecords, nil)
		if err != nil {
			return nil, err
		}
		c.Profile, err = labelExternalStacks(c.Profile, result.Activities, sampleRecords, nil, uint64(result.TargetPID))
		if err != nil {
			return nil, err
		}
		if stackSidecar.exists {
			why := "sidecar omitted by capture file budget or unreadable"
			for _, item := range omitted {
				if item.Name == cudacapture.ExternalStacksName {
					why = item.Reason
					break
				}
			}
			c.Profile.Comments = append(c.Profile.Comments, "cudaprof: external stack sidecar omitted: "+why)
		}
	}
	// Modules are declared from what was actually embedded, not from what the
	// agent reported writing. The journal census in capture-status.json says
	// what the target loaded; this says what the reader can disassemble, and
	// the two differ exactly by the budget.
	// The driver's number for each stored image, so a sample labelled with
	// it can be matched to the module it came from. The census names the
	// file the agent wrote; the attachment carries that name under its
	// worker namespace, so the census name is a suffix of it.
	driverID := func(attachment string) uint32 {
		for _, m := range result.Modules {
			if m.File != "" && (attachment == m.File || strings.HasSuffix(attachment, "-"+m.File)) {
				return m.ID
			}
		}
		return 0
	}
	for _, a := range kept {
		if !strings.HasSuffix(a.name, ".cubin") {
			continue
		}
		c.Modules = append(c.Modules, cudacapture.Module{
			ID:       strings.TrimSuffix(a.name, ".cubin"),
			DriverID: driverID(a.name),
			Code:     cudacapture.CodeFiles{CUBIN: a.name},
		})
	}
	attachUnits(c, result.Activities, attachments)
	// PC samples are projected after the modules are declared: a sample names
	// its module by the driver's number, and this is where that number is
	// attached to a stored image.
	projectPCSamples(c, result.PCSamples, attachments)
	projectExecuted(c, attachments)
	// The top-level code roles describe the capture as a whole and stay
	// unavailable: there is no single cubin for a process that loaded several.
	// Saying so, rather than dropping the role, is what keeps a reader from
	// reading the absence as "no device code was captured".
	if len(c.Modules) > 0 {
		c.CodeUnavailable["cubin"] = fmt.Sprintf("no whole-capture CUBIN; %d module images are attached individually", len(c.Modules))
	}
	for name, data := range attachments {
		c.Files[name] = data
	}
	if len(provData) > 0 {
		c.Files["capture-provenance.json"] = provData
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return nil, err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".cudaprof-status-")
	if err != nil {
		return nil, err
	}
	tmpName := tmp.Name()
	var output io.Writer = tmp
	if result.planBudget != nil {
		defer os.Remove(tmpName)
		output = &capturePlanOutputWriter{writer: tmp, remaining: int64(result.planBudget.MaxBytes)}
	}
	if err := cudacapture.WriteLimits(output, c, limits); err != nil {
		_ = tmp.Close()
		return nil, err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return nil, err
	}
	if err := tmp.Close(); err != nil {
		return nil, err
	}
	// Link publishes the completed file without replacing an existing output.
	// Unlike rename-on-Unix, this keeps a concurrent capture from clobbering a
	// previously selected recording. The temporary file and destination stay
	// in one directory, so publication is atomic with respect to readers.
	if err := os.Link(tmpName, path); err != nil {
		_ = os.Remove(tmpName)
		return nil, fmt.Errorf("publish capture: %w", err)
	}
	if err := os.Remove(tmpName); err != nil {
		return nil, fmt.Errorf("remove temporary capture: %w", err)
	}
	return result.OmittedFiles, nil
}

// writeCaptureStatus is writeCapture, discarding the omissions.
func writeCaptureStatus(path string, result captureResult, budget captureArtifactBudget) error {
	_, err := writeCapture(path, result, budget)
	return err
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

// printCaptureHelp documents the capture flags. The budget defaults, the
// collection-policy vocabulary and the partial outcome are interpolated from
// the constants that enforce them rather than transcribed, so help that
// disagrees with behaviour cannot be written.
func printCaptureHelp() {
	fmt.Print(captureHelp())
}

// captureHelp is the help text itself, separated from printing it so a test can
// read it without swapping os.Stdout for a pipe.
func captureHelp() string {
	return fmt.Sprintf(`Usage: cudaprof capture [options] -- command [args...]

Options:
  --capture-plan <path>         run a validated bounded provider plan in separate passes
  --capture-plan-cooperative    accept target-owned provider publications and code receipts;
                                requires --capture-plan, preserves replay isolation
  --inject <path>               path to CUPTI agent shared library (or set CUDAPROF_INJECT);
                                the default is cupti_agent.so beside this command or in
                                GOBIN, built by go generate ./x/cmd/cudaprof
  --inject-mode <mode>          how the agent is loaded: preload (default) sets LD_PRELOAD,
                                so every process the target starts loads it; cuda-injection
                                sets CUDA_INJECTION64_PATH, so the driver loads it during
                                cuInit and only processes that use CUDA write a journal
  --no-inject                   run target uninstrumented without injection library
                                CUDAPROF_CORRELATE defaults to 1; an inherited value
                                such as 0 is preserved
  --json                        write JSON status summary to stdout
  --launch-manifest             attach an opt-in metadata-only launch manifest;
                                unknown callback layouts remain UNREAD
  --iteration-marker <regex>    after writing -o, run the cudaprof iterations gate with
                                iterations beginning at kernels matching regex, and fail
                                the capture with status 2 when it fails
  --skip-iterations <N>         warm-up iterations the gate excludes (default 1)
  --expect-markers <N>          the number of times the marker must fire
  --stamp-env <names>           record the values of these environment variables in the
                                capture, comma separated, exact names or a NAME* prefix.
                                The capture always records a digest of the target's whole
                                environment, which says whether two captures are the same
                                arm; this names the variables whose values are disclosed
  -o <path>                     write capture output to file or directory. An existing
                                output is kept and the capture refused, unless it holds no
                                activity records: a capture that observed nothing must not
                                block the retry under its own name. The decision is the
                                file's record count, never a flag or the name
  --max-records <N>             maximum records decoded from one journal (default %d);
                                bounds decoding only -- it cannot recover a record the
                                producer never wrote, and a journal that exceeds it fails
                                the capture rather than being silently truncated
  --capture-file-bytes <N>      maximum bytes for one embedded artifact (default %d);
                                a journal is measured against this compressed, and one that
                                does not fit is omitted, which makes the capture unusable
                                for cudaprof gaps
  --capture-total-bytes <N>     maximum bytes for the whole capture artifact (default %[3]d)
  --observe-isolation         sample GPU residency under an existing live gpu-lock; never acquire one
  --isolation-receipt FILE      retain a bounded parent observer v1 receipt; unbound stays unchecked
  --require-isolation          require sampled clear after retaining the capture
  --allow-partial               exit zero for a capture that omitted an attachment over the
                                budget or holds a journal over --max-records. Without it
                                the capture is still written, and the command exits 2
                                naming the flags to rerun with, sized from this run. The
                                bounds used are recorded as "limits" in capture-status.json
                                and in the capture manifest, so cudaprof's readers admit
                                the capture without restating them
  --capture-range <mode>        %[4]s (default) or %[5]s, which collects only between
                                the target's own cuProfilerStart and cuProfilerStop
                                (driver.ProfilerStart/ProfilerStop in Go); sets
                                CUDAPROF_CAPTURE_RANGE in the target. Each range edge in
                                phase_receipt records the GPUs' SM and memory clocks,
                                temperature and power, read through NVML by cudaprof as
                                it sees the edge, and lag_nanos after it; wall_clock
                                places the edges in wall time
  --finalize <trigger>          when the agent flushes and finalizes: %[6]s (default),
                                %[7]s, on the target's cuProfilerStop, or %[11]s, on
                                the target's exit_group syscall. A Go target exits through
                                exit_group and runs neither the agent's atexit handler nor
                                its ELF destructor, so %[6]s finalization never fires for it;
                                a ProfilerStop before exit, or %[11]s, is what publishes
                                the capture. capture warns before running a Go binary
                                with neither, and fails the run with outcome
                                go-exit-not-finalized when its journal then holds no
                                activity. %[11]s holds the syscall with a seccomp
                                filter until the capture is finalized; the filter sets
                                no_new_privs (an exec cannot gain privileges) and is
                                inherited by children, whose exit_group fails with ENOSYS
                                if they outlive the target. Independent of --capture-range,
                                which decides what is collected rather than when it is
                                published; sets CUDAPROF_FINALIZE
  --graph-trace <mode>          how CUDA graph replays are traced: %[9]s (default), one
                                kernel record per graph node carrying its graph and node
                                ids, or %[10]s, one envelope per replay and no records for
                                the kernels inside it (CUPTI's GRAPH_TRACE kind, cheaper to
                                trace). The capture states the mode in effect as the agent
                                capability's graph_trace_mode; sets CUDAPROF_GRAPH_TRACE
  --flush-interval <duration>   have the agent flush CUPTI's activity buffers this often, so a
                                target that never finalizes still publishes records (sets
                                CUDAPROF_FLUSH_INTERVAL_MS, whole milliseconds). Such a run
                                leaves no finalized record and needs --recover-unfinalized
  --recover-unfinalized         admit records from a generation that decoded completely and
                                passed every identity and ordering check but wrote no finalized
                                record. Its own outcome is unknown, so the capture is reported
                                as %[8]s and is never successful
  --trace                       enable CUDA activity tracing (default true)
  --api-activity[=false]        collect runtime and driver API attribution (default true);
                                it is ~87-92%% of the records a journal holds and a kernel
                                timeline does not need it, so --api-activity=false is the one
                                lever that cuts journal volume by an order of magnitude. It
                                changes what is collected, not what is reported: sets
                                CUDAPROF_API_ACTIVITY in the target
  --host-sample-hz <n>          sample native host thread states and CPU ticks at n Hz (1-1000);
                                0 disables inherited host sampling. This records no stacks;
                                correlate host stacks separately with perf
  --unified-memory             opt in to supported UM migration/fault tracing; native refusals retained
  --memory-policy              retain managed-range query/advice/prefetch request callbacks
  --stacks                     enable instrumented Go/Python stack journals; requires target adapters
  --memsets                     collect memset activity. Off by default: with it on,
                                cuGraphLaunch has faulted inside libcuda for a CUDA-graph
                                MLX workload on GB10. Without it a target's memsets are
                                counted as idle. The capture states the choice as the agent
                                capability's memsets_requested; sets CUDAPROF_MEMSETS to 0
                                or 1 in the target, overriding an inherited value
  --pcsampling                  sample program counters inside the target, so the capture
                                carries per-line device work and not only per-kernel time
                                (sets CUDAPROF_PCSAMPLING). Off by default: measured on GB10
                                it costs 25-27 microseconds of host drain per launch
  --pcsampling-period <n>       sampling period exponent, 5 to 31, smaller is more frequent
                                (implies --pcsampling, sets CUDAPROF_PCSAMPLING_PERIOD)
  --crash-trace <n>             if the target dies of a fault (SIGSEGV, SIGBUS, SIGILL, SIGFPE
                                or SIGABRT), have the agent write its last n CUDA API calls,
                                1 to 4096, to the target's standard error before the process
                                dies (sets CUDAPROF_CRASH_TRACE). Every API call then passes
                                through an agent callback, and a Go target that crashes in C
                                code loses its own traceback
  --gemms                       record each cuBLASLt matmul (shape, types, beta, algorithm)
                                and the kernels it launched, for cudaprof gemms (sets
                                CUDAPROF_GEMMS). Off by default: the library formats a line
                                per call
  --follow-children[=false]     count the generations of processes the target spawned in this
                                capture's totals and projection (default true). Injection is
                                inherited across exec, so a child runs the agent and writes its
                                own generation either way; =false leaves those records out of
                                the totals while their journals stay attached, their counts and
                                dispositions stay in worker_generations, and a child that never
                                finalized still costs the run its completeness. The capture
                                publishes the choice as children_followed
  --exclude <kernel>            drop kernels whose name contains this substring from the
                                projection (repeatable). The agent has already run, so this
                                removes nothing from the capture: the totals above stay journal
                                totals, excluded_records and excluded_nanos report what the
                                projection left out (activity_samples == activity_records -
                                excluded_records), the attached journals still hold every
                                record, and cudaprof gaps, which reads them, still sees the
                                excluded kernels. API records are not matched
  --timeout <duration>          execution timeout
  --drain-timeout <duration>    drain timeout after target completion
  --require-complete            require complete capture without dropped records

Without a capture plan, capture records the existing CUPTI activity timeline.
The opt-in cooperative plan composes target-owned range, PM and SASS collection
with separate replay identities. Missing or incomplete providers remain so.
github.com/tmc/lib/nvidia/x/cuptirange owns range profiling with user replay;
cuptimetrics lists the metrics a given chip offers.
`, captureMaxRecords, captureFileBytes, captureTotalBytes,
		captureRangeNone, captureRangeProfilerAPI, finalizeExit, finalizeProfilerStop, captureOutcomePartialRecovered,
		graphTraceNode, graphTraceGraph, finalizeExitGroup)
}

type deviceProvenance struct {
	Device            string `json:"device,omitempty"`
	Arch              string `json:"arch,omitempty"`
	ComputeCapability string `json:"compute_capability,omitempty"`
	SMCount           int    `json:"sm_count,omitempty"`
	MaxWarpsPerSM     int    `json:"max_warps_per_sm,omitempty"`
	MaxBlocksPerSM    int    `json:"max_blocks_per_sm,omitempty"`
}
