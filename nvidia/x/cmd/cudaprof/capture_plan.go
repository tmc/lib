package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

const capturePlanFileLimit = 1 << 20

type capturePlanRunner func(context.Context, captureOptions) (captureResult, error)

// runCaptureWithPlan keeps the original launcher intact when no plan is set.
// Each admitted activity/PC pass is a separate execution, not state-restoring
// replay. Current external journals cannot bind compiler/workload identities.
func runCaptureWithPlan(ctx context.Context, opts captureOptions, run capturePlanRunner) (captureResult, error) {
	if opts.CapturePlanFile == "" {
		if opts.CapturePlanCooperative {
			return captureResult{}, fmt.Errorf("cooperative capture requires a capture plan")
		}
		return run(ctx, opts)
	}
	f, err := os.Open(opts.CapturePlanFile)
	if err != nil {
		return captureResult{}, fmt.Errorf("open capture plan: %w", err)
	}
	plan, err := cudaprof.ReadCapturePlan(f, capturePlanFileLimit)
	closeErr := f.Close()
	if err != nil {
		return captureResult{}, err
	}
	if closeErr != nil {
		return captureResult{}, closeErr
	}
	opts.CapturePlanFile, err = filepath.Abs(opts.CapturePlanFile)
	if err != nil {
		return captureResult{}, err
	}
	return executeCapturePlan(ctx, opts, plan, run)
}

func executeCapturePlan(ctx context.Context, opts captureOptions, plan cudaprof.CapturePlan, run capturePlanRunner) (captureResult, error) {
	if err := plan.Validate(); err != nil {
		return captureResult{}, err
	}
	start := time.Now()
	ctx, cancel := context.WithTimeout(ctx, time.Duration(plan.Budget.MaxWallNanos))
	defer cancel()
	manifest := cudaprof.CaptureManifest{Version: cudaprof.CapturePlanVersion, Plan: plan}
	var result captureResult
	var firstErr error
	var retained uint64
	var rawBytes int
	nativePasses := uint64(len(plan.Passes))
	for _, pass := range plan.Passes {
		active := opts.CapturePlanCooperative
		for _, provider := range pass.Providers {
			if provider == cudaprof.CaptureActivity || provider == cudaprof.CapturePC {
				active = true
			}
		}
		var observed captureResult
		passExecuted := false
		var runErr error
		budgetErr := ""
		if ctx.Err() != nil {
			budgetErr = "capture plan wall budget exhausted"
		}
		if retained >= plan.Budget.MaxRecords {
			budgetErr = "capture plan record budget exhausted"
		}
		if rawBytes >= plan.Budget.MaxBytes {
			budgetErr = "capture plan byte budget exhausted"
		}
		if budgetErr != "" && firstErr == nil {
			firstErr = fmt.Errorf("%s", budgetErr)
		}
		var raw []byte
		if active && budgetErr == "" && firstErr == nil {
			next := opts
			next.CapturePlanFile = ""
			next.Env = append([]string(nil), opts.Env...)
			if next.Env == nil {
				next.Env = os.Environ()
			}
			next.Env = setCaptureEnv(next.Env, "CUDAPROF_PCSAMPLING", "0")
			next.Env = setCaptureEnv(next.Env, "CUDAPROF_CAPTURE_PLAN", "")
			next.Env = setCaptureEnv(next.Env, "CUDAPROF_CAPTURE_PASS", "")
			if opts.CapturePlanCooperative {
				next.Env = setCaptureEnv(next.Env, "CUDAPROF_CAPTURE_PLAN", opts.CapturePlanFile)
				next.Env = setCaptureEnv(next.Env, "CUDAPROF_CAPTURE_PASS", pass.ID)
				next.Env = setCaptureEnv(next.Env, "CUDAPROF_CAPTURE_MAX_RECORDS", strconv.FormatUint(plan.Budget.MaxRecords-retained, 10))
				next.Env = setCaptureEnv(next.Env, "CUDAPROF_CAPTURE_MAX_BYTES", strconv.Itoa(plan.Budget.MaxBytes-rawBytes))
				// Every declared pass already reserves one launch. This target may
				// consume its reservation and the still-unallocated native replays.
				next.Env = setCaptureEnv(next.Env, "CUDAPROF_CAPTURE_MAX_PASSES", strconv.FormatUint(1+uint64(plan.Budget.MaxPasses)-nativePasses, 10))
				next.Env = setCaptureEnv(next.Env, "CUDAPROF_CAPTURE_DEADLINE_UNIX_NANOS", strconv.FormatInt(start.Add(time.Duration(plan.Budget.MaxWallNanos)).UnixNano(), 10))
			}
			next.PCSampling = false
			next.Trace = false
			for _, provider := range pass.Providers {
				if provider == cudaprof.CaptureActivity {
					next.Trace = true
				}
				if provider == cudaprof.CapturePC {
					next.PCSampling = true
					next.PCSamplingPeriod = int(pass.Configs[provider].Period)
				}
			}
			if opts.CapturePlanCooperative && !next.Trace && !next.PCSampling && next.Inject == "" {
				next.NoInject = true
			}
			next.MaxJournalRecords = int64(plan.Budget.MaxRecords - retained)
			next.CaptureFileBytes = int64(plan.Budget.MaxBytes - rawBytes)
			next.CaptureTotalBytes = next.CaptureFileBytes
			remaining := time.Until(start.Add(time.Duration(plan.Budget.MaxWallNanos)))
			if next.Timeout <= 0 || next.Timeout > remaining {
				next.Timeout = remaining
			}
			if next.DrainTimeout <= 0 || next.DrainTimeout > remaining {
				next.DrainTimeout = remaining
			}
			passExecuted = true
			observed, runErr = run(ctx, next)
			if result.Session == "" {
				result = observed
			}
			if runErr != nil {
				firstErr = runErr
			}
			var err error
			raw, err = capturePlanRaw(observed.ScratchDir, plan.Budget.MaxBytes-rawBytes)
			if err != nil {
				budgetErr = err.Error()
				firstErr = err
			} else {
				rawBytes += len(raw)
			}
		}
		for _, provider := range pass.Providers {
			r := cudaprof.CaptureProviderResult{Provider: provider, PassID: pass.ID, Replay: pass.Replay, Status: "unavailable", EvidenceClass: "unread", Scope: "process", Reason: "provider requires target-cooperative owner capture; controller cannot profile child context"}
			if provider == cudaprof.CapturePM {
				r.Scope = "device-wide"
			}
			if provider == cudaprof.CaptureActivity || provider == cudaprof.CapturePC {
				r.Process = uint64(max(observed.TargetPID, 0))
				if provider == cudaprof.CapturePC {
					r.Scope = "target-context-first"
				}
				if provider == pass.Providers[0] {
					r.Raw = raw
				}
				r.Reason = "external journal lacks admitted image/compiler/workload and observed environment identity"
				if len(raw) > 0 {
					r.Status = "partial"
					r.Decoded = observed.CollectorFinalized
					r.CleanupComplete = observed.CollectorFinalized
				}
				var count uint64
				if provider == cudaprof.CaptureActivity {
					count = observed.ActivityRecords
				} else {
					if observed.PCSampling != nil {
						count = observed.PCSampling.Folded
					}
				}
				charged := count
				if provider == cudaprof.CapturePC && observed.PCSampling != nil {
					charged = max(charged, observed.PCSampling.Total)
				}
				if charged > plan.Budget.MaxRecords-retained {
					budgetErr = "capture plan record budget exceeded"
					firstErr = fmt.Errorf("%s", budgetErr)
				} else {
					retained += charged
					if passExecuted {
						r.Counts.Retained = &count
					}
				}
				if observed.RecordLoss != nil && provider == cudaprof.CaptureActivity {
					dropped := observed.DroppedRecords
					r.Counts.Dropped = &dropped
				}
				if provider == cudaprof.CapturePC && observed.PCSampling != nil {
					r.Config.Period = observed.PCSampling.Period
					offered, dropped, unnamed := observed.PCSampling.Total, observed.PCSampling.HardwareLost, observed.PCSampling.Unkeyed
					r.Counts.Offered = &offered
					r.Counts.Dropped = &dropped
					r.Counts.Unnamed = &unnamed
				}
				if provider == cudaprof.CapturePC && observed.PCSampling == nil {
					r.Status = "unavailable"
					r.Counts.Retained = nil
					r.Reason = "injected PC sampler result unavailable"
				}
				if runErr != nil {
					r.Status = "failed"
					r.Reason = runErr.Error()
				}
				if budgetErr != "" {
					r.Status = "failed"
					r.Reason = budgetErr
					r.Raw = nil
				}
				if firstErr != nil && observed.Session == "" && budgetErr == "" {
					r.Status = "failed"
					r.Reason = "previous pass failed; pass not executed"
				}
			}
			if opts.CapturePlanCooperative && passExecuted {
				published, found, err := capturePlanPublication(observed, plan, pass, provider, plan.Budget.MaxBytes-rawBytes)
				if len(r.Raw) > 0 && len(published.Raw) > 0 {
					var packet capturePlanCodePacket
					if err := json.Unmarshal(published.Raw, &packet); err != nil {
						return result, err
					}
					packet.Journals = r.Raw
					joined, err := json.Marshal(packet)
					if err != nil {
						return result, err
					}
					rawBytes -= len(r.Raw)
					published.Raw = joined
				}
				if err != nil {
					if len(published.Raw) > 0 {
						r = published
						r.EvidenceClass = "unread"
						r.Values = nil
						r.Counts = cudaprof.CaptureCounts{}
						r.Decoded = false
						rawBytes += len(r.Raw)
					}
					r.Status = "failed"
					r.Reason = err.Error()
					if firstErr == nil {
						firstErr = err
					}
				} else if found {

					if provider == cudaprof.CaptureActivity || provider == cudaprof.CapturePC {
						if reconcileErr := reconcileCapturePlanJournal(observed, &published); reconcileErr != nil {
							published.Status = "failed"
							published.Counts = cudaprof.CaptureCounts{}
							published.Values = nil
							published.EvidenceClass = "unread"
							published.Reason = reconcileErr.Error()
							if firstErr == nil {
								firstErr = reconcileErr
							}
						}
					}
					r = published
					if n := r.NativePasses; n != nil && n.Started > 1 {
						extra := n.Started - 1
						if extra > uint64(plan.Budget.MaxPasses)-nativePasses {
							r.Status = "failed"
							r.Reason += "; capture plan native pass budget exceeded"
							if firstErr == nil {
								firstErr = fmt.Errorf("capture plan native pass budget exceeded")
							}
						} else {
							nativePasses += extra
						}
					}

					charged := uint64(0)
					for _, count := range []*uint64{r.Counts.Retained, r.Counts.Offered, r.Counts.Dropped, r.Counts.NonUser, r.Counts.Unnamed} {
						if count != nil {
							charged = max(charged, *count)
						}
					}
					if ((provider != cudaprof.CaptureActivity && provider != cudaprof.CapturePC) && charged > plan.Budget.MaxRecords-retained) || len(r.Raw) > plan.Budget.MaxBytes-rawBytes {
						r.Status = "failed"
						r.Reason += "; capture plan publication budget exceeded"
						r.Raw = nil
						if firstErr == nil {
							firstErr = fmt.Errorf("capture plan publication budget exceeded")
						}
					} else {
						if provider != cudaprof.CaptureActivity && provider != cudaprof.CapturePC {
							retained += charged
						}
						rawBytes += len(r.Raw)
					}
				}
			}
			manifest.Results = append(manifest.Results, r)
		}
	}
	manifest.ElapsedNanos = time.Since(start).Nanoseconds()
	if ctx.Err() != nil {
		firstErr = fmt.Errorf("capture plan wall budget exhausted: %w", ctx.Err())
		for i := range manifest.Results {
			if manifest.Results[i].Status == "complete" || manifest.Results[i].Status == "partial" {
				manifest.Results[i].Status = "failed"
			}
			manifest.Results[i].Reason += "; " + firstErr.Error()
		}
	}
	var encoded bytes.Buffer
	if err := cudaprof.WriteCaptureManifest(&encoded, manifest); err != nil {
		result.CapturePlanError = err.Error()
		if firstErr == nil {
			firstErr = err
		}
	} else {
		result.CapturePlanManifest = append(json.RawMessage(nil), encoded.Bytes()...)
	}
	result.planBudget = &plan.Budget
	result.CaptureSuccess = manifest.Complete() && firstErr == nil && result.TargetExitCode == 0 && result.TargetSignal == "" && !result.Cancelled
	if !result.CaptureSuccess {
		result.CaptureError = "capture plan incomplete: external identities, cooperative providers or loaded code proof unavailable"
	} else {
		result.CaptureError = ""
	}
	if firstErr != nil {
		result.CapturePlanError = firstErr.Error()
	}
	return result, firstErr
}

type capturePlanRawFile struct {
	Name string `json:"name"`
	Data []byte `json:"data"`
}

// capturePlanRaw retains byte-exact journals inside a bounded JSON/base64
// envelope. Files are never truncated into something resembling a valid row.
func capturePlanRaw(dir string, limit int) ([]byte, error) {
	if dir == "" {
		return nil, nil
	}
	var files []capturePlanRawFile
	remaining := limit
	for _, attachment := range scratchAttachments(dir) {
		if !cudacapture.IsJournalName(attachment.name) {
			continue
		}
		if attachment.size > int64(remaining) {
			return nil, fmt.Errorf("capture plan raw journal byte budget exceeded")
		}
		f, err := os.Open(attachment.path)
		if err != nil {
			return nil, err
		}
		data, err := io.ReadAll(io.LimitReader(f, int64(remaining)+1))
		closeErr := f.Close()
		if err != nil {
			return nil, err
		}
		if closeErr != nil {
			return nil, closeErr
		}
		if len(data) > remaining {
			return nil, fmt.Errorf("capture plan raw journal byte budget exceeded")
		}
		remaining -= len(data)
		files = append(files, capturePlanRawFile{attachment.name, data})
	}
	if len(files) == 0 {
		return nil, nil
	}
	data, err := json.Marshal(files)
	if err != nil {
		return nil, err
	}
	if len(data) > limit {
		return nil, fmt.Errorf("capture plan raw envelope byte budget exceeded")
	}
	return data, nil
}

// capturePlanOutputWriter bounds the encoded outer capture, including archive
// headers and profile bytes, before publication. Failed temporary files are
// removed by writeCapture.
type capturePlanOutputWriter struct {
	writer    io.Writer
	remaining int64
}

func (w *capturePlanOutputWriter) Write(p []byte) (int, error) {
	if int64(len(p)) > w.remaining {
		return 0, fmt.Errorf("capture plan encoded output byte budget exceeded")
	}
	n, err := w.writer.Write(p)
	w.remaining -= int64(n)
	return n, err
}

func capturePlanPublication(observed captureResult, plan cudaprof.CapturePlan, pass cudaprof.CapturePass, provider cudaprof.CaptureProvider, limit int) (cudaprof.CaptureProviderResult, bool, error) {
	if observed.ScratchDir == "" {
		return cudaprof.CaptureProviderResult{}, false, nil
	}
	data, err := readCapturePlanFile(observed.ScratchDir, "capture-provider-"+string(provider)+".json", limit)
	if errors.Is(err, os.ErrNotExist) {
		return cudaprof.CaptureProviderResult{}, false, nil
	}
	if err != nil {
		return cudaprof.CaptureProviderResult{}, false, err
	}
	publication, err := cudaprof.ReadCapturePublication(bytes.NewReader(data), plan, limit)
	if err != nil {
		packet, _ := json.Marshal(capturePlanCodePacket{Publication: data})
		if len(packet) > limit {
			return cudaprof.CaptureProviderResult{}, false, fmt.Errorf("capture publication diagnostic byte budget exceeded")
		}
		return cudaprof.CaptureProviderResult{Provider: provider, PassID: pass.ID, Replay: pass.Replay, Raw: packet}, true, fmt.Errorf("read capture publication: %w", err)
	}
	r := publication.Result
	var code []byte
	if publication.Artifact != nil {
		code, err = admitCapturePlanCode(observed.ScratchDir, limit-len(data), publication)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				if r.Status == "complete" {
					r.Status = "partial"
				}
				r.Reason += "; loaded code proof unavailable: " + err.Error()
			} else {
				r.Status = "failed"
				r.Reason += "; loaded code proof invalid: " + err.Error()
			}
		}
	}
	packet, packetErr := json.Marshal(capturePlanCodePacket{Publication: data, CodeCapture: code})
	if packetErr != nil {
		return r, false, packetErr
	}
	if len(packet) > limit {
		r.Raw = nil
		return r, false, fmt.Errorf("capture publication and code packet byte budget exceeded")
	}
	r.Raw = packet
	if r.Status == "failed" && err != nil {
		return r, true, err
	}
	if r.Provider != provider || r.PassID != pass.ID || r.Replay != pass.Replay {
		return r, false, fmt.Errorf("capture publication provider/pass/replay mismatch")
	}
	if observed.TargetPID > 0 && r.Process != 0 && r.Process != uint64(observed.TargetPID) {
		return r, false, fmt.Errorf("capture publication process mismatch")
	}
	return r, true, nil
}

// capturePlanCodePacket retains exact producer publication and admitted existing
// cudacapture code artifact bytes; the outer manifest keeps this JSON as Raw.
type capturePlanCodePacket struct {
	Publication []byte `json:"publication"`
	CodeCapture []byte `json:"code_capture"`
	Journals    []byte `json:"journals,omitempty"`
}

// reconcileCapturePlanJournal compares publisher claims with independently
// decoded injected records. A publisher cannot supply missing native counters.
func reconcileCapturePlanJournal(observed captureResult, r *cudaprof.CaptureProviderResult) error {
	partial := func(reason string) {
		if r.Status == "complete" {
			r.Status = "partial"
		}
		r.Reason += "; " + reason
	}
	match := func(name string, claimed *uint64, actual uint64) error {
		if claimed != nil && *claimed != actual {
			return fmt.Errorf("capture publication %s differs from injected journal", name)
		}
		return nil
	}
	if observed.TargetPID <= 0 || r.Process != uint64(observed.TargetPID) {
		return fmt.Errorf("capture publication process differs from injected journal")
	}
	generationKnown := false
	for _, w := range observed.WorkerGenerations {
		if w.Process == r.Process && w.Generation == r.Generation && w.Finalized && w.Disposition == "finalized" && w.Failure == "" {
			generationKnown = true
		}
	}
	if !generationKnown || !observed.CollectorFinalized || observed.RunCompleteness != runComplete {
		partial("injected generation/finalization or whole-run completeness unavailable")
	}
	var contextID uint32
	var start, stop uint64
	for _, a := range observed.Activities {
		if a.Process != r.Process || !a.Valid {
			continue
		}
		if a.Context != 0 {
			if contextID != 0 && contextID != a.Context {
				return fmt.Errorf("capture publication has multiple injected contexts")
			}
			contextID = a.Context
		}
		if a.Start != 0 && (start == 0 || a.Start < start) {
			start = a.Start
		}
		if a.End > stop {
			stop = a.End
		}
	}
	if contextID == 0 {
		partial("decimal CUPTI context ID unavailable")
	} else if r.Context != strconv.FormatUint(uint64(contextID), 10) {
		return fmt.Errorf("capture publication context differs from injected journal")
	}
	if start == 0 || stop == 0 {
		partial("injected native interval unavailable")
	} else if r.Start != start || r.Stop != stop {
		return fmt.Errorf("capture publication interval differs from injected journal")
	}
	if r.Provider == cudaprof.CaptureActivity {
		if err := match("retained", r.Counts.Retained, observed.ActivityRecords); err != nil {
			return err
		}
		// Admit the native per-kind inventory, never substitute global journal
		// admission for an activity-specific denominator.
		r.Counts.NonUser = nil
		r.Counts.Unnamed = nil
		loss := observed.RecordLoss
		var offered, dropped uint64
		var known bool
		if loss != nil {
			var err error
			offered, dropped, known, err = loss.activityLoss(observed.ActivityRecords)
			if err != nil {
				return err
			}
		}
		if !known {
			r.Counts.Offered = nil
			r.Counts.Dropped = nil
			partial("activity-specific offered/loss denominator unavailable")
			return nil
		}
		if err := match("offered", r.Counts.Offered, offered); err != nil {
			return err
		}
		if err := match("dropped", r.Counts.Dropped, dropped); err != nil {
			return err
		}
		// Native records do not classify non-user or unnamed activity coverage.
		partial("activity non-user/unnamed coverage unavailable")
		return nil
	}
	r.Counts.NonUser = nil
	pc := observed.PCSampling
	if pc == nil {
		r.Counts = cudaprof.CaptureCounts{}
		partial("injected PC sampler summary unavailable")
		return nil
	}
	for _, check := range []struct {
		name   string
		claim  *uint64
		actual uint64
	}{
		{"retained", r.Counts.Retained, pc.Folded}, {"offered", r.Counts.Offered, pc.Total}, {"dropped", r.Counts.Dropped, pc.HardwareLost}, {"unnamed", r.Counts.Unnamed, pc.Unkeyed},
	} {
		if err := match(check.name, check.claim, check.actual); err != nil {
			return err
		}
	}
	if r.Config.Period != pc.Period {
		return fmt.Errorf("capture publication PC period differs from injected journal")
	}
	if pc.Overflows != 0 {
		partial("injected PC fold table overflow")
	}
	partial("PC non-user and loaded-module sample coverage unavailable")
	return nil
}
