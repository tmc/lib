package main

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/tmc/lib/nvidia/internal/cudainject/journal"
	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

type captureHandshake struct {
	Protocol         int    `json:"protocol"`
	Session          string `json:"session"`
	Process          uint64 `json:"process"`
	Generation       uint64 `json:"generation"`
	TransportReady   bool   `json:"transport_ready"`
	InjectionFailure string `json:"injection_failure,omitempty"`
}

type captureRecordPayload struct {
	State               string     `json:"state"`
	Kind                string     `json:"kind"`
	Subtype             string     `json:"subtype"`
	ModuleID            uint32     `json:"module_id"`
	ModuleSize          uint64     `json:"module_size"`
	ModuleFile          string     `json:"module_file"`
	ModuleStored        bool       `json:"module_stored"`
	ModuleReason        string     `json:"module_reason"`
	ModuleUnchecked     bool       `json:"module_unchecked"`
	PCFunction          string     `json:"pc_function"`
	PCOffset            uint64     `json:"pc_offset"`
	PCSamples           uint64     `json:"pc_samples"`
	Name                string     `json:"name"`
	Error               string     `json:"error"`
	Message             string     `json:"message"`
	Operation           string     `json:"operation"`
	Valid               bool       `json:"valid"`
	Start               uint64     `json:"start"`
	End                 uint64     `json:"end"`
	Bytes               uint64     `json:"bytes"`
	Queued              uint64     `json:"queued"`
	Submitted           uint64     `json:"submitted"`
	CopyKind            uint8      `json:"copy_kind"`
	Value               uint32     `json:"value"`
	OverheadKind        uint32     `json:"overhead_kind"`
	ObjectKind          uint32     `json:"object_kind"`
	SyncType            uint32     `json:"sync_type"`
	CudaEventID         uint32     `json:"cuda_event_id"`
	Device              uint32     `json:"device"`
	Context             uint32     `json:"context"`
	Stream              uint32     `json:"stream"`
	Correlation         uint32     `json:"correlation"`
	RuntimeCorrelation  uint32     `json:"runtime_correlation"`
	ExternalKind        uint32     `json:"external_kind"`
	ExternalID          uint64     `json:"external_id"`
	Domain              string     `json:"domain"`
	LifecycleEvent      string     `json:"lifecycle_event"`
	DependencyEvent     string     `json:"dependency_event"`
	Site                string     `json:"site"`
	Ordinal             uint64     `json:"ordinal"`
	CallbackID          uint32     `json:"callback_id"`
	ProcessID           uint32     `json:"process_id"`
	ThreadID            uint32     `json:"thread_id"`
	MarkerDomain        string     `json:"marker_domain"`
	MarkerID            uint32     `json:"marker_id"`
	MarkerFlags         uint32     `json:"marker_flags"`
	ReturnValue         uint32     `json:"return_value"`
	GridDim             [3]uint32  `json:"grid"`
	GridDimRaw          *[3]uint32 `json:"grid_dim"`
	GridX               uint32     `json:"grid_x"`
	GridY               uint32     `json:"grid_y"`
	GridZ               uint32     `json:"grid_z"`
	BlockDim            [3]uint32  `json:"block"`
	BlockDimRaw         *[3]uint32 `json:"block_dim"`
	BlockX              uint32     `json:"block_x"`
	BlockY              uint32     `json:"block_y"`
	BlockZ              uint32     `json:"block_z"`
	RegistersPerThread  uint32     `json:"registers_per_thread"`
	Registers           uint32     `json:"registers"`
	DynamicSharedMemory uint32     `json:"dynamic_shared_memory"`
	Smem                uint32     `json:"smem"`
	GraphID             uint32     `json:"graph_id"`
	GraphNodeID         uint64     `json:"graph_node_id"`
	Edge                string     `json:"edge"`
	API                 string     `json:"api"`
	Index               uint32     `json:"index"`
	ToggleFailures      uint32     `json:"toggle_failures"`
	FlushNanos          uint64     `json:"flush_nanos"`
	Dropped             uint64     `json:"dropped"`
	// The finalized record carries the breakdown of what the producer lost.
	// Dropped above is their sum, and a sum cannot be acted on: the mechanisms
	// have different remedies and one of them is not a remedy the reader has.
	NativeDropped   uint64             `json:"native_dropped"`
	QueueDropped    uint64             `json:"queue_dropped"`
	QueueOversize   uint64             `json:"queue_oversize_dropped"`
	QueueContention uint64             `json:"queue_contention_dropped"`
	QueueCapacity   uint64             `json:"queue_capacity_dropped"`
	Offered         uint64             `json:"offered"`
	Retained        uint64             `json:"retained"`
	ByKind          *captureKindLosses `json:"by_kind"`
	// The pcsampling summary record. The agent folds samples in fixed-size
	// tables, so these say how much of the run the pcsample records above
	// actually account for.
	PCPeriod             uint32 `json:"pc_period"`
	PCHardwareBytes      uint64 `json:"pc_hardware_bytes"`
	PCDistinct           uint32 `json:"pc_distinct"`
	PCWritten            uint32 `json:"pc_written"`
	PCFolded             uint64 `json:"pc_folded"`
	PCTotal              uint64 `json:"pc_total"`
	PCHardwareLost       uint64 `json:"pc_hw_dropped"`
	PCUnkeyed            uint64 `json:"pc_unkeyed"`
	PCOverflows          uint64 `json:"pc_overflows"`
	BackpressureMaxNanos uint64 `json:"backpressure_max_nanos"`
	DrainWriteMaxNanos   uint64 `json:"drain_write_max_nanos"`
}

// observeOptions carries the caller's explicit choices into journal
// observation. The zero value is the default policy: only generations that
// wrote a finalized record contribute anything, and the generations of
// processes the target spawned are part of the capture.
type observeOptions struct {
	injectionRequested bool
	// cudaInjection says the agent was loaded through CUDA_INJECTION64_PATH,
	// which only a process that initializes CUDA loads. A target with no
	// generation of its own is then a target that ran no CUDA, as a
	// preloaded target reports in a journal of its own.
	cudaInjection bool
	noInject      bool
	// targetOnly restricts the totals and the projection to the target's own
	// generation. It is spelled negatively because following children is the
	// default -- injection is inherited across exec, so a child is already
	// running the agent and already writing its own generation, and a capture
	// of a wrapper script that reported nothing would be describing a run it
	// did observe. --follow-children=false sets this.
	//
	// It withholds records from the totals; it never withholds evidence. Every
	// generation stays in WorkerGenerations with its own counts and
	// disposition, every worker journal stays attached to the artifact, and an
	// unfollowed generation that failed to qualify still costs the run its
	// completeness. The one thing that changes is which records the capture
	// claims as its own, and ChildrenFollowed publishes that choice.
	targetOnly bool
	// recoverUnfinalized admits records from a generation that decoded
	// completely but never finalized. Such a generation is labelled
	// "unfinalized-partial", counted separately in RecoveredGenerations, and
	// forces the partial-recovered outcome; it never makes the capture
	// successful. It applies to the target generation as well as to workers:
	// a periodic flush (CUDAPROF_FLUSH_INTERVAL_MS) publishes real records from
	// a target that then exits without finalizing, and refusing those is
	// refusing the whole capture.
	recoverUnfinalized bool
	maxJournalRecords  int64
	// goTarget is the target's identity when it is a Go binary run under an
	// agent that finalizes on exit (see goExitRisk). A target journal that
	// then ends unfinalized with no activity record is reported as
	// captureOutcomeGoExit, naming the flag that would have published it.
	goTarget *executableIdentity
}

// observeCaptureSession observes one capture session and always stamps the
// whole-run completeness of what it observed, at every terminal path. A
// session that retained no activity records is never successful.
func observeCaptureSession(result captureResult, dir, session string, opts observeOptions) captureResult {
	result.ChildrenFollowed = !opts.targetOnly
	return refuseEmpty(withRunCompleteness(observeSession(result, dir, session, opts)))
}

// withRunCompleteness classifies the run from evidence the capture already
// retains. It never modifies CaptureSuccess: the two answer different
// questions, and collapsing them is what let a target-qualified boolean stand
// for a run whose worker generations failed.
func withRunCompleteness(result captureResult) captureResult {
	result.UnqualifiedGenerations = unqualifiedGenerations(result.WorkerGenerations)
	result.RunCompleteness = runCompletenessOf(result.TransportReady, result.CollectorFinalized, result.UnqualifiedGenerations, result.DroppedRecords, result.ActivityRecords)
	return result
}

// runCompletenessOf is the single classification rule, shared by the writer and
// by inspect so a reader can never report a different answer than the capture.
func runCompletenessOf(transportReady, collectorFinalized bool, unqualified, dropped, records uint64) string {
	switch {
	case !transportReady:
		return runUnknown
	case !collectorFinalized:
		return runFailed
	case unqualified > 0 || dropped > 0:
		return runPartial
	case records == 0:
		return runEmpty
	}
	return runComplete
}

// unqualifiedGenerations counts generations that did not qualify. A generation
// qualifies only by finalizing without a reported failure; a finalized
// generation that retained no records is qualified, because a zero count is
// not evidence of lost work.
func unqualifiedGenerations(workers []captureWorker) uint64 {
	var n uint64
	for _, w := range workers {
		if w.Disposition == captureWorkerNoCUDA && w.Failure == "" {
			// A generation that ran no CUDA had nothing to finalize and lost
			// nothing by not doing so. Counting it would turn every capture of
			// a wrapper script partial.
			continue
		}
		if w.Disposition != "finalized" || w.Failure != "" {
			n++
		}
	}
	return n
}

func observeSession(result captureResult, dir, session string, opts observeOptions) captureResult {
	paths := []string{filepath.Join(dir, "handshake.json")}
	if matches, err := filepath.Glob(filepath.Join(dir, "process-*", "handshake.json")); err == nil {
		paths = append(paths, matches...)
	}
	type handshakeFile struct {
		path string
		h    captureHandshake
	}
	files := make([]handshakeFile, 0, len(paths))
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			result.Outcome, result.CaptureError = captureOutcomeTransport, fmt.Sprintf("read handshake: %v", err)
			return result
		}
		var h captureHandshake
		if err := json.Unmarshal(data, &h); err != nil {
			result.Outcome, result.CaptureError = captureOutcomeTransport, fmt.Sprintf("parse handshake: %v", err)
			return result
		}
		if h.Protocol != captureProtocolVersion || h.Session != session || h.Process == 0 || h.Generation == 0 {
			result.Outcome, result.CaptureError = captureOutcomeTransport, "handshake identity or protocol mismatch"
			return result
		}
		files = append(files, handshakeFile{path: filepath.Dir(path), h: h})
	}
	if len(files) == 0 {
		if opts.cudaInjection {
			result.Outcome = captureOutcomeNoCUDA
			result.CaptureError = "no process loaded the agent: under --inject-mode=cuda-injection the driver loads it only during cuInit, so either nothing the target ran initialized CUDA or the driver did not load CUDA_INJECTION64_PATH"
		}
		return result
	}
	var selected *handshakeFile
	var selectedState journalState
	var selectedMetrics journalMetrics
	var selectedRecords wire.Stats
	var selectedActivities []captureActivity
	var selectedAPIs []captureAPI
	var selectedPhaseEdges []capturePhaseEdge
	var workerProducers []contributingProducer
	var workerActivities []captureActivity
	var workerAPIs []captureAPI
	var workerDuration, workerCopyBytes, workerExternal, workerHostSamples, workerDropped, workerRecords uint64
	var workerModules []captureModule
	var workerPCSamples []capturePCSample
	workers := make([]captureWorker, 0)
	var targetHandshakeSeen bool
	var targetInjectionFailure string
	var targetFailureState journalState
	var targetJournalErr error
	var targetObserved bool
	var targetObservedState journalState
	// targetObservedTransport is the transport readiness the target itself
	// reported in its handshake, kept for the refusal path below.
	var targetObservedTransport bool
	var targetObservedActivity uint64
	var targetRecovered bool
	// targetNoCUDA reports a target that ran no CUDA of its own. It is not a
	// failure and not a recovery: there was nothing to collect in that
	// process, and the capture is whatever the processes it started collected.
	var targetNoCUDA bool
	// injectedWorker is the first finalized worker generation, which stands
	// in for the target's own capability record when cuda-injection left the
	// target without a generation.
	var injectedWorker *journalState
	var injectedWorkerTransport bool
	// Recovering a worker generation means admitting its records, so it is
	// only on offer when those records are being admitted at all.
	recoverWorkers := opts.recoverUnfinalized && !opts.targetOnly
	for i := range files {
		f := &files[i]
		if f.h.Process != uint64(result.TargetPID) {
			worker := captureWorker{Process: f.h.Process, Generation: f.h.Generation, Disposition: "incomplete"}
			if f.h.InjectionFailure != "" {
				worker.Failure, worker.Disposition = f.h.InjectionFailure, "injection-failed"
			} else if records, state, metrics, activities, apis, _, err := observeJournal(filepath.Join(f.path, "capture.journal"), session, int(f.h.Process), opts.maxJournalRecords); err == nil || (recoverWorkers && errors.Is(err, errJournalNotFinalized)) {
				result.noteJournal(state, err)
				// Identity is checked before anything is contributed, whether
				// or not the generation finalized.
				if state.generation != f.h.Generation {
					worker.Failure, worker.Disposition = "handshake generation mismatch", "identity-mismatch"
					workers = append(workers, worker)
					continue
				}
				worker.Records, worker.ActivityRecords, worker.Finalized = state.records, records.Records, state.finalized
				if !opts.targetOnly && (state.finalized || err != nil) {
					for i := range activities {
						activities[i].Process = f.h.Process
					}
					workerProducers = append(workerProducers, contributingProducer{process: f.h.Process, revision: state.agentRevision, records: records.Records})
					workerActivities = append(workerActivities, activities...)
					workerAPIs = append(workerAPIs, apis...)
					workerDuration += metrics.duration
					workerCopyBytes += metrics.copyBytes
					workerExternal += metrics.external
					workerHostSamples += metrics.hostSamples
					workerModules = append(workerModules, metrics.modules...)
					workerPCSamples = append(workerPCSamples, metrics.pcSamples...)
					workerDropped += records.Dropped
					workerRecords += records.Records
				}
				switch {
				case err != nil:
					// Recovered but unfinalized: the loss disposition, if the
					// producer reported one, stays visible in Failure.
					worker.Disposition = captureWorkerUnfinalizedPartial
					worker.Failure = state.failure
					if worker.Failure == "" {
						worker.Failure = err.Error()
					}
					result.RecoveredGenerations++
					result.RecoveredActivityRecords += records.Records
					result.RecoveredDroppedRecords += records.Dropped
				case state.failure != "":
					worker.Failure, worker.Disposition = state.failure, state.failure
				case state.finalized:
					worker.Disposition = "finalized"
					if injectedWorker == nil {
						st := state
						injectedWorker, injectedWorkerTransport = &st, f.h.TransportReady
					}
				}
			} else {
				result.noteJournal(state, err)
				worker.Failure = err.Error()
				if state.failure != "" {
					worker.Failure = state.failure
					if state.failureError != "" {
						worker.Failure += ": " + state.failureError
					}
				}
			}
			workers = append(workers, worker)
			continue
		}
		targetHandshakeSeen = true
		if f.h.InjectionFailure != "" {
			targetInjectionFailure = f.h.InjectionFailure
			workers = append(workers, captureWorker{Process: f.h.Process, Generation: f.h.Generation, Failure: f.h.InjectionFailure, Disposition: "injection-failed"})
			continue
		}
		records, state, metrics, activities, apis, innerWorkers, err := observeJournal(filepath.Join(f.path, "capture.journal"), session, result.TargetPID, opts.maxJournalRecords)
		result.noteJournal(state, err)
		if state.failure != "" {
			targetFailureState = state
		}
		// A target whose journal ends unfinalized having never been asked for
		// an activity buffer ran no CUDA. CUPTI asks only when it has device
		// work to store, and the producer notes that request as it happens, so
		// the note is present exactly when records were there to lose. A shell
		// that starts a CUDA program is the ordinary case, and refusing the
		// run over the shell discards every record the program collected.
		if err != nil && errors.Is(err, errJournalNotFinalized) && state.ranNoCUDA(records) && state.generation == f.h.Generation {
			targetNoCUDA = true
			targetObservedState, targetObservedTransport = state, f.h.TransportReady
			workers = append(workers, captureWorker{Process: f.h.Process, Generation: f.h.Generation, Records: state.records, Disposition: captureWorkerNoCUDA})
			continue
		}
		admitPartial := err != nil && recoverableTargetGeneration(opts, err, state, records)
		if err != nil && !admitPartial {
			targetJournalErr = err
			// observeJournal returns errJournalNotFinalized together with
			// everything it did decode. Keep those values for the diagnosis
			// below; the capture is still refused, but it must not describe
			// what it read as absent.
			targetObserved, targetObservedState, targetObservedActivity = true, state, records.Records
			targetObservedTransport = f.h.TransportReady
			failMsg := err.Error()
			if state.failure != "" {
				failMsg = state.failure
				if state.failureError != "" {
					failMsg += ": " + state.failureError
				}
			}
			workers = append(workers, captureWorker{Process: f.h.Process, Generation: f.h.Generation, Failure: failMsg, Disposition: "invalid"})
			continue
		}
		if selected != nil {
			return resultWithWorkers(result, workers, "capture journal contains multiple finalized target generations")
		}
		if state.generation != f.h.Generation {
			workers = append(workers, captureWorker{Process: f.h.Process, Generation: f.h.Generation, Failure: "handshake generation mismatch", Disposition: "identity-mismatch"})
			continue
		}
		if admitPartial {
			// The target's own generation is now one of the unqualified ones,
			// and it says so in the manifest like any other: a reader counting
			// dispositions must see it, and RunCompleteness is computed from
			// exactly this list plus CollectorFinalized, which stays false.
			targetRecovered = true
			result.RecoveredGenerations++
			result.RecoveredActivityRecords += records.Records
			result.RecoveredDroppedRecords += records.Dropped
			workers = append(workers, captureWorker{Process: f.h.Process, Generation: f.h.Generation, Records: state.records, ActivityRecords: records.Records, Failure: err.Error(), Disposition: captureWorkerUnfinalizedPartial})
		}
		workers = append(workers, innerWorkers...)
		selected, selectedState, selectedMetrics, selectedRecords = f, state, metrics, records
		selectedActivities, selectedAPIs = activities, apis
		selectedPhaseEdges = selectedMetrics.phaseEdges
	}
	// Under cuda-injection a target that never initialized CUDA never loaded
	// the agent, so it has no generation to say so. Its children's are the
	// capture, exactly as they are behind a preloaded target that reported
	// no CUDA, and the first of them supplies the capability record.
	if opts.cudaInjection && !targetHandshakeSeen && injectedWorker != nil {
		targetNoCUDA = true
		targetObservedState, targetObservedTransport = *injectedWorker, injectedWorkerTransport
	}
	result.WorkerGenerations = workers
	// mergeWorkers folds the followed generations into the totals. The target
	// path and the non-CUDA-parent path both need it, and a capture must not
	// count a worker differently depending on which process held the
	// capability record.
	mergeWorkers := func(result captureResult) captureResult {
		// Modules are merged before the activity guard: a generation that
		// loaded device code and launched none of it still loaded it, and the
		// image is the evidence of what was there to launch.
		result.Modules = append(result.Modules, workerModules...)
		result.PCSamples = append(result.PCSamples, workerPCSamples...)
		if len(workerActivities) == 0 {
			return result
		}
		result.Activities = append(result.Activities, workerActivities...)
		result.ActivityNanos += workerDuration
		result.CopyBytes += workerCopyBytes
		result.ActivityRecords += workerRecords
		result.DroppedRecords += workerDropped
		result.ExternalRecords += workerExternal
		result.HostSampleRecords += workerHostSamples
		result.APIRecords = append(result.APIRecords, workerAPIs...)
		return result
	}
	if selected == nil {
		if targetInjectionFailure != "" {
			result.Outcome, result.CaptureError = captureOutcomeInjection, targetInjectionFailure
		} else if targetFailureState.failure != "" {
			errStr := targetFailureState.failure
			if targetFailureState.failureError != "" {
				errStr += ": " + targetFailureState.failureError
			}
			if targetFailureState.failure == "cupti_unavailable" {
				result.Outcome, result.CaptureError = captureOutcomeUnsupported, errStr
			} else {
				result.Outcome, result.CaptureError = captureOutcomeInjection, errStr
			}
		} else if targetJournalErr != nil {
			// Only the unfinalized case is named for its stage. Every other
			// journal error -- truncation, bad ordering, identity mismatch --
			// is a failure to get intact records across, which is what
			// transport-failed says.
			outcome := captureOutcomeTransport
			if errors.Is(targetJournalErr, errJournalNotFinalized) {
				outcome = captureOutcomeNotFinalized
			}
			// The diagnosis expands the sentinel wherever the journal says
			// what to change. Where it does not -- it returns the error
			// unchanged -- the hint can still name the cause from the exit
			// code, so the two compose rather than both speaking at once.
			diagnosis := unfinalizedTargetDiagnosis(targetJournalErr, targetObservedState, targetObservedActivity)
			if diagnosis == targetJournalErr.Error() {
				diagnosis += unfinalizedHint(targetJournalErr, result)
			}
			if opts.goTarget != nil && outcome == captureOutcomeNotFinalized && targetObservedState.bootstrap && targetObservedActivity == 0 {
				outcome, diagnosis = captureOutcomeGoExit, goExitDiagnosis(opts.goTarget)
			}
			result.Outcome, result.CaptureError = outcome, diagnosis
		} else if targetNoCUDA {
			// The capture is the followed generations' work. It keeps the
			// target's capability record -- injection is inherited across
			// exec, so it describes the same agent and the same flags -- with
			// the producer scope saying whose records the totals are.
			result.TransportReady = targetObservedTransport
			result.BootstrapSeen = targetObservedState.bootstrap
			result.CollectorFinalized = followedCollectorsPublished(workers)
			result.Capability = stampProducerScope(stampProducer(targetObservedState.capabilityData, buildProducer()), targetObservedState.agentRevision, workerProducers)
			result = mergeWorkers(result)
			if result.ActivityRecords > 0 {
				result.Outcome, result.CaptureSuccess = captureOutcomeCollected, true
			} else {
				// Same answer a finalized generation gets for the same
				// evidence: the agent came up and nothing ran under it.
				result.Outcome = "initialized-no-activity"
			}
			return applyPartialRecovery(result)
		} else if len(workers) > 0 && workers[len(workers)-1].Failure != "" {
			w := workers[len(workers)-1]
			if w.Disposition == "injection-failed" {
				result.Outcome, result.CaptureError = captureOutcomeInjection, w.Failure
			} else {
				result.Outcome, result.CaptureError = captureOutcomeTransport, w.Failure
			}
		} else if !targetHandshakeSeen {
			if !opts.injectionRequested {
				if opts.noInject {
					result.Outcome = captureOutcomeNoCUDA
					result.CaptureSuccess = true
					result.RunCompleteness = runComplete
					return result
				}
				result.Outcome, result.CaptureError = captureOutcomeNoCUDA, "no injection library configured (specify --inject /path/to/agent.so or CUDAPROF_INJECT)"
			} else {
				result.Outcome, result.CaptureError = captureOutcomeTransport, "no identity-matching target handshake"
			}
		} else {
			result.Outcome, result.CaptureError = captureOutcomeTransport, "target journal incomplete"
		}
		// A refused journal is not an unread one. Reporting a recorded
		// bootstrap as bootstrap_seen:false sends the reader hunting for an
		// injection or transport fault that did not happen; the real fault is
		// whatever kept the producer from finalizing.
		//
		// transport_ready is the same mistake in the field next to it. It was
		// set only on the path that reaches a usable journal, so a target that
		// handed over a handshake and wrote records still published
		// transport_ready:false, and runCompletenessOf turned that into
		// "unknown" -- whose own definition is "no transport was established".
		// A reader is then told the channel never came up by two fields at
		// once, about a run where it demonstrably did. Report what the target
		// said instead, which leaves the failure where it belongs: transport
		// up, collector not finalized, run_completeness "failed".
		if targetObserved {
			result.BootstrapSeen = targetObservedState.bootstrap
			result.TransportReady = targetObservedTransport
		}
		return applyPartialRecovery(result)
	}
	result.TransportReady = true
	result.BootstrapSeen, result.CollectorFinalized = selectedState.bootstrap, selectedState.finalized
	result.ActivityRecords, result.DroppedRecords = selectedRecords.Records, selectedRecords.Dropped
	result.ExternalRecords = selectedMetrics.external
	result.HostSampleRecords = selectedMetrics.hostSamples
	result.Modules = selectedMetrics.modules
	result.PCSamples, result.PCSampling = selectedMetrics.pcSamples, selectedMetrics.pcSampling
	result.RecordLoss = selectedMetrics.loss.orNil()
	result.ActivityNanos, result.CopyBytes, result.Activities = selectedMetrics.duration, selectedMetrics.copyBytes, selectedActivities
	result.APIRecords = selectedAPIs
	result = mergeWorkers(result)
	for i := range result.Activities {
		if result.Activities[i].Process == 0 {
			result.Activities[i].Process = uint64(result.TargetPID)
		}
	}
	result.Capability = stampProducer(selectedState.capabilityData, buildProducer())
	// The capability record comes from the target's journal alone, but the
	// activity records above it may have come from several processes, each with
	// its own agent. Say so when they are not the same producer.
	result.Capability = stampProducerScope(result.Capability, selectedState.agentRevision, workerProducers)
	result.PhaseReceipt = buildPhaseReceipt(result, selectedPhaseEdges)
	if targetRecovered {
		// The generation decoded completely and passed every identity and
		// ordering check, so its records stand; what is unknown is whether it
		// had more to write. It therefore never reaches the outcome ladder
		// below -- no branch of it can describe a generation that did not
		// finalize -- and applyPartialRecovery is what labels the result.
		//
		// The receipt is built above this, not below it: a recovered
		// generation has phase edges like any other, and skipping the ladder
		// is not a reason to publish it without one.
		return applyPartialRecovery(result)
	}
	if selectedState.failure != "" {
		errStr := selectedState.failure
		if selectedState.failureError != "" {
			errStr += ": " + selectedState.failureError
		}
		result.CaptureError = errStr
		if selectedState.failure == "cupti_unavailable" {
			result.Outcome = captureOutcomeUnsupported
		} else {
			result.Outcome = captureOutcomeInjection
		}
		return result
	}
	if !selectedState.bootstrap || !selectedState.capability || !selectedState.finalized {
		result.Outcome = "initialized-no-activity"
		return result
	}
	if selectedState.noCUDA && result.ActivityRecords == 0 {
		result.Outcome, result.CaptureSuccess = captureOutcomeObserved, true
	} else if selectedState.initializedNoActivity && result.ActivityRecords == 0 {
		result.Outcome = "initialized-no-activity"
	} else if result.ActivityRecords > 0 {
		result.Outcome, result.CaptureSuccess = captureOutcomeCollected, true
	}
	_ = opts.injectionRequested
	return applyPartialRecovery(result)
}

// captureWorkerUnfinalizedPartial labels a worker generation whose records
// were admitted by explicit diagnostic partial recovery. It is never a
// substitute for "finalized": the generation's own completion is unknown, so
// its counts are a floor, not a total.
const captureWorkerUnfinalizedPartial = "unfinalized-partial"

// captureWorkerNoCUDA labels a generation that ran no CUDA: the agent loaded,
// CUPTI never asked it for an activity buffer, and the process ended without
// finalizing a journal that had nothing in it to publish.
const captureWorkerNoCUDA = "no-cuda"

// followedCollectorsPublished reports whether the collection a target that ran
// no CUDA stood aside for was published.
//
// The target itself had nothing to finalize, so the question is only about the
// generations that did collect, and the answer is true when none of them ended
// without publishing -- including the run where no other process started at
// all, which collected nothing and lost nothing. A generation that published a
// failure is published: the failure is in its record and counts as unqualified,
// which is what turns the run partial rather than failed.
func followedCollectorsPublished(workers []captureWorker) bool {
	for _, w := range workers {
		if w.Disposition != captureWorkerNoCUDA && w.Disposition != "finalized" {
			return false
		}
	}
	return true
}

// recoverableTargetGeneration reports whether an unfinalized target generation
// may contribute its records as an explicit partial. Three conditions, each
// for its own reason:
//
// The sentinel, because every other observeJournal error means the journal did
// not decode and its records were never trustworthy.
//
// No producer failure, because a target that recorded one has a cause worth
// more than its records: the existing path reports that cause as the outcome
// (cupti_unavailable becomes unsupported, anything else injection-failed), and
// relabelling it partial-recovered would bury the one thing the caller needs.
//
// At least one activity record, because a generation that flushed nothing has
// nothing to recover. Admitting it would replace the diagnosis below with an
// empty partial capture, which is the same silence with a friendlier name.
func recoverableTargetGeneration(opts observeOptions, err error, state journalState, records wire.Stats) bool {
	return opts.recoverUnfinalized && errors.Is(err, errJournalNotFinalized) && state.failure == "" && records.Records > 0
}

// unfinalizedTargetDiagnosis expands the missing-finalize sentinel for the one
// journal shape that has a remedy, and leaves every other one alone.
//
// "capture journal missing finalized record" is true of a Go target and tells
// it nothing. The shape is specific: the agent loaded, published its
// capability, and then wrote neither an activity record nor a finalized one.
// That is not an injection fault and not a missing libcupti -- both of those
// stop before the capability record. CUPTI hands activity buffers over only
// when they fill or when something flushes them, and every flush the agent
// owns hangs off process exit: its ELF destructor, its atexit handler, its
// SIGTERM terminator thread. A Go runtime exits through exit_group and reaches
// none of them, so the buffers die with the process. Measured on 2026-09-20:
// one agent, 51 records from a C target and 0 from a Go one, and a --timeout
// SIGTERM did not change it.
//
// A generation that did write activity records reached a flush, so that
// diagnosis would be wrong for it. It gets the other one: its records decoded
// and are being thrown away for want of a flag.
func unfinalizedTargetDiagnosis(err error, state journalState, activityRecords uint64) string {
	if !errors.Is(err, errJournalNotFinalized) || !state.bootstrap || !state.capability {
		return err.Error()
	}
	if activityRecords > 0 {
		return fmt.Sprintf("%v: the journal holds %d activity records that decoded completely and passed every identity and ordering check, and the generation that wrote them never finalized, so its own outcome is unknown; re-run with --recover-unfinalized to admit them as an explicitly partial capture", err, activityRecords)
	}
	return err.Error() + ": the agent loaded and published its capability, then wrote no activity record and never finalized" +
		" -- CUPTI releases activity buffers only on flush or when they fill, and a target that exits through exit_group, as every Go program does," +
		" runs none of the agent's exit-time flushes. Either have the target call driver.ProfilerStop() (github.com/tmc/lib/nvidia/driver) before it exits" +
		" and re-run with --finalize=profiler-stop, which publishes the capture on that call -- add --capture-range=cudaProfilerApi to collect only between" +
		" ProfilerStart and ProfilerStop -- or re-run with --flush-interval=250ms --recover-unfinalized, which flushes while the target is alive and admits" +
		" what it flushed as an explicitly partial capture"
}

// applyPartialRecovery marks a result that includes recovered unfinalized
// records. It never sets CaptureSuccess and never leaves a collected outcome
// standing, so no consumer can read the totals as a complete profile.
func applyPartialRecovery(result captureResult) captureResult {
	if result.RecoveredGenerations == 0 {
		return result
	}
	result.Outcome, result.CaptureSuccess = captureOutcomePartialRecovered, false
	if result.CaptureError == "" {
		result.CaptureError = fmt.Sprintf("%d generation(s) recovered without a finalized record", result.RecoveredGenerations)
	}
	return result
}

func resultWithWorkers(result captureResult, workers []captureWorker, message string) captureResult {
	result.WorkerGenerations = workers
	result.Outcome, result.CaptureError = captureOutcomeTransport, message
	return applyPartialRecovery(result)
}

type journalState struct {
	bootstrap, capability, finalized, noCUDA, initializedNoActivity bool
	// deviceActive reports the producer's note that CUPTI asked it for an
	// activity buffer. It is what separates a generation that lost its
	// records from one that had none to lose: both end with a journal holding
	// bootstrap and capability alone.
	deviceActive   bool
	generation     uint64
	records        uint64
	capabilityData json.RawMessage
	// agentRevision is the source revision the agent that wrote this journal
	// stamped into its capability record, "" when it named none. A session
	// merges journals from several processes and publishes one capability
	// record; this is what says whether that record describes them all.
	agentRevision string
	failure       string
	failureError  string
}

// ranNoCUDA reports whether the generation this state describes ran no CUDA
// at all, by the rule of [timeline.NoCUDA.Proven].
func (s journalState) ranNoCUDA(records wire.Stats) bool {
	return timeline.NoCUDA{
		DeviceActive: s.deviceActive,
		Failed:       s.failure != "",
		Work:         records.Records,
		Capability:   s.capabilityData,
	}.Proven()
}

type journalMetrics struct {
	duration, copyBytes, external, hostSamples uint64
	loss                                       captureLoss
	phaseEdges                                 []capturePhaseEdge
	modules                                    []captureModule
	pcSamples                                  []capturePCSample
	pcSampling                                 *capturePCSampling
}

// capturePCSample is one program counter the agent's sampler saw, folded over
// the whole run. Offset is relative to Function, which is how CUPTI reports a
// sampled PC; ModuleID is the module whose image CUPTI's cubin CRC matched, 0
// when no loaded image did.
type capturePCSample struct {
	ModuleID uint32 `json:"module_id"`
	Function string `json:"function"`
	Offset   uint64 `json:"offset"`
	Samples  uint64 `json:"samples"`
}

// capturePCSampling is what the sampler itself did. Folded against Total says
// how many samples reached the table, and the rest name the three ways a
// sample does not: the hardware buffer filled, the fold table filled, or a
// function name was too long to key on.
type capturePCSampling struct {
	Period        uint32 `json:"period"`
	HardwareBytes uint64 `json:"hardware_bytes"`
	Distinct      uint32 `json:"distinct"`
	Written       uint32 `json:"written"`
	Folded        uint64 `json:"folded"`
	Total         uint64 `json:"total"`
	HardwareLost  uint64 `json:"hardware_dropped"`
	Unkeyed       uint64 `json:"unkeyed"`
	Overflows     uint64 `json:"buffer_overflows"`
}

// captureModule is one module image the agent saw the driver load. File names
// the image beside the journal and is empty when Stored is false, which is the
// state a module too large for the agent's budget is recorded in: a module the
// capture declined to keep and a module the target never loaded are different
// facts, and only one of them is silence.
//
// Unchecked marks a load the agent wrote without looking for an earlier copy,
// because its table of seen modules was already full. Such a record may
// repeat one that came before it.
type captureModule struct {
	ID        uint32 `json:"module_id"`
	Size      uint64 `json:"size"`
	File      string `json:"file,omitempty"`
	Stored    bool   `json:"stored"`
	Reason    string `json:"reason,omitempty"`
	Unchecked bool   `json:"unchecked,omitempty"`
}

// errJournalNotFinalized reports a journal that decoded completely but whose
// producer never wrote a finalized record. Unlike every other observeJournal
// error it is returned together with the recovered values.
//
// A force-killed serving worker is one cause and was for a long time the only
// one documented here. It is not the common one. A Go target that simply
// returns from main exits through the runtime's own exit_group, which runs
// neither the ELF destructor nor atexit and sends itself no signal, so all
// three of the agent's finalization routes are bypassed and the journal ends
// unfinalized while the target exits 0 and reports its own work as done.
// Measured on GB10: a Go target collected zero records where a C++ target
// doing the same device work under the same agent and the same cudaprof
// collected cleanly.
//
// This matters to a reader because the two causes look identical here and
// point opposite ways. A force-kill says something went wrong upstream; a
// clean Go exit says nothing went wrong at all and the target simply cannot
// finalize this producer as written.
var errJournalNotFinalized = errors.New("capture journal missing finalized record")

// unfinalizedHint names the cause of an unfinalized journal when the evidence
// points at one, and says nothing when it does not.
//
// "capture journal missing finalized record" is true and unhelpful: it
// describes the artifact rather than what happened, and the two causes point
// opposite ways. A force-killed worker says something upstream went wrong. A
// target that exited 0 with a bootstrap on record and no activity at all
// says nothing went wrong -- it ran, it was instrumented, and it left through
// an exit path that reaches none of the agent's finalization routes.
//
// The hint is conditional on that signature because a caveat printed on every
// unfinalized journal is one a reader learns to skip, and then it is not a
// hint. A killed worker usually has activity records and a non-zero status, so
// it does not match and is not told a Go story about itself.
func unfinalizedHint(err error, result captureResult) string {
	if !errors.Is(err, errJournalNotFinalized) {
		return ""
	}
	if result.TargetExitCode != 0 || result.ActivityRecords != 0 {
		return ""
	}
	return "\n\tThe target exited 0 having written a bootstrap and no activity. That is what a\n" +
		"\truntime which leaves through the exit_group syscall looks like: the agent's ELF\n" +
		"\tdestructor and atexit both run from libc's exit(3), and its SIGTERM terminator\n" +
		"\tneeds a signal nobody sent, so none of the three fire and CUPTI's buffers are\n" +
		"\tdiscarded unflushed. A Go target does this by returning from main.\n" +
		"\tHave the target call cudaprofagent.Finalize() before it exits, or capture with\n" +
		"\t--finalize exit-group."
}

// Serving workers can legitimately emit more than the transport decoder's
// small default record bound. Keep an explicit finite controller bound while
// allowing representative framework bursts to be inspected.
const controllerMaxJournalRecords = timeline.MaxRecords

func observeJournal(path, session string, targetPID int, maxRecords int64) (wire.Stats, journalState, journalMetrics, []captureActivity, []captureAPI, []captureWorker, error) {
	state, metrics := journalState{}, journalMetrics{}
	raw, err := hex.DecodeString(session)
	if err != nil || len(raw) != 16 {
		return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("invalid capture session nonce")
	}
	var id [16]byte
	copy(id[:], raw)
	maxBound := int64(captureMaxRecords)
	if maxRecords > 0 {
		maxBound = maxRecords
	}
	limitsMaxRecords := uint32(maxBound)
	if maxBound > math.MaxUint32 {
		limitsMaxRecords = math.MaxUint32
	}
	recovered, err := journal.Recover(path, wire.Limits{MaxRecords: limitsMaxRecords})
	if err != nil {
		if errors.Is(err, wire.ErrRecordLimit) {
			// The count is what makes the remedy exact. It is taken only
			// here, once the journal has already been refused.
			n, cerr := countJournalRecords(path)
			if cerr != nil {
				return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, fmt.Errorf("capture journal exceeded limit of %d records (rerun with a larger --max-records): %w", maxBound, err)
			}
			return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, &journalRecordLimitError{Limit: maxBound, Records: n, Err: err}
		}
		return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, err
	}
	if recovered.Stats.Gaps != 0 {
		return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, fmt.Errorf("capture journal has %d missing sequence record(s)", recovered.Stats.Gaps)
	}
	if !recovered.Complete {
		return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("capture journal is truncated or incomplete")
	}
	activities := make([]captureActivity, 0)
	apis := make([]captureAPI, 0)
	workers := make(map[[2]uint64]*captureWorker)
	var activity, validActivity, generation, targetRecordCount uint64
	var terminal bool
	for _, record := range recovered.Records {
		state.records++
		if record.Session != id || record.Generation == 0 {
			return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, wire.ErrSession
		}
		if int(record.Process) != targetPID {
			key := [2]uint64{record.Process, record.Generation}
			worker := workers[key]
			if worker == nil {
				worker = &captureWorker{Process: record.Process, Generation: record.Generation, Disposition: "incomplete"}
				workers[key] = worker
			}
			worker.Records++
			if record.Kind == wire.KindActivity {
				worker.ActivityRecords++
			}
			var workerPayload captureRecordPayload
			if err := json.Unmarshal(record.Payload, &workerPayload); err != nil {
				worker.Failure = "invalid typed record"
				continue
			}
			switch record.Kind {
			case wire.KindFinalized:
				worker.Finalized = true
				if workerPayload.State == "producer_failure" {
					worker.Failure = workerPayload.State
				}
			case wire.KindFailure:
				worker.Failure = workerPayload.State
			}
			if worker.Finalized {
				if worker.Failure != "" {
					worker.Disposition = worker.Failure
				} else {
					worker.Disposition = "finalized"
				}
			}
			continue
		}
		if generation == 0 {
			generation = record.Generation
			state.generation = generation
		} else if generation != record.Generation {
			// A preload/exec chain can leave an incomplete generation for the
			// same PID before the actual target generation starts. Preserve that
			// stream as a worker disposition and qualify only the finalized
			// generation; never merge identities by PID alone.
			if state.finalized {
				return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("capture journal contains multiple finalized target generations")
			}
			worker := &captureWorker{Process: uint64(targetPID), Generation: generation, Records: targetRecordCount, ActivityRecords: activity, Finalized: state.finalized, Failure: state.failure, Disposition: "incomplete"}
			workers[[2]uint64{worker.Process, worker.Generation}] = worker
			generation = record.Generation
			state, metrics = journalState{}, journalMetrics{}
			state.generation = generation
			activities, apis = activities[:0], apis[:0]
			activity, validActivity, targetRecordCount, terminal = 0, 0, 0, false
		}
		targetRecordCount++
		var payload captureRecordPayload
		if len(record.Payload) > 0 && json.Unmarshal(record.Payload, &payload) != nil {
			return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("capture journal contains invalid typed record")
		}
		switch record.Kind {
		case wire.KindBootstrap:
			if terminal || state.bootstrap || payload.Kind != "bootstrap" {
				return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("invalid bootstrap ordering")
			}
			state.bootstrap = true
		case wire.KindCapability:
			if terminal || !state.bootstrap || state.capability || payload.Kind != "capability" {
				return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("invalid capability ordering")
			}
			state.capability = true
			state.capabilityData = append(json.RawMessage(nil), record.Payload...)
			state.agentRevision = agentRevision(record.Payload)
		case wire.KindActivity:
			if terminal || !state.bootstrap || !state.capability || payload.Kind != "activity" {
				return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("untyped or out-of-order activity")
			}
			// Markers and capture-range edges share this record stream but are
			// not device work: a marker is an NVTX range on the host, an edge
			// is the producer's own note of where a capture range opened or
			// closed. Neither can become a timeline entry, and neither is a
			// reason to reject the capture that carries it. CUPTI's MARKER
			// kind is enabled whenever the runtime supports it rather than on
			// request, so any target that emits NVTX -- PyTorch, cuBLAS and
			// NCCL all do -- produces marker records as a matter of course.
			//
			// The check stays a whitelist: an unrecognized subtype is still a
			// producer this consumer does not understand, and admitting it
			// silently would let a record of unknown shape through.
			//
			// Markers are carried into the activity slice even though they are
			// not device work, because BuildActivityProfile pairs them into
			// NVTX ranges and labels the device activity that ran inside them.
			// Dropping them here leaves that labelling permanently dead: the
			// ranges are paired from this very slice, so an empty slice yields
			// no ranges and no label, on every capture, with nothing failing.
			// They still become no sample -- BuildActivityProfile skips them
			// when it builds one -- and they are counted as neither activity
			// nor valid activity below, which are counts of device work.
			//
			// Capture-range edges are not carried: they are the producer's own
			// bookkeeping, ActivityRecord has no field for an edge, an api or
			// an index, so carrying one would keep a record that has lost
			// everything that made it meaningful.
			if payload.Subtype == "marker" {
				activities = append(activities, captureActivity{
					Subtype:      payload.Subtype,
					Name:         payload.Name,
					Valid:        payload.Valid,
					Start:        payload.Start,
					End:          payload.End,
					Device:       payload.Device,
					Context:      payload.Context,
					Stream:       payload.Stream,
					ProcessID:    payload.ProcessID,
					ThreadID:     payload.ThreadID,
					MarkerDomain: payload.MarkerDomain,
					MarkerID:     payload.MarkerID,
					MarkerFlags:  payload.MarkerFlags,
				})
				continue
			}
			if payload.Subtype == "capture_range" {
				metrics.phaseEdges = append(metrics.phaseEdges, capturePhaseEdge{
					Sequence:       record.Sequence,
					Timestamp:      record.Timestamp,
					Edge:           payload.Edge,
					API:            payload.API,
					Index:          payload.Index,
					ToggleFailures: payload.ToggleFailures,
					FlushNanos:     payload.FlushNanos,
				})
				continue
			}
			// device_active is the producer saying CUPTI asked it for a
			// buffer. Like a capture-range edge it is bookkeeping rather than
			// device work, so it is counted as neither activity nor valid
			// activity and carries nothing a timeline could hold.
			if payload.Subtype == "device_active" {
				state.deviceActive = true
				continue
			}
			// CUPTI overhead, the graph replay envelope and synchronization
			// windows are carried like markers are: they reach the consumer
			// intact, and they are counted as neither activity nor valid
			// activity, because those two are counts of device work and none
			// of these three is device work. An overhead interval is CUPTI's
			// own cost, a sync window is the host waiting, and a graph trace
			// spans the kernel records of the nodes it replayed -- counting
			// that one would charge the same device time twice.
			if payload.Subtype == "unified_memory" || payload.Subtype == "overhead" || payload.Subtype == "graph_trace" || payload.Subtype == "sync" {
				activities = append(activities, captureActivity{
					Subtype:      payload.Subtype,
					Valid:        payload.Valid && payload.Start != 0 && payload.End >= payload.Start && payload.End-payload.Start <= math.MaxInt64,
					Start:        payload.Start,
					End:          payload.End,
					Device:       payload.Device,
					Context:      payload.Context,
					Stream:       payload.Stream,
					Correlation:  payload.Correlation,
					ProcessID:    payload.ProcessID,
					ThreadID:     payload.ThreadID,
					GraphID:      payload.GraphID,
					OverheadKind: payload.OverheadKind,
					ObjectKind:   payload.ObjectKind,
					SyncType:     payload.SyncType,
					CudaEventID:  payload.CudaEventID,
				})
				continue
			}
			if payload.Subtype != "kernel" && payload.Subtype != "memcpy" && payload.Subtype != "memset" {
				return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("unknown activity subtype")
			}
			valid := payload.Valid && payload.Start != 0 && payload.End >= payload.Start && payload.End-payload.Start <= math.MaxInt64
			activity++
			if valid {
				validActivity++
			}
			grid := payload.GridDim
			if payload.GridDimRaw != nil {
				grid = *payload.GridDimRaw
			}
			if grid == [3]uint32{0, 0, 0} && (payload.GridX != 0 || payload.GridY != 0 || payload.GridZ != 0) {
				grid = [3]uint32{payload.GridX, payload.GridY, payload.GridZ}
			}
			block := payload.BlockDim
			if payload.BlockDimRaw != nil {
				block = *payload.BlockDimRaw
			}
			if block == [3]uint32{0, 0, 0} && (payload.BlockX != 0 || payload.BlockY != 0 || payload.BlockZ != 0) {
				block = [3]uint32{payload.BlockX, payload.BlockY, payload.BlockZ}
			}
			regs := payload.RegistersPerThread
			if regs == 0 && payload.Registers != 0 {
				regs = payload.Registers
			}
			smem := payload.DynamicSharedMemory
			if smem == 0 && payload.Smem != 0 {
				smem = payload.Smem
			}
			activities = append(activities, captureActivity{
				Subtype:             payload.Subtype,
				Name:                payload.Name,
				Valid:               valid,
				Start:               payload.Start,
				End:                 payload.End,
				Bytes:               payload.Bytes,
				Queued:              payload.Queued,
				Submitted:           payload.Submitted,
				CopyKind:            payload.CopyKind,
				Value:               payload.Value,
				Device:              payload.Device,
				Context:             payload.Context,
				Stream:              payload.Stream,
				Correlation:         payload.Correlation,
				RuntimeCorrelation:  payload.RuntimeCorrelation,
				GridDim:             grid,
				BlockDim:            block,
				RegistersPerThread:  regs,
				DynamicSharedMemory: smem,
				GraphID:             payload.GraphID,
				GraphNodeID:         payload.GraphNodeID,
				ModuleID:            payload.ModuleID,
			})
			if !valid {
				continue
			}
			if payload.Subtype == "memcpy" {
				if math.MaxUint64-metrics.copyBytes < payload.Bytes {
					return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("copy byte count overflows")
				}
				metrics.copyBytes += payload.Bytes
			} else {
				d := payload.End - payload.Start
				if math.MaxUint64-metrics.duration < d {
					return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("activity duration overflows")
				}
				metrics.duration += d
			}
		case wire.KindExternal:
			if terminal || !state.bootstrap || !state.capability || payload.Kind != "external_correlation" {
				return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("untyped or out-of-order external correlation")
			}
			metrics.external++
			// The payload, not only the count. An external-correlation record
			// is the only thing that says which framework scope a kernel ran
			// under, and the join that reads it (cuptiactivity.MatchExternal)
			// wants the links and the device work in one stream. Carrying it
			// in the activity slice is what gets it there: captureResult has
			// exactly one channel for per-record data, and a record with no
			// interval costs the profile nothing -- BuildActivityProfile skips
			// this subtype the way it skips a marker.
			//
			// Correlation is the native ID the link is keyed on; a record
			// without one links nothing, and keeping it would only add an
			// unjoinable row to every capture.
			if payload.Correlation != 0 {
				activities = append(activities, captureActivity{
					Subtype:      "external_correlation",
					Correlation:  payload.Correlation,
					ExternalKind: payload.ExternalKind,
					ExternalID:   payload.ExternalID,
				})
			}
		case wire.KindAPI:
			if terminal || !state.bootstrap || !state.capability || (payload.Kind != "api" && payload.Kind != "graph_lifecycle" && payload.Kind != "dependency" && payload.Kind != "memory_policy") {
				return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("untyped or out-of-order API record")
			}
			if payload.Kind == "memory_policy" {
				if payload.End < payload.Start || payload.End > math.MaxInt64 {
					return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("invalid memory policy timestamp")
				}
				continue
			}
			if payload.Kind == "dependency" {
				if payload.DependencyEvent == "" || payload.Ordinal == 0 || (payload.Site != "enter" && payload.Site != "exit") || (payload.Domain != "driver" && payload.Domain != "runtime") || payload.End < payload.Start || payload.End > math.MaxInt64 {
					return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("invalid dependency metadata")
				}
				continue
			}
			if payload.Kind == "graph_lifecycle" {
				// Lifecycle callbacks share the API wire kind, but are metadata.
				// Retain their raw journal records without adding timing calls:
				// driver observations can repeat an activity API correlation ID,
				// and resource callbacks have no ordinary API domain.
				if payload.LifecycleEvent == "" || (payload.Domain != "driver" && payload.Domain != "runtime" && payload.Domain != "resource") || payload.End < payload.Start || payload.End > math.MaxInt64 {
					return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("invalid graph lifecycle metadata")
				}
				continue
			}
			if payload.Domain != "driver" && payload.Domain != "runtime" {
				return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("unknown API domain")
			}
			apis = append(apis, captureAPI{Domain: payload.Domain, CallbackID: payload.CallbackID, Start: payload.Start, End: payload.End, ProcessID: payload.ProcessID, ThreadID: payload.ThreadID, Correlation: payload.Correlation, ReturnValue: payload.ReturnValue})
		case wire.KindGEMM:
			// cuBLASLt calls and the launches they made. The agent writes
			// them only while collecting, from the target's own threads, so
			// they are ordered like API records. They are read back from
			// the journal by the gemms command, not summarized here.
			if terminal || !state.bootstrap || !state.capability || (payload.Kind != "gemm" && payload.Kind != "gemm_launch") {
				return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("untyped or out-of-order gemm record")
			}
		case wire.KindHost:
			// Host run-state samples. They are not device work and never
			// enter the activity totals: this stream says what the host was
			// doing during a device gap, which is exactly the question the
			// activity stream cannot answer. The ordering check is the same
			// as every other ring-borne kind because the producer starts its
			// sampler only after the capability record is on disk.
			if terminal || !state.bootstrap || !state.capability || payload.Kind != "host_sample" {
				return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("untyped or out-of-order host sample")
			}
			metrics.hostSamples++
		case wire.KindModule:
			// Module images. Like host samples these are not device work and
			// never enter the activity totals; unlike them they are not gated
			// by the capture range, because a module is loaded when the target
			// builds its code and that is routinely before the range opens.
			// The ordering requirement is only that the producer has
			// identified itself.
			if terminal || !state.bootstrap || payload.Kind != "module" {
				return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("untyped or out-of-order module record")
			}
			metrics.modules = append(metrics.modules, captureModule{
				ID:        payload.ModuleID,
				Size:      payload.ModuleSize,
				File:      payload.ModuleFile,
				Stored:    payload.ModuleStored,
				Reason:    payload.ModuleReason,
				Unchecked: payload.ModuleUnchecked,
			})
		case wire.KindPCSample:
			// PC samples and the sampler's own summary, gated like module
			// records on nothing more than the producer having identified
			// itself: both are written at finalization, after the ring has
			// stopped taking device work.
			if terminal || !state.bootstrap {
				return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("out-of-order PC sample record")
			}
			switch payload.Kind {
			case "pcsample":
				metrics.pcSamples = append(metrics.pcSamples, capturePCSample{
					ModuleID: payload.ModuleID,
					Function: payload.PCFunction,
					Offset:   payload.PCOffset,
					Samples:  payload.PCSamples,
				})
			case "pcsampling":
				metrics.pcSampling = &capturePCSampling{
					Period:        payload.PCPeriod,
					HardwareBytes: payload.PCHardwareBytes,
					Distinct:      payload.PCDistinct,
					Written:       payload.PCWritten,
					Folded:        payload.PCFolded,
					Total:         payload.PCTotal,
					HardwareLost:  payload.PCHardwareLost,
					Unkeyed:       payload.PCUnkeyed,
					Overflows:     payload.PCOverflows,
				}
			default:
				return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("untyped PC sample record")
			}
		case wire.KindFinalized:
			if terminal || !state.bootstrap || !state.capability {
				return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("invalid finalized ordering")
			}
			terminal, state.finalized = true, true
			if payload.Dropped > recovered.Stats.Dropped {
				recovered.Stats.Dropped = payload.Dropped
			}
			metrics.loss = captureLoss{
				byKind:          payload.ByKind,
				Native:          payload.NativeDropped,
				Queue:           payload.QueueDropped,
				QueueOversize:   payload.QueueOversize,
				QueueContention: payload.QueueContention,
				QueueCapacity:   payload.QueueCapacity,
				Offered:         payload.Offered,
				Retained:        payload.Retained,

				BackpressureMaxNanos: payload.BackpressureMaxNanos,
				DrainWriteMaxNanos:   payload.DrainWriteMaxNanos,
			}
			switch payload.State {
			case "no_cuda_observed":
				state.noCUDA = true
			case "initialized_no_activity":
				state.initializedNoActivity = true
			case "collected":
				if validActivity == 0 {
					return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("collected finalization without valid activity")
				}
			case "producer_failure":
				if state.failure == "" {
					state.failure = payload.State
				}
			default:
				return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("unknown finalized state")
			}
		case wire.KindFailure:
			if terminal {
				return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("failure after finalized record")
			}
			state.failure = payload.State
			if state.failure == "" {
				state.failure = "producer_failure"
			}
			if payload.Error != "" {
				state.failureError = payload.Error
			} else if payload.Message != "" {
				if payload.Operation != "" {
					state.failureError = fmt.Sprintf("%s: %s", payload.Operation, payload.Message)
				} else {
					state.failureError = payload.Message
				}
			}
		default:
			return wire.Stats{}, state, journalMetrics{}, nil, nil, nil, errors.New("unknown record kind")
		}
	}
	recovered.Stats.Records = activity
	workerList := make([]captureWorker, 0, len(workers))
	for _, worker := range workers {
		workerList = append(workerList, *worker)
	}
	sort.Slice(workerList, func(i, j int) bool {
		if workerList[i].Process != workerList[j].Process {
			return workerList[i].Process < workerList[j].Process
		}
		return workerList[i].Generation < workerList[j].Generation
	})
	if !state.finalized {
		// Every structural check above passed, so the decoded records are
		// trustworthy; only the generation's own outcome is unknown. Return
		// them alongside the sentinel so a caller that explicitly asks for
		// diagnostic partial recovery can use them, and so every caller that
		// does not keeps treating this as a plain error.
		return recovered.Stats, state, metrics, activities, apis, workerList, errJournalNotFinalized
	}
	return recovered.Stats, state, metrics, activities, apis, workerList, nil
}
