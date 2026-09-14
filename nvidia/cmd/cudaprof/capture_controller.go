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
	"strings"

	"github.com/tmc/lib/nvidia/internal/cudainject/journal"
	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
)

type captureHandshake struct {
	Protocol         int    `json:"protocol"`
	Session          string `json:"session"`
	Process          uint64 `json:"process"`
	Generation       uint64 `json:"generation"`
	TransportReady   bool   `json:"transport_ready"`
	InjectionFailure string `json:"injection_failure,omitempty"`
}

const (
	recordBootstrap  uint16 = 1
	recordActivity   uint16 = 2
	recordFinalized  uint16 = 3
	recordCapability uint16 = 4
	recordFailure    uint16 = 5
	recordExternal   uint16 = 6
	recordAPI        uint16 = 7
)

type captureRecordPayload struct {
	State               string     `json:"state"`
	Kind                string     `json:"kind"`
	Subtype             string     `json:"subtype"`
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
	Device              uint32     `json:"device"`
	Context             uint32     `json:"context"`
	Stream              uint32     `json:"stream"`
	Correlation         uint32     `json:"correlation"`
	RuntimeCorrelation  uint32     `json:"runtime_correlation"`
	ExternalKind        uint32     `json:"external_kind"`
	ExternalID          uint64     `json:"external_id"`
	Domain              string     `json:"domain"`
	CallbackID          uint32     `json:"callback_id"`
	ProcessID           uint32     `json:"process_id"`
	ThreadID            uint32     `json:"thread_id"`
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
}

// observeOptions carries the caller's explicit choices into journal
// observation. The zero value is the strict default: only generations that
// wrote a finalized record contribute anything.
type observeOptions struct {
	injectionRequested bool
	// recoverUnfinalized admits records from a worker generation that decoded
	// completely but never finalized. Such a generation is labelled
	// "unfinalized-partial", counted separately in RecoveredGenerations, and
	// forces the partial-recovered outcome; it never makes the capture
	// successful.
	recoverUnfinalized bool
	maxJournalRecords  int64
}

// observeCaptureSession observes one capture session and always stamps the
// whole-run completeness of what it observed, at every terminal path.
func observeCaptureSession(result captureResult, dir, session string, opts observeOptions) captureResult {
	return withRunCompleteness(observeSession(result, dir, session, opts))
}

// withRunCompleteness classifies the run from evidence the capture already
// retains. It never modifies CaptureSuccess: the two answer different
// questions, and collapsing them is what let a target-qualified boolean stand
// for a run whose worker generations failed.
func withRunCompleteness(result captureResult) captureResult {
	result.UnqualifiedGenerations = unqualifiedGenerations(result.WorkerGenerations)
	result.RunCompleteness = runCompletenessOf(result.TransportReady, result.CollectorFinalized, result.UnqualifiedGenerations)
	return result
}

// runCompletenessOf is the single classification rule, shared by the writer and
// by inspect so a reader can never report a different answer than the capture.
func runCompletenessOf(transportReady, collectorFinalized bool, unqualified uint64) string {
	switch {
	case !transportReady:
		return runUnknown
	case !collectorFinalized:
		return runFailed
	case unqualified > 0:
		return runPartial
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
		return result
	}
	var selected *handshakeFile
	var selectedState journalState
	var selectedMetrics journalMetrics
	var selectedRecords wire.Stats
	var selectedActivities []captureActivity
	var selectedAPIs []captureAPI
	var workerActivities []captureActivity
	var workerAPIs []captureAPI
	var workerDuration, workerCopyBytes, workerExternal, workerDropped, workerRecords uint64
	workers := make([]captureWorker, 0)
	var targetHandshakeSeen bool
	var targetInjectionFailure string
	var targetFailureState journalState
	var targetJournalErr error
	for i := range files {
		f := &files[i]
		if f.h.Process != uint64(result.TargetPID) {
			worker := captureWorker{Process: f.h.Process, Generation: f.h.Generation, Disposition: "incomplete"}
			if f.h.InjectionFailure != "" {
				worker.Failure, worker.Disposition = f.h.InjectionFailure, "injection-failed"
			} else if records, state, metrics, activities, apis, _, err := observeJournal(filepath.Join(f.path, "capture.journal"), session, int(f.h.Process), opts.maxJournalRecords); err == nil || (opts.recoverUnfinalized && errors.Is(err, errJournalNotFinalized)) {
				// Identity is checked before anything is contributed, whether
				// or not the generation finalized.
				if state.generation != f.h.Generation {
					worker.Failure, worker.Disposition = "handshake generation mismatch", "identity-mismatch"
					workers = append(workers, worker)
					continue
				}
				worker.Records, worker.ActivityRecords, worker.Finalized = state.records, records.Records, state.finalized
				if state.finalized || err != nil {
					for i := range activities {
						activities[i].Process = f.h.Process
					}
					workerActivities = append(workerActivities, activities...)
					workerAPIs = append(workerAPIs, apis...)
					workerDuration += metrics.duration
					workerCopyBytes += metrics.copyBytes
					workerExternal += metrics.external
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
				}
			} else {
				worker.Failure = err.Error()
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
		if state.failure != "" {
			targetFailureState = state
		}
		if err != nil {
			targetJournalErr = err
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
		workers = append(workers, innerWorkers...)
		selected, selectedState, selectedMetrics, selectedRecords = f, state, metrics, records
		selectedActivities, selectedAPIs = activities, apis
	}
	result.WorkerGenerations = workers
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
			result.Outcome, result.CaptureError = captureOutcomeTransport, targetJournalErr.Error()
		} else if len(workers) > 0 && workers[len(workers)-1].Failure != "" {
			w := workers[len(workers)-1]
			if w.Disposition == "injection-failed" {
				result.Outcome, result.CaptureError = captureOutcomeInjection, w.Failure
			} else {
				result.Outcome, result.CaptureError = captureOutcomeTransport, w.Failure
			}
		} else if !targetHandshakeSeen {
			if !opts.injectionRequested {
				result.Outcome, result.CaptureError = captureOutcomeNoCUDA, "no injection library configured (specify --inject /path/to/agent.so or CUDAPROF_INJECT)"
			} else {
				result.Outcome, result.CaptureError = captureOutcomeTransport, "no identity-matching target handshake"
			}
		} else {
			result.Outcome, result.CaptureError = captureOutcomeTransport, "target journal incomplete"
		}
		return applyPartialRecovery(result)
	}
	result.TransportReady = true
	result.BootstrapSeen, result.CollectorFinalized = selectedState.bootstrap, selectedState.finalized
	result.ActivityRecords, result.DroppedRecords = selectedRecords.Records, selectedRecords.Dropped
	result.ExternalRecords = selectedMetrics.external
	result.ActivityNanos, result.CopyBytes, result.Activities = selectedMetrics.duration, selectedMetrics.copyBytes, selectedActivities
	result.APIRecords = selectedAPIs
	if len(workerActivities) != 0 {
		result.Activities = append(result.Activities, workerActivities...)
		result.ActivityNanos += workerDuration
		result.CopyBytes += workerCopyBytes
		result.ActivityRecords += workerRecords
		result.DroppedRecords += workerDropped
		result.ExternalRecords += workerExternal
		result.APIRecords = append(result.APIRecords, workerAPIs...)
	}
	for i := range result.Activities {
		if result.Activities[i].Process == 0 {
			result.Activities[i].Process = uint64(result.TargetPID)
		}
	}
	result.Capability = selectedState.capabilityData
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

// applyPartialRecovery marks a result that includes recovered unfinalized
// records. It never sets CaptureSuccess and never leaves a collected outcome
// standing, so no consumer can read the totals as a complete profile.
func applyPartialRecovery(result captureResult) captureResult {
	if result.RecoveredGenerations == 0 {
		return result
	}
	result.Outcome, result.CaptureSuccess = captureOutcomePartialRecovered, false
	if result.CaptureError == "" {
		result.CaptureError = fmt.Sprintf("%d worker generation(s) recovered without a finalized record", result.RecoveredGenerations)
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
	generation                                                      uint64
	records                                                         uint64
	capabilityData                                                  json.RawMessage
	failure                                                         string
	failureError                                                    string
}
type journalMetrics struct{ duration, copyBytes, external uint64 }

// errJournalNotFinalized reports a journal that decoded completely but whose
// producer never wrote a finalized record, as happens when a serving framework
// force-kills a worker. Unlike every other observeJournal error it is returned
// together with the recovered values.
var errJournalNotFinalized = errors.New("capture journal missing finalized record")

// Serving workers can legitimately emit more than the transport decoder's
// small default record bound. Keep an explicit finite controller bound while
// allowing representative framework bursts to be inspected.
const controllerMaxJournalRecords = 1 << 20

func observeJournal(path, session string, targetPID int, maxRecords int64) (wire.Stats, journalState, journalMetrics, []captureActivity, []captureAPI, []captureWorker, error) {
	raw, err := hex.DecodeString(session)
	if err != nil || len(raw) != 16 {
		return wire.Stats{}, journalState{}, journalMetrics{}, nil, nil, nil, errors.New("invalid capture session nonce")
	}
	var id [16]byte
	copy(id[:], raw)
	maxBound := int64(controllerMaxJournalRecords)
	if maxRecords > 0 {
		maxBound = maxRecords
	}
	limitsMaxRecords := uint32(maxBound)
	if maxBound > math.MaxUint32 {
		limitsMaxRecords = math.MaxUint32
	}
	recovered, err := journal.Recover(path, wire.Limits{MaxRecords: limitsMaxRecords})
	if err != nil {
		if strings.Contains(err.Error(), "record count exceeds limit") {
			return wire.Stats{}, journalState{}, journalMetrics{}, nil, nil, nil, fmt.Errorf("capture journal exceeded limit of %d records (re-run with --max-records=N or reduce workload size): %w", maxBound, err)
		}
		return wire.Stats{}, journalState{}, journalMetrics{}, nil, nil, nil, err
	}
	if !recovered.Complete {
		return wire.Stats{}, journalState{}, journalMetrics{}, nil, nil, nil, errors.New("capture journal is truncated or incomplete")
	}
	state, metrics := journalState{}, journalMetrics{}
	activities := make([]captureActivity, 0)
	apis := make([]captureAPI, 0)
	workers := make(map[[2]uint64]*captureWorker)
	var activity, validActivity, generation, targetRecordCount uint64
	var terminal bool
	for _, record := range recovered.Records {
		state.records++
		if record.Session != id || record.Generation == 0 {
			return wire.Stats{}, journalState{}, journalMetrics{}, nil, nil, nil, wire.ErrSession
		}
		if int(record.Process) != targetPID {
			key := [2]uint64{record.Process, record.Generation}
			worker := workers[key]
			if worker == nil {
				worker = &captureWorker{Process: record.Process, Generation: record.Generation, Disposition: "incomplete"}
				workers[key] = worker
			}
			worker.Records++
			if record.Kind == recordActivity {
				worker.ActivityRecords++
			}
			var workerPayload captureRecordPayload
			if err := json.Unmarshal(record.Payload, &workerPayload); err != nil {
				worker.Failure = "invalid typed record"
				continue
			}
			switch record.Kind {
			case recordFinalized:
				worker.Finalized = true
				if workerPayload.State == "producer_failure" {
					worker.Failure = workerPayload.State
				}
			case recordFailure:
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
				return wire.Stats{}, journalState{}, journalMetrics{}, nil, nil, nil, errors.New("capture journal contains multiple finalized target generations")
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
			return wire.Stats{}, journalState{}, journalMetrics{}, nil, nil, nil, errors.New("capture journal contains invalid typed record")
		}
		switch record.Kind {
		case recordBootstrap:
			if terminal || state.bootstrap || payload.Kind != "bootstrap" {
				return wire.Stats{}, journalState{}, journalMetrics{}, nil, nil, nil, errors.New("invalid bootstrap ordering")
			}
			state.bootstrap = true
		case recordCapability:
			if terminal || !state.bootstrap || state.capability || payload.Kind != "capability" {
				return wire.Stats{}, journalState{}, journalMetrics{}, nil, nil, nil, errors.New("invalid capability ordering")
			}
			state.capability = true
			state.capabilityData = append(json.RawMessage(nil), record.Payload...)
		case recordActivity:
			if terminal || !state.bootstrap || !state.capability || payload.Kind != "activity" {
				return wire.Stats{}, journalState{}, journalMetrics{}, nil, nil, nil, errors.New("untyped or out-of-order activity")
			}
			if payload.Subtype != "kernel" && payload.Subtype != "memcpy" {
				return wire.Stats{}, journalState{}, journalMetrics{}, nil, nil, nil, errors.New("unknown activity subtype")
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
				Device:              payload.Device,
				Context:             payload.Context,
				Stream:              payload.Stream,
				Correlation:         payload.Correlation,
				RuntimeCorrelation:  payload.RuntimeCorrelation,
				GridDim:             grid,
				BlockDim:            block,
				RegistersPerThread:  regs,
				DynamicSharedMemory: smem,
			})
			if !valid {
				continue
			}
			if payload.Subtype == "memcpy" {
				if math.MaxUint64-metrics.copyBytes < payload.Bytes {
					return wire.Stats{}, journalState{}, journalMetrics{}, nil, nil, nil, errors.New("copy byte count overflows")
				}
				metrics.copyBytes += payload.Bytes
			} else {
				d := payload.End - payload.Start
				if math.MaxUint64-metrics.duration < d {
					return wire.Stats{}, journalState{}, journalMetrics{}, nil, nil, nil, errors.New("activity duration overflows")
				}
				metrics.duration += d
			}
		case recordExternal:
			if terminal || !state.bootstrap || !state.capability || payload.Kind != "external_correlation" {
				return wire.Stats{}, journalState{}, journalMetrics{}, nil, nil, nil, errors.New("untyped or out-of-order external correlation")
			}
			metrics.external++
		case recordAPI:
			if terminal || !state.bootstrap || !state.capability || payload.Kind != "api" {
				return wire.Stats{}, journalState{}, journalMetrics{}, nil, nil, nil, errors.New("untyped or out-of-order API record")
			}
			if payload.Domain != "driver" && payload.Domain != "runtime" {
				return wire.Stats{}, journalState{}, journalMetrics{}, nil, nil, nil, errors.New("unknown API domain")
			}
			apis = append(apis, captureAPI{Domain: payload.Domain, CallbackID: payload.CallbackID, Start: payload.Start, End: payload.End, ProcessID: payload.ProcessID, ThreadID: payload.ThreadID, Correlation: payload.Correlation, ReturnValue: payload.ReturnValue})
		case recordFinalized:
			if terminal || !state.bootstrap || !state.capability {
				return wire.Stats{}, journalState{}, journalMetrics{}, nil, nil, nil, errors.New("invalid finalized ordering")
			}
			terminal, state.finalized = true, true
			switch payload.State {
			case "no_cuda_observed":
				state.noCUDA = true
			case "initialized_no_activity":
				state.initializedNoActivity = true
			case "collected":
				if validActivity == 0 {
					return wire.Stats{}, journalState{}, journalMetrics{}, nil, nil, nil, errors.New("collected finalization without valid activity")
				}
			case "producer_failure":
				state.failure = payload.State
			default:
				return wire.Stats{}, journalState{}, journalMetrics{}, nil, nil, nil, errors.New("unknown finalized state")
			}
		case recordFailure:
			if terminal {
				return wire.Stats{}, journalState{}, journalMetrics{}, nil, nil, nil, errors.New("failure after finalized record")
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
			return wire.Stats{}, journalState{}, journalMetrics{}, nil, nil, nil, errors.New("unknown record kind")
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
