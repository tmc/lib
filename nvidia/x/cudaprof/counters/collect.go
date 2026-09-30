package counters

import (
	"errors"
	"fmt"
	"reflect"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaprof/shape"
	"github.com/tmc/lib/nvidia/x/cuptirange"
	"github.com/tmc/lib/nvidia/x/cuptisass"
)

// CollectOptions names one qualified historical launch and bounds fresh replay.
// Arm is base or target. LockDeclared records the caller's existing GPU lock;
// it is not proof that unrelated processes obey that lock. ToolVersion must
// identify the application/collector build, for example its executable SHA-256.
type CollectOptions struct {
	Selection            Selection
	Arm                  string
	CaptureSHA256        string
	ExpectedOutputSHA256 string
	ToolVersion          string
	Metrics              []string
	MaxPasses            uint
	LockDeclared         bool
}

// Witness is supplied by the application after a replay and output check.
// Identity must describe the work actually submitted, and OutputSHA256 must
// hash its completed output. This application witness is separate from native
// symbol/geometry observations; neither is inferred from the historical label.
type Witness struct {
	Pass         uint           `json:"pass"`
	Identity     shape.Identity `json:"application_identity"`
	OutputSHA256 string         `json:"completed_output_sha256"`
}

// Replay restores one workload before each pass and submits one direct kernel.
// Run must join the work and hash completed output before returning its witness.
// Prepare and Synchronize run with the retained CUDA context current. Prepare
// is outside profiling. Kernel graphs, extended and cooperative launches lack
// the direct geometry witness required by Collect and are refused.
type Replay struct {
	Prepare     func(uint) error
	Run         func(uint) (Witness, error)
	Synchronize func() error
}

// Collection retains native progress, application witnesses and every metric,
// including failures. Counter values are not an ordinary timing baseline.
type Collection struct {
	DiagnosticTrace        *cuptirange.LaunchTrace       `json:"diagnostic_launch_trace,omitempty"`
	DiagnosticCounterBytes int                           `json:"diagnostic_counter_bytes,omitempty"`
	NativeFailureCode      *int32                        `json:"native_failure_code,omitempty"`
	Version                int                           `json:"version"`
	Arm                    string                        `json:"arm"`
	LockDeclared           bool                          `json:"gpu_lock_declared"`
	Selection              Selection                     `json:"historical_selection"`
	ConfigurationPasses    uint                          `json:"configuration_passes"`
	Quality                *cuptirange.CollectionQuality `json:"native_quality,omitempty"`
	Ranges                 []cuptirange.RangeInfo        `json:"native_ranges,omitempty"`
	Witnesses              []Witness                     `json:"application_replay_witnesses"`
	Receipts               []Receipt                     `json:"metrics"`
	Failure                string                        `json:"failure,omitempty"`
	Limits                 string                        `json:"limits"`
}

// Collect profiles one selected direct launch in the calling application's
// retained primary context. It queries definitions and counter availability,
// uses bounded user replay, checks native symbol/geometry on every pass, and
// compares application identity/output witnesses with the expected values.
// The original capture and workload's semantic equivalence remain caller-owned.
// A child process has a different context and cannot be profiled by this call.
func Collect(dev *cuda.Device, o CollectOptions, replay Replay) (result *Collection, err error) {
	result = &Collection{Version: 1, Arm: o.Arm, LockDeclared: o.LockDeclared, Selection: o.Selection, Witnesses: []Witness{}, Receipts: []Receipt{}, Limits: "historical selection is not replay; direct native launch symbol/geometry and application-supplied completed-output witnesses are checked separately; caller declares isolation; user replay changes execution and counter overhead is unmeasured; counter values do not establish a bottleneck or causal timing attribution"}
	defer func() {
		if err != nil {
			result.Failure = err.Error()
			var native cupti.Result
			if errors.As(err, &native) {
				code := int32(native)
				result.NativeFailureCode = &code
			}
			for i := range result.Receipts {
				result.Receipts[i].Value = nil
				result.Receipts[i].ReplayComplete = false
				result.Receipts[i].Failure = err.Error()
			}
		}
	}()
	if err = validateCollectOptions(o); err != nil {
		return result, err
	}
	if dev == nil || replay.Run == nil || replay.Prepare == nil || replay.Synchronize == nil {
		return result, errors.New("counters: device and complete replay callbacks are required")
	}
	launch := o.Selection.Base
	if o.Arm == "target" {
		launch = o.Selection.Target
	}
	for _, name := range o.Metrics {
		result.Receipts = append(result.Receipts, Receipt{Identity: o.Selection.Identity, CaptureSHA256: o.CaptureSHA256, Journal: launch.Journal, RawSymbol: launch.RawSymbol, ToolVersion: o.ToolVersion, Replay: "user replay; state restored outside profiling; one direct kernel; application identity and completed output verified", Properties: cuptirange.MetricProperties{Name: name}, Overhead: "unmeasured; collection is separate from ordinary timing"})
	}
	name, err := dev.Name()
	if err != nil {
		return result, err
	}
	uuid, err := dev.Raw().Uuid()
	if err != nil {
		return result, err
	}
	driverVersion, err := driver.DriverGetVersion()
	if err != nil {
		return result, err
	}
	for i := range result.Receipts {
		result.Receipts[i].Device = fmt.Sprintf("%x:%s", uuid.Bytes, name)
		result.Receipts[i].DriverVersion = fmt.Sprint(driverVersion)
	}
	var chip string
	var available []byte
	err = dev.WithContext(func() error {
		var e error
		chip, e = cuptisass.DeviceChipName(uint32(dev.Ordinal()))
		if e != nil {
			return e
		}
		available, e = cuptirange.CounterAvailability(dev, 16<<20)
		return e
	})
	for i := range result.Receipts {
		result.Receipts[i].Chip = chip
	}
	if err != nil {
		return result, err
	}
	plan, err := cuptirange.NewPlan(chip, available, o.Metrics, 16<<20)
	if plan != nil {
		defer func() { err = errors.Join(err, plan.Close()) }()
	}
	if err != nil {
		return result, err
	}
	_, result.ConfigurationPasses = plan.Configuration()
	for i, name := range o.Metrics {
		p, e := plan.Properties(name, 4096)
		if e != nil {
			return result, e
		}
		result.Receipts[i].Properties = p
	}
	capture, collectErr := plan.CaptureReplay(dev, cuptirange.Limits{MaxRanges: 1, MaxRangeTreeNodes: 1, MaxCounterBytes: 16 << 20, MaxPasses: o.MaxPasses}, cuptirange.CaptureOptions{Mode: cuptirange.Kernels}, cuptirange.Replay{Prepare: replay.Prepare, Synchronize: replay.Synchronize, Run: func(pass uint, _ *cuptirange.Scope) error {
		witness, e := replay.Run(pass)
		if e != nil {
			return e
		}
		witness.Pass = pass
		result.Witnesses = append(result.Witnesses, witness)
		return validateReplayWitness(o, witness)
	}})
	if capture != nil {
		result.Quality = &capture.Quality
		result.Ranges = capture.RangeInfo
		result.DiagnosticTrace = capture.DiagnosticTrace
		result.DiagnosticCounterBytes = len(capture.DiagnosticImage)
		defer func() { err = errors.Join(err, capture.Close()) }()
	}
	if collectErr != nil {
		return result, collectErr
	}
	if capture == nil || len(capture.RangeInfo) != 1 || capture.Ranges != 1 || len(result.Witnesses) == 0 || !capture.Quality.AllPassesSubmitted || !capture.Quality.Decoded || capture.Quality.DroppedRanges != 0 {
		return result, errors.New("counters: incomplete native replay")
	}
	if err := validateNativeRange(o, launch, capture.RangeInfo[0]); err != nil {
		return result, err
	}
	values, err := plan.Evaluate(capture.Image, 0)
	if err != nil {
		return result, err
	}
	for i, value := range values {
		value := value
		result.Receipts[i].Value = &value
		result.Receipts[i].ReplayComplete = true
	}
	return result, nil
}

func validateCollectOptions(o CollectOptions) error {
	if !o.LockDeclared {
		return errors.New("counters: an existing GPU lock declaration is required")
	}
	if o.Arm != "base" && o.Arm != "target" || !validSHA256(o.CaptureSHA256) || !validSHA256(o.ExpectedOutputSHA256) || o.ToolVersion == "" || o.MaxPasses == 0 || o.MaxPasses > 64 || len(o.Metrics) == 0 || len(o.Metrics) > 32 {
		return errors.New("counters: invalid collection options")
	}
	report := shape.Report{Version: 1, Pass: true, Groups: []shape.Group{{Identity: o.Selection.Identity, Base: []shape.Launch{o.Selection.Base}, Target: []shape.Launch{o.Selection.Target}}}}
	if _, err := Select(report, o.Selection.Identity); err != nil {
		return err
	}
	seen := make(map[string]bool)
	for _, name := range o.Metrics {
		if name == "" || seen[name] {
			return errors.New("counters: invalid metric selection")
		}
		seen[name] = true
	}
	return nil
}
func validateReplayWitness(o CollectOptions, w Witness) error {
	if !reflect.DeepEqual(w.Identity, o.Selection.Identity) || w.OutputSHA256 != o.ExpectedOutputSHA256 {
		return errors.New("counters: application replay identity or completed output changed")
	}
	return nil
}
func validateNativeRange(o CollectOptions, launch shape.Launch, r cuptirange.RangeInfo) error {
	if r.Kind != "kernel" || r.Name != launch.RawSymbol || !r.GeometryAvailable {
		return errors.New("counters: native replay symbol or direct geometry unavailable")
	}
	for i := range r.Grid {
		if int(r.Grid[i]) != o.Selection.Identity.Grid[i] || int(r.Block[i]) != o.Selection.Identity.Block[i] {
			return errors.New("counters: native replay launch geometry differs from selection")
		}
	}
	if shared := o.Selection.Identity.SharedMemory; shared != nil && r.SharedMemory != *shared {
		return errors.New("counters: native replay shared memory differs from selection")
	}
	return nil
}
