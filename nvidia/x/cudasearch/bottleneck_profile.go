package cudasearch

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/cuptibench"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

// BottleneckMetrics names the three native percentage metrics used by the
// classifier. Availability must be checked on the actual chip by cuptibench;
// unavailable metrics have no substitute. The names and achieved-occupancy
// mapping follow NVIDIA's CUPTI metric documentation:
// https://docs.nvidia.com/cuda/archive/11.0/cupti/Cupti/r_main.html.
func BottleneckMetrics() []string {
	return []string{"dram__throughput.avg.pct_of_peak_sustained_elapsed", "sm__throughput.avg.pct_of_peak_sustained_elapsed", "sm__warps_active.avg.pct_of_peak_sustained_active"}
}

// BottleneckProfile binds one whole-operation counter capture to its workload,
// input, artifact and compiler. State remains a heuristic category, even when
// its counters were observed on hardware. ID seals this exact in-process value;
// serialization does not retain HardwareObserved. Results are producer claims,
// so callers must retain and independently review the raw capture and outputs.
type BottleneckProfile struct {
	WorkloadKey, InputDigest, ArtifactDigest, CompilerKey string
	TargetArch, DeviceUUID, CaptureID, ID                 string
	LockPID                                               int
	Metrics                                               ProfileMetrics
	State                                                 BottleneckState
	seal                                                  string
}

func (p BottleneckProfile) digest() string {
	p.ID, p.seal = "", ""
	b, err := json.Marshal(p)
	if err != nil {
		return ""
	}
	return fmt.Sprintf("%x", sha256.Sum256(b))
}

// HardwareObserved reports whether this exact profile was admitted on a live
// device in this process. Integrity is not authentication of arbitrary counters.
func (p BottleneckProfile) HardwareObserved() bool {
	return p.seal != "" && p.ID == p.seal && p.digest() == p.seal
}

// CollectBottleneckProfile collects one whole operation with bounded CUPTI user
// replay under an inherited GPU lock. p supplies execution identities; w owns
// buffers, preparation and callbacks and must honor ctx. The raw result must be
// retained alongside artifacts and correctness outputs. A nil device returns a
// pending state and ErrUnqualifiedOutcome. Unavailable counters fail explicitly.
// Collection is separate from uninstrumented candidate timing and acknowledges
// warm caches; the resulting category describes that prepared execution only.
// A cleanup failure retains its plan in an error implementing io.Closer;
// callers must retry Close before destroying the device or workload resources.
func CollectBottleneckProfile(ctx context.Context, device *cuda.Device, p BottleneckProfile, w cuptibench.Workload) (BottleneckProfile, *cuptibench.Result, error) {
	p = pendingBottleneckProfile(p)
	if ctx == nil {
		return p, nil, fmt.Errorf("cudasearch: nil profile context")
	}
	if device == nil {
		p, err := observeBottleneckProfile(nil, p, nil)
		return p, nil, err
	}
	if err := ctx.Err(); err != nil {
		return p, nil, err
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		return p, nil, errors.Join(cudatrial.ErrUnqualifiedOutcome, err)
	}
	if err := validateBottleneckIdentity(p); err != nil {
		return p, nil, err
	}
	check := func() error {
		if err := ctx.Err(); err != nil {
			return err
		}
		_, err := cudalock.Inherited(os.Getenv)
		return err
	}
	if prepare := w.Prepare; prepare != nil {
		w.Prepare = func() error {
			if err := check(); err != nil {
				return err
			}
			return prepare()
		}
	}
	if run := w.Run; run != nil {
		w.Run = func(scope *cuptibench.Scope) error {
			if err := check(); err != nil {
				return err
			}
			return run(scope)
		}
	}
	plan, err := cuptibench.New(device, cuptibench.Config{Metrics: BottleneckMetrics(), Mode: cuptibench.Whole, Operations: 1, CachePolicy: cuptirange.CachePreserve, AllowWarmCacheTraffic: true, MaxConfigBytes: 8 << 20, Limits: cuptirange.Limits{MaxRanges: 2, MaxRangeTreeNodes: 2, MaxCounterBytes: 16 << 20, MaxPasses: 32}})
	if err != nil {
		return p, nil, closeBottleneckPlan(plan, err)
	}
	raw, err := plan.Collect(w)
	err = closeBottleneckPlan(plan, errors.Join(err, ctx.Err()))
	if err != nil {
		return p, raw, err
	}
	p, err = observeBottleneckProfile(device, p, raw)
	return p, raw, err
}

type bottleneckCleanupError struct {
	err  error
	plan *cuptibench.Plan
}

func (e *bottleneckCleanupError) Error() string { return e.err.Error() }
func (e *bottleneckCleanupError) Unwrap() error { return e.err }
func (e *bottleneckCleanupError) Close() error  { return e.plan.Close() }

func closeBottleneckPlan(plan *cuptibench.Plan, err error) error {
	if plan == nil {
		return err
	}
	if cleanup := plan.Close(); cleanup != nil {
		return &bottleneckCleanupError{errors.Join(err, cleanup), plan}
	}
	return err
}

func pendingBottleneckProfile(p BottleneckProfile) BottleneckProfile {
	p.TargetArch, p.DeviceUUID, p.CaptureID, p.ID, p.seal = "", "", "", "", ""
	p.LockPID = 0
	p.Metrics = ProfileMetrics{}
	p.State = BottleneckState{PendingHardware: true, Reason: "profile hardware observation pending"}
	return p
}

func observeBottleneckProfile(device *cuda.Device, p BottleneckProfile, result *cuptibench.Result) (BottleneckProfile, error) {
	p = pendingBottleneckProfile(p)
	if device == nil {
		return p, cudatrial.ErrUnqualifiedOutcome
	}
	owner, err := cudalock.Inherited(os.Getenv)
	if err != nil {
		return p, errors.Join(cudatrial.ErrUnqualifiedOutcome, err)
	}
	metrics, err := admitBottleneckCapture(p, result)
	if err != nil {
		return p, err
	}
	err = device.WithContext(func() error {
		major, minor, err := device.ComputeCapability()
		if err != nil {
			return err
		}
		p.TargetArch = fmt.Sprintf("sm_%d%d", major, minor)
		uuid, err := device.Raw().Uuid()
		if err != nil {
			return err
		}
		p.DeviceUUID = fmt.Sprintf("%x", uuid.Bytes)
		// cuptibench currently retains the driver's struct formatting. Compare
		// that exact producer encoding, and publish canonical UUID bytes here.
		if p.DeviceUUID == "00000000000000000000000000000000" || result.Metadata["device_uuid"] != fmt.Sprintf("%x", uuid) {
			return fmt.Errorf("%w: counter device mismatch", cudatrial.ErrUnqualifiedOutcome)
		}
		return nil
	})
	if err != nil {
		return p, err
	}
	p.Metrics, p.LockPID, p.CaptureID = metrics, owner.PID, result.Metadata["capture_id"]
	p.State = ClassifyBottleneck(metrics)
	p.State.PendingHardware = false
	p.State.Reason = "hardware-observed counters; heuristic category"
	p.ID = p.digest()
	p.seal = p.ID
	return p, nil
}

// admitBottleneckCapture is shared by the actual constructor and host refusal
// controls. Passing it alone never establishes a hardware observation.
func admitBottleneckCapture(p BottleneckProfile, result *cuptibench.Result) (ProfileMetrics, error) {
	bad := func(s string) (ProfileMetrics, error) {
		return ProfileMetrics{}, fmt.Errorf("%w: %s", cudatrial.ErrUnqualifiedOutcome, s)
	}
	if err := validateBottleneckIdentity(p); err != nil {
		return ProfileMetrics{}, err
	}
	if result == nil || result.Diagnostics != nil || result.Aggregate != nil || result.Timing != nil || result.Operations != 1 || len(result.Ranges) != 1 || result.Metadata["mode"] != "whole" || result.Metadata["capture_id"] == "" || result.Metadata["preparation"] == "" || result.Metadata["cupti_version"] == "" || result.Metadata["cuda_driver_version"] == "" {
		return bad("counter capture is not an identified whole operation")
	}
	q := result.Quality
	if !q.AllPassesSubmitted || !q.Decoded || q.DroppedRanges != 0 || q.PassesStarted == 0 || q.PassesStarted != q.PassesStopped {
		return bad("incomplete or lossy counter capture")
	}
	r := result.Ranges[0]
	if r.Kind != "operation" || r.ParentID != "" || !r.Inclusive || len(r.Metrics) != 3 {
		return bad("ambiguous counter range")
	}
	var m ProfileMetrics
	fields := []**float64{&m.MemoryThroughput, &m.ComputeThroughput, &m.WarpOccupancy}
	for i, name := range BottleneckMetrics() {
		v := r.Metrics[i]
		if v.Name != name || v.Unit != "percent" || v.Additive {
			return bad("counter metric or unit mismatch")
		}
		x := v.Value
		*fields[i] = &x
	}
	if !ClassifyBottleneck(m).StateName.Valid() {
		return bad("invalid hardware percentage")
	}
	return m, nil
}

func validateBottleneckIdentity(p BottleneckProfile) error {
	for _, s := range []string{p.WorkloadKey, p.InputDigest, p.ArtifactDigest, p.CompilerKey} {
		b, err := hex.DecodeString(s)
		if err != nil || len(b) != 32 || hex.EncodeToString(b) != s {
			return fmt.Errorf("%w: invalid counter execution identity", cudatrial.ErrUnqualifiedOutcome)
		}
	}
	return nil
}
