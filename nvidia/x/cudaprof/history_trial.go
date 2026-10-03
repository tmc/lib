package cudaprof

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudaprofilehistory"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

// HistoryTrialQualification is explicit producer qualification for trial
// timings. Native is not inferred from Complete or from a nonzero duration.
type HistoryTrialQualification struct {
	Native   bool
	Basis    string
	Producer string
	Locator  string
	Samples  int
	Loss     *cudaevidence.Loss
}

// HistoryTrial records an actual trial receipt against caller-supplied source,
// artifact, workload, input and plan provenance. Correctness is supplied
// independently. Only explicit Timings are retained; legacy duration fields
// never provide a performance fallback. Failed attempts remain observations.
// A supplied correctness result requires Receipt.Digest to be the same SHA-256
// output digest; legacy or absent digest schemes can be retained as Unknown.
// Receipt encoding is capped at 1 MiB, including conservative JSON escaping.
func HistoryTrial(base cudaprofilehistory.Record, trial cudatrial.Receipt, correctness cudaprofilehistory.Correctness, q HistoryTrialQualification) (cudaprofilehistory.Record, error) {
	base.Key = ""
	base.Correctness = correctness
	if correctness.Verdict != cudaprofilehistory.Unknown && trial.Digest != correctness.ActualDigest {
		return base, fmt.Errorf("history trial: correctness output differs from trial digest")
	}
	errText := ""
	if trial.Err != nil {
		errText = trial.Err.Error()
	}
	// Check caller-owned collections before encoding or copying their contents.
	remaining := 1<<20 - 4096
	charge := func(s string) bool {
		if len(s) > remaining/6 {
			return false
		}
		remaining -= 6*len(s) + 32
		return remaining >= 0
	}
	if len(trial.Provenance) > 4096 || len(trial.Timings) > 4096 || len(trial.Before.Values) > 4096 || len(trial.After.Values) > 4096 {
		return base, fmt.Errorf("history trial: receipt collection budget exceeded")
	}
	for _, s := range []string{trial.Name, trial.Digest, string(trial.TimingKind), trial.Before.Device, trial.After.Device, errText} {
		if !charge(s) {
			return base, fmt.Errorf("history trial: receipt byte budget exceeded")
		}
	}
	for k, v := range trial.Provenance {
		if !charge(k) || !charge(v) {
			return base, fmt.Errorf("history trial: receipt byte budget exceeded")
		}
	}
	for k := range trial.Timings {
		if !charge(string(k)) {
			return base, fmt.Errorf("history trial: receipt byte budget exceeded")
		}
	}
	for k := range trial.Before.Values {
		if !charge(string(k)) {
			return base, fmt.Errorf("history trial: receipt byte budget exceeded")
		}
	}
	for k := range trial.After.Values {
		if !charge(string(k)) {
			return base, fmt.Errorf("history trial: receipt byte budget exceeded")
		}
	}
	retainedTrial := trial
	retainedTrial.Err = nil // Error text is retained separately; arbitrary error objects are not JSON data.
	packet, err := json.Marshal(struct {
		Receipt cudatrial.Receipt `json:"receipt"`
		Error   string            `json:"error"`
	}{retainedTrial, errText})
	if err != nil {
		return base, err
	}
	if len(packet) > 1<<20 {
		return base, fmt.Errorf("history trial: encoded receipt byte budget exceeded")
	}
	base.Receipt = packet
	h := sha256.Sum256(packet)
	base.ReceiptDigest = hex.EncodeToString(h[:])
	if base.Artifact == nil {
		base.Status = "compile_failed"
		base.Reason = "trial has no compiled image"
	} else if trial.Err != nil || !trial.Complete {
		base.Status = "execution_failed"
		base.Reason = "trial incomplete: " + errText
	} else if !q.Native {
		base.Status = "unqualified"
		base.Reason = "native timing qualification unavailable"
	} else {
		base.Status = "complete"
		base.Reason = ""
	}
	keys := make([]string, 0, len(trial.Timings))
	for k := range trial.Timings {
		keys = append(keys, string(k))
	}
	sort.Strings(keys)
	if base.Artifact == nil && len(keys) > 0 {
		return base, fmt.Errorf("history trial: missing image carries timings")
	}
	retained := base.Facts[:0:0]
	for _, f := range base.Facts {
		if f.Axis != "performance" {
			retained = append(retained, f)
		}
	}
	base.Facts = retained
	for _, k := range keys {
		switch cudatrial.TimingKind(k) {
		case cudatrial.TimingWallElapsed, cudatrial.TimingSummedKernel, cudatrial.TimingGPUBusy, cudatrial.TimingAPICall:
		default:
			return base, fmt.Errorf("history trial: unsupported explicit timing %q", k)
		}
		ns := trial.Timings[cudatrial.TimingKind(k)].Nanoseconds()
		if ns < 0 {
			return base, fmt.Errorf("history trial: negative explicit timing")
		}
		n := float64(ns)
		if n >= math.Exp2(63) || int64(n) != ns {
			v, err := cudaevidence.New(cudaevidence.Record{Reason: "int64 timing cannot be represented exactly as a numeric evidence value", Source: cudaevidence.Source{Identity: base.Identity, Producer: q.Producer, Locator: q.Locator}})
			if err != nil {
				return base, err
			}
			base.Facts = append(base.Facts, cudaprofilehistory.Fact{Axis: "performance", Name: k, Value: v})
			continue
		}
		v, err := cudaevidence.New(cudaevidence.Record{Class: cudaevidence.Measured, Number: &n, Unit: "ns", Basis: q.Basis, Source: cudaevidence.Source{Identity: base.Identity, Producer: q.Producer, Locator: q.Locator}, Loss: q.Loss})
		if err != nil {
			return base, err
		}
		base.Facts = append(base.Facts, cudaprofilehistory.Fact{Axis: "performance", Name: k, Value: v, Qualified: q.Native, Samples: q.Samples})
	}
	if err := base.Validate(); err != nil {
		return base, err
	}
	return base, nil
}
