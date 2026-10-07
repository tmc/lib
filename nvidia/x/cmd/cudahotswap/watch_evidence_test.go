package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"path/filepath"

	"github.com/tmc/lib/nvidia/internal/nativereceipt"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
)

// These are the three source bodies used by the real native watcher consumer.
func nativeWatchSource(index int) string {
	switch index {
	case 0:
		return "extern \"C\" __global__ void replay(float*x,float*y,const float*w,unsigned n){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=x[i]-w[i];}"
	case 1:
		return "extern \"C\" __global__ void replay(float*x){x[threadIdx.x]=0.0f;}"
	default:
		return "extern \"C\" __global__ void replay(float*x,float*y,const float*w,unsigned n){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=w[i]+x[i];}"
	}
}

func watchDigest(body []byte) string { return fmt.Sprintf("%x", sha256.Sum256(body)) }
func watchDigestValid(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == sha256.Size
}

type watchCompilation struct {
	Source                                    cudahotswap.Source
	SourceSHA256, PTXSHA256, CUBINSHA256, Key string
	PTXBytes, CUBINBytes                      int
}
type watchLoad struct {
	Compilation       int
	Function, Context uint64
	LoadError         string
}
type watchEvent struct {
	Kernel string
	Result *cudahotswap.PushResult
	Error  string `json:",omitempty"`
}
type watchObservedEvent struct {
	Event    watchEvent
	Selected cudahotswap.State
}
type watchEvidence struct {
	Version      int
	Process      nativereceipt.Process
	WatchPath    string
	Baseline     cudahotswap.State
	Compilations []watchCompilation
	Loads        []watchLoad
	Events       []watchObservedEvent
	Cleanup      struct {
		Attempted, Complete bool
		Error               string
		Attempts            []string
		ClosedService       bool
	}
	Qualification bool
}

func validateWatchEvent(index int, baseline cudahotswap.State, path string, compiled watchCompilation, observed watchObservedEvent) error {
	if index < 0 || index >= 3 || baseline.ConfigRevision == math.MaxUint64 {
		return errors.New("watch event: invalid index or revision")
	}
	event := observed.Event
	result := event.Result
	if result == nil || result.Report.Compile == nil || event.Kernel != "kernel" {
		return errors.New("watch event: missing result or compilation")
	}
	compile := result.Report.Compile
	if compile.SourceSHA256 != watchDigest([]byte(nativeWatchSource(index))) || compile.SourceSHA256 != compiled.SourceSHA256 || compile.SourceBytes != len(compiled.Source.Code) {
		return errors.New("watch event: compilation identity differs")
	}
	// Push returns before preparing a candidate on ABI refusal. The recorder
	// still retains the actual compiler artifact independently of this report.
	if index == 1 {
		if compile.PTXSHA256 != "" || compile.PTXBytes != 0 {
			return errors.New("watch event: ABI refusal reported prepared PTX")
		}
	} else if compile.PTXSHA256 != compiled.PTXSHA256 || compile.PTXBytes != compiled.PTXBytes || !watchDigestValid(compile.PTXSHA256) {
		return errors.New("watch event: compilation PTX identity differs")
	}
	want := []cudahotswap.Outcome{cudahotswap.OutcomeWrongAnswer, cudahotswap.OutcomeABIMismatch, cudahotswap.OutcomeOK}[index]
	if result.Report.Outcome != want || (event.Error != "") != (index < 2) || result.Published != (index == 2) || result.RolledBack {
		return errors.New("watch event: outcome differs")
	}
	if result.Selected != observed.Selected || result.Report.Generation != result.State.Generation || result.Report.ConfigRevision != result.State.ConfigRevision {
		return errors.New("watch event: observed state differs")
	}
	expected := baseline
	expected.ConfigRevision++
	expected.Label = "local watch"
	if index < 2 {
		if result.Selected != expected {
			return errors.New("watch event: refusal changed selection")
		}
		expected.Label = "watch: " + path
		if result.State != expected {
			return errors.New("watch event: refused state differs")
		}
	} else {
		state := result.State
		if state != result.Selected || state.Name != "kernel" || state.Kind != "kernel" || state.Version <= baseline.Version || state.Generation <= baseline.Generation || state.ConfigRevision != expected.ConfigRevision || state.SourceDigest != compiled.SourceSHA256 || state.CompilerKey != compiled.Key || state.Label != "watch: "+path {
			return errors.New("watch event: repair selection differs")
		}
	}
	if index == 1 {
		if result.Report.Replay != nil {
			return errors.New("watch event: ABI refusal ran replay")
		}
	} else {
		r := result.Report.Replay
		if r == nil || !r.Native || r.Clock != "cuda_event_interval" || r.Passed != (index == 2) || len(r.Outputs) != 1 || r.Outputs[0].Arg != 1 || r.Outputs[0].Passed != (index == 2) || r.Outputs[0].Candidate.Elements != 32 || r.Outputs[0].Candidate.NonFinite != 0 {
			return errors.New("watch event: missing actual replay outcome")
		}
		want, got := watchExpectedOutputs(index)
		candidate, err := cudahotswap.MeasureReplay(got, want)
		if err != nil {
			return err
		}
		noise, err := cudahotswap.MeasureReplay(want, want)
		if err != nil {
			return err
		}
		out := r.Outputs[0]
		if !watchMetricsEqual(out.Candidate, candidate) || !watchMetricsEqual(out.Noise, noise) || !out.WriteOnly {
			return errors.New("watch event: source numerical or poisoned output comparison differs")
		}
		if err := validateWatchTrials(r, watchOutputDigest(want), watchOutputDigest(got)); err != nil {
			return err
		}

	}
	return nil
}

func validateWatchEvidence(e watchEvidence) error {
	p := e.Process
	if e.Version != 1 || e.Qualification || p.PID <= 1 || p.Parent <= 1 || p.Group <= 1 || p.Start == 0 || !filepath.IsAbs(p.Executable) || !watchDigestValid(p.ExecutableSHA256) || !filepath.IsAbs(e.WatchPath) || filepath.Clean(e.WatchPath) != e.WatchPath {
		return errors.New("watch evidence: process or schema identity")
	}
	if len(e.Compilations) != 4 || len(e.Loads) != 3 || len(e.Events) != 3 {
		return errors.New("watch evidence: incomplete inventory")
	}
	baselineCode := fmt.Sprintf("extern \"C\" __global__ void %s(float *x, float *y, const float *w, unsigned n) { unsigned i = blockIdx.x*blockDim.x+threadIdx.x; if (i<n) y[i]=x[i]+w[i]; }", "replay")
	for i, c := range e.Compilations {
		code := baselineCode
		if i > 0 {
			code = nativeWatchSource(i - 1)
		}
		if c.Source.Code != code || c.Source.Filename != e.WatchPath || c.Source.Entry != "replay" || len(c.Source.Flags) != 0 || c.SourceSHA256 != watchDigest([]byte(code)) || c.Key == "" || !watchDigestValid(c.PTXSHA256) || !watchDigestValid(c.CUBINSHA256) || c.PTXBytes <= 0 || c.PTXBytes > 8<<20 || c.CUBINBytes <= 0 || c.CUBINBytes > 64<<20 {
			return errors.New("watch evidence: compilation body binding")
		}
	}
	b := e.Baseline
	if b.Name != "kernel" || b.Kind != "kernel" || b.Version == 0 || b.Generation == 0 || b.ConfigRevision == 0 || b.SourceDigest != e.Compilations[0].SourceSHA256 || b.CompilerKey != e.Compilations[0].Key || b.Label != "replay check" {
		return errors.New("watch evidence: baseline binding")
	}
	for i, l := range e.Loads {
		if l.Compilation != []int{0, 1, 3}[i] || l.Function == 0 || l.Context == 0 || l.Context != e.Loads[0].Context || l.LoadError != "" {
			return errors.New("watch evidence: actual load or completion binding")
		}

	}
	for i, event := range e.Events {
		if err := validateWatchEvent(i, b, e.WatchPath, e.Compilations[i+1], event); err != nil {
			return err
		}
	}
	if !e.Cleanup.Attempted || !e.Cleanup.Complete || e.Cleanup.Error != "" || !e.Cleanup.ClosedService || len(e.Cleanup.Attempts) == 0 || e.Cleanup.Attempts[len(e.Cleanup.Attempts)-1] != "" {
		return errors.New("watch evidence: session cleanup incomplete")
	}
	return nil
}

// These expectations derive only from the real session's x=i+1, w=3*(i+1)
// initialization and the literal source operators, never substitute an observed
// native result. Both arithmetic expressions are exactly representable FP32.
func watchExpectedOutputs(index int) (want, got []float32) {
	for i := 1; i <= 32; i++ {
		x, w := float32(i), float32(3*i)
		want = append(want, x+w)
		if index == 0 {
			got = append(got, x-w)
		} else {
			got = append(got, w+x)
		}
	}
	return want, got
}

func watchMetricsEqual(a, b cudahotswap.ReplayMetrics) bool {
	return a.Elements == b.Elements && a.NonFinite == b.NonFinite && a.Exact == b.Exact &&
		a.MaxAbs != nil && b.MaxAbs != nil && *a.MaxAbs == *b.MaxAbs &&
		a.RMS != nil && b.RMS != nil && *a.RMS == *b.RMS &&
		a.MaxULP != nil && b.MaxULP != nil && *a.MaxULP == *b.MaxULP
}

func watchOutputDigest(v []float32) string {
	raw := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(x))
	}
	return watchDigest(raw)
}

func validateWatchTrials(r *cudahotswap.ReplayReport, baseline, candidate string) error {
	if len(r.Trials) != 10 {
		return errors.New("watch event: incomplete paired rounds")
	}
	wins, losses := 0, 0
	for round := 1; round <= 5; round++ {
		arms := []string{"baseline", "candidate"}
		if round%2 == 0 {
			arms[0], arms[1] = arms[1], arms[0]
		}
		var base, cand int64
		for j, arm := range arms {
			trial := r.Trials[(round-1)*2+j]
			digest := baseline
			if arm == "candidate" {
				digest = candidate
			}
			if trial.Round != round || trial.Arm != arm || !trial.Complete || trial.Error != "" || trial.DeviceNanos == nil || *trial.DeviceNanos <= 0 || trial.Digest != digest {
				return errors.New("watch event: paired native trial binding differs")
			}
			if arm == "baseline" {
				base = *trial.DeviceNanos
			} else {
				cand = *trial.DeviceNanos
			}
		}
		if cand < base {
			wins++
		}
		if cand > base {
			losses++
		}
	}
	n := wins + losses
	p := float64(1)
	if n != 0 {
		term := math.Ldexp(1, -n)
		sum := term
		for k := 1; k <= min(wins, losses); k++ {
			term *= float64(n-k+1) / float64(k)
			sum += term
		}
		p = min(1, 2*sum)
	}
	if r.CandidateWins != wins || r.BaselineWins != losses || r.SignP == nil || *r.SignP != p {
		return errors.New("watch event: paired timing statistics differ")
	}
	return nil
}

func validateWatchArtifacts(dir string, e watchEvidence) error {
	if len(e.Compilations) != 4 {
		return errors.New("watch artifacts: incomplete compilation inventory")
	}
	for i, c := range e.Compilations {
		for _, part := range []struct {
			suffix, hash string
			bytes, limit int
		}{
			{"cu", c.SourceSHA256, len(c.Source.Code), 8 << 20},
			{"ptx", c.PTXSHA256, c.PTXBytes, 8 << 20},
			{"cubin", c.CUBINSHA256, c.CUBINBytes, 64 << 20},
		} {
			if part.bytes <= 0 || part.bytes > part.limit || !watchDigestValid(part.hash) {
				return errors.New("watch artifacts: invalid metadata extent or digest")
			}
			body, err := watchReadArtifact(filepath.Join(dir, fmt.Sprintf("compile-%d.%s", i, part.suffix)), int64(part.limit))
			if err != nil {
				return fmt.Errorf("watch artifacts: %w", err)
			}
			if len(body) != part.bytes || watchDigest(body) != part.hash {
				return errors.New("watch artifacts: retained body differs")
			}
		}
	}
	return nil
}
