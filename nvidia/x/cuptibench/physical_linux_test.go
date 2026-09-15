//go:build linux

package cuptibench_test

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cuptibench"
	"github.com/tmc/lib/nvidia/x/cuptirange"
	"github.com/tmc/lib/nvidia/x/cuptisass"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// Opt-in qualification fixture, not a DS4 implementation or a traffic model.
// Run on an exclusively owned device with CUPTIBENCH_PHYSICAL_TEST=1 and retain
// go test -json output. An opted-in unavailable metric/backend fails the gate.
// The CTA metric counts launched blocks (CUPTI's range profiling tutorial uses
// this exact metric): https://docs.nvidia.com/cupti/tutorial/tutorial.html
const physicalMetric = "sm__ctas_launched.sum"
const physicalBlock = 32
const physicalCapacity = 8 * physicalBlock
const physicalSource = `
extern "C" __global__ void fixture_reset(unsigned *state, unsigned *mid, unsigned *out) {
 unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
 state[i] = i; mid[i] = 0; out[i] = 0;
}
extern "C" __global__ void fixture_project(unsigned *state, unsigned *mid) {
 unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
 mid[i] = state[i] + 3;
}
extern "C" __global__ void fixture_decode(unsigned *state, unsigned *mid, unsigned *out) {
 unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
 out[i] = 2 * mid[i]; state[i]++;
}
__device__ unsigned long long timer_ns() {
 unsigned long long t;
 asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
 return t;
}
extern "C" __global__ void fixture_wait(unsigned *out, unsigned index, unsigned value) {
 unsigned long long start = timer_ns();
 while (timer_ns() - start < 20000000ULL) {}
 if (threadIdx.x == 0) out[index] = value;
}
`

type physicalPipeline struct {
	dev                          *cuda.Device
	streams                      [2]*cuda.Stream
	projected, decoded           *cuda.Event
	state, mid, out              cudamem.Slice[uint32]
	reset, project, decode, wait driver.Function
	grid                         uint32
	prepares, runs               int
}

func physicalCheck(tb testing.TB, err error) {
	tb.Helper()
	if err != nil {
		tb.Fatal(err)
	}
}

func physicalClose(tb testing.TB, close func() error) {
	tb.Helper()
	tb.Cleanup(func() {
		if err := close(); err != nil {
			tb.Error(err)
		}
	})
}

func newPhysicalPipeline(tb testing.TB, grid uint32) *physicalPipeline {
	tb.Helper()
	if os.Getenv("CUPTIBENCH_PHYSICAL_TEST") != "1" {
		tb.Skip("unqualified: set CUPTIBENCH_PHYSICAL_TEST=1 on an exclusively owned CUDA device")
	}
	dev, err := cuda.NewDevice(0)
	physicalCheck(tb, err)
	physicalClose(tb, dev.Close)
	p := &physicalPipeline{dev: dev, grid: grid}
	for i := range p.streams {
		p.streams[i], err = dev.NewStreamWithFlags(uint32(driver.StreamNonBlocking))
		physicalCheck(tb, err)
		physicalClose(tb, p.streams[i].Destroy)
	}
	p.projected, err = dev.NewEventWithFlags(uint32(driver.EventDisableTiming))
	physicalCheck(tb, err)
	physicalClose(tb, p.projected.Destroy)
	p.decoded, err = dev.NewEventWithFlags(uint32(driver.EventDisableTiming))
	physicalCheck(tb, err)
	physicalClose(tb, p.decoded.Destroy)
	for _, slice := range []*cudamem.Slice[uint32]{&p.state, &p.mid, &p.out} {
		*slice, err = cudamem.Alloc[uint32](dev, physicalCapacity)
		physicalCheck(tb, err)
		physicalClose(tb, (*slice).Free)
	}
	cm, err := nvrtcx.Compile(physicalSource, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	physicalCheck(tb, err)
	module, err := cm.Load(dev)
	physicalCheck(tb, err)
	physicalClose(tb, func() error { return dev.WithContext(module.Unload) })
	physicalCheck(tb, dev.WithContext(func() error {
		for name, fn := range map[string]*driver.Function{"fixture_reset": &p.reset, "fixture_project": &p.project, "fixture_decode": &p.decode, "fixture_wait": &p.wait} {
			f, err := module.GetFunction(name)
			if err != nil {
				return err
			}
			*fn = f
		}
		return nil
	}))
	tb.Logf("physical fixture sha256=%x grid=%d block=%d cache=uncontrolled; synthetic dependency pipeline, not DS4", sha256.Sum256([]byte(physicalSource)), grid, physicalBlock)
	return p
}

func (p *physicalPipeline) synchronize() error {
	return errors.Join(p.streams[0].Synchronize(nil), p.streams[1].Synchronize(nil))
}
func (p *physicalPipeline) prepare() error {
	p.prepares++
	return cudalaunch.Kernel(p.reset).Grid(8).Block(physicalBlock).On(p.streams[0]).Launch(p.state, p.mid, p.out)
}
func (p *physicalPipeline) projectStage() error {
	if err := cudalaunch.Kernel(p.project).Grid(p.grid).Block(physicalBlock).On(p.streams[0]).Launch(p.state, p.mid); err != nil {
		return err
	}
	if err := p.projected.Record(p.streams[0]); err != nil {
		return err
	}
	return p.streams[1].WaitEvent(p.projected)
}
func (p *physicalPipeline) decodeStage() error {
	if err := cudalaunch.Kernel(p.decode).Grid(p.grid).Block(physicalBlock).On(p.streams[1]).Launch(p.state, p.mid, p.out); err != nil {
		return err
	}
	if err := p.decoded.Record(p.streams[1]); err != nil {
		return err
	}
	// The next logical operation reads state advanced by decode.
	return p.streams[0].WaitEvent(p.decoded)
}
func (p *physicalPipeline) workload(isolated string) cuptibench.Workload {
	w := cuptibench.Workload{Streams: p.streams[:], Prepare: p.prepare, Synchronize: p.synchronize, Preparation: "GPU reset to state[i]=i once per sequence; cache uncontrolled"}
	switch isolated {
	case "":
		w.View = "full"
		w.Run = func(s *cuptibench.Scope) error {
			p.runs++
			return s.Region("layer00", func() error {
				if err := s.Region("project", p.projectStage); err != nil {
					return err
				}
				return s.Region("decode", p.decodeStage)
			})
		}
	case "project":
		w.View = "isolated"
		w.Run = func(s *cuptibench.Scope) error { p.runs++; return s.Region("project", p.projectStage) }
	case "decode":
		w.View = "isolated"
		w.Prepare = func() error {
			if err := p.prepare(); err != nil {
				return err
			}
			return p.projectStage()
		}
		w.Preparation += "; unmeasured project prefix provides intermediate fixture"
		w.Run = func(s *cuptibench.Scope) error { p.runs++; return s.Region("decode", p.decodeStage) }
	default:
		panic("invalid test stage")
	}
	return w
}

func physicalPlan(tb testing.TB, p *physicalPipeline, mode cuptibench.Mode, operations int) *cuptibench.Plan {
	tb.Helper()
	plan, err := cuptibench.New(p.dev, cuptibench.Config{CachePolicy: cuptirange.CachePreserve, Metrics: []string{physicalMetric}, Mode: mode, Operations: operations, MaxConfigBytes: 16 << 20, Limits: cuptirange.Limits{MaxRanges: 128, MaxRangeTreeNodes: 256, MaxCounterBytes: 32 << 20, MaxPasses: 128}})
	if plan != nil {
		physicalClose(tb, plan.Close)
	}
	physicalCheck(tb, err)
	return plan
}

func physicalResult(tb testing.TB, p *physicalPipeline, plan *cuptibench.Plan, isolated string, operations int) *cuptibench.Result {
	tb.Helper()
	beforePrepare, beforeRun := p.prepares, p.runs
	r, err := plan.Collect(p.workload(isolated))
	physicalCheck(tb, err)
	if r == nil || r.Operations != operations || r.Timing != nil {
		tb.Fatalf("invalid collection result: %+v", r)
	}
	q := r.Quality
	if !q.Decoded || !q.AllPassesSubmitted || q.DroppedRanges != 0 || q.PassesStarted == 0 || q.PassesStarted != q.PassesStopped {
		tb.Fatalf("invalid collection quality: %+v", q)
	}
	if p.prepares-beforePrepare != int(q.PassesStarted) || p.runs-beforeRun != operations*int(q.PassesStarted) {
		tb.Fatalf("replay accounting prepare=%d run=%d operations=%d quality=%+v", p.prepares-beforePrepare, p.runs-beforeRun, operations, q)
	}
	for _, key := range []string{"device_uuid", "chip", "cuda_driver_version", "cupti_version", "capture_id", "preparation"} {
		if r.Metadata[key] == "" {
			tb.Fatalf("missing physical provenance %s", key)
		}
	}
	raw, err := json.Marshal(r)
	physicalCheck(tb, err)
	tb.Logf("physical result=%s", raw)
	return r
}

func checkPhysicalOutput(tb testing.TB, p *physicalPipeline, stage string, operations int) {
	tb.Helper()
	physicalCheck(tb, p.synchronize())
	state, mid, out := make([]uint32, physicalCapacity), make([]uint32, physicalCapacity), make([]uint32, physicalCapacity)
	physicalCheck(tb, p.state.CopyToHost(state))
	physicalCheck(tb, p.mid.CopyToHost(mid))
	physicalCheck(tb, p.out.CopyToHost(out))
	for i := 0; i < int(p.grid)*physicalBlock; i++ {
		wantState, wantMid, wantOut := uint32(i+operations), uint32(i+operations+2), uint32(2*(i+operations+2))
		if stage == "project" {
			wantState = uint32(i)
			wantMid = uint32(i + 3)
			wantOut = 0
		}
		if stage == "decode" {
			wantMid = uint32(i + 3)
			wantOut = 2 * wantMid
		}
		if state[i] != wantState || mid[i] != wantMid || out[i] != wantOut {
			tb.Fatalf("%s output[%d] state/mid/out=%d/%d/%d want=%d/%d/%d", stage, i, state[i], mid[i], out[i], wantState, wantMid, wantOut)
		}
	}
}

func physicalValue(tb testing.TB, r cuptibench.Range) float64 {
	tb.Helper()
	if len(r.Metrics) != 1 || r.Metrics[0].Name != physicalMetric {
		tb.Fatalf("wrong physical metrics: %+v", r)
	}
	return r.Metrics[0].Value
}

func TestPhysicalCuptibenchViews(t *testing.T) {
	p := newPhysicalPipeline(t, 4)
	for _, test := range []struct {
		name string
		mode cuptibench.Mode
		ops  int
	}{{"whole", cuptibench.Whole, 1}, {"whole-sequence", cuptibench.Whole, 3}, {"regions", cuptibench.Regions, 1}, {"kernels", cuptibench.Kernels, 3}} {
		t.Run(test.name, func(t *testing.T) {
			plan := physicalPlan(t, p, test.mode, test.ops)
			r := physicalResult(t, p, plan, "", test.ops)
			checkPhysicalOutput(t, p, "", test.ops)
			want := float64(2 * p.grid * uint32(test.ops))
			switch test.mode {
			case cuptibench.Whole:
				if len(r.Ranges) != 1 || r.Ranges[0].Kind != "operation" || physicalValue(t, r.Ranges[0]) != want {
					t.Fatalf("whole counter, reset exclusion or normalization failure: %+v want CTAs=%g", r.Ranges, want)
				}
			case cuptibench.Regions:
				if len(r.Ranges) != 4 {
					t.Fatalf("range tree=%+v", r.Ranges)
				}
				for _, rg := range r.Ranges {
					expected := want
					if rg.Name == "project" || rg.Name == "decode" {
						expected = float64(p.grid)
					}
					if rg.ID == "" || !rg.Inclusive || physicalValue(t, rg) != expected {
						t.Fatalf("region identity/count=%+v expected=%g", rg, expected)
					}
				}
			case cuptibench.Kernels:
				if len(r.Ranges) != 2*test.ops {
					t.Fatalf("kernel count=%d want=%d (preparation contamination)", len(r.Ranges), 2*test.ops)
				}
				for _, rg := range r.Ranges {
					if rg.Kind != "kernel" || rg.ID == "" || strings.Contains(rg.Name, "reset") || (!strings.Contains(rg.Name, "fixture_project") && !strings.Contains(rg.Name, "fixture_decode")) || physicalValue(t, rg) != float64(p.grid) {
						t.Fatalf("unexpected native kernel range: %+v", rg)
					}
				}
			}
			// Reuse must restore the same mutable sequence, not continue prior state.
			_ = physicalResult(t, p, plan, "", test.ops)
			checkPhysicalOutput(t, p, "", test.ops)
		})
	}
}

func TestPhysicalCuptibenchCounterControls(t *testing.T) {
	p := newPhysicalPipeline(t, 2)
	plan := physicalPlan(t, p, cuptibench.Whole, 3)
	baseline := physicalResult(t, p, plan, "", 3)
	if physicalValue(t, baseline.Ranges[0]) != 12 {
		t.Fatalf("baseline CTAs=%g want=12", physicalValue(t, baseline.Ranges[0]))
	}
	p.grid = 4
	changed := physicalResult(t, p, plan, "", 3)
	checkPhysicalOutput(t, p, "", 3)
	value := physicalValue(t, changed.Ranges[0])
	if value != 24 || value != 2*physicalValue(t, baseline.Ranges[0]) {
		t.Fatalf("changed workload response=%g baseline=%g", value, physicalValue(t, baseline.Ranges[0]))
	}
	// A wrong denominator (operations plus one) must fail the exact block oracle.
	if value/float64(changed.Operations) != 8 || value/float64(changed.Operations+1) == 8 {
		t.Fatal("normalization oracle admitted wrong operation count")
	}
	if changed.Quality.PassesStarted != uint(changed.Operations) && value/float64(changed.Quality.PassesStarted) == 8 {
		t.Fatal("normalization oracle admitted pass count as logical operations")
	}
	// Deliberately put the reset kernel inside collection. The block oracle must
	// detect its eight extra blocks instead of accepting a contaminated result.
	one := physicalPlan(t, p, cuptibench.Whole, 1)
	contaminated := p.workload("")
	run := contaminated.Run
	contaminated.Run = func(scope *cuptibench.Scope) error {
		if err := p.prepare(); err != nil {
			return err
		}
		return run(scope)
	}
	mutant, err := one.Collect(contaminated)
	physicalCheck(t, err)
	if len(mutant.Ranges) != 1 || physicalValue(t, mutant.Ranges[0]) != 16 || physicalValue(t, mutant.Ranges[0]) == 8 {
		t.Fatalf("reset contamination oracle failed: %+v", mutant.Ranges)
	}
}

func TestPhysicalCuptibenchIsolated(t *testing.T) {
	// Each selected child constructs its own device fixture. No sibling or full
	// benchmark is required to have executed before an isolated stage.
	for _, stage := range []string{"project", "decode"} {
		t.Run(stage, func(t *testing.T) {
			p := newPhysicalPipeline(t, 4)
			plan := physicalPlan(t, p, cuptibench.Kernels, 1)
			r := physicalResult(t, p, plan, stage, 1)
			if len(r.Ranges) != 1 || !strings.Contains(r.Ranges[0].Name, "fixture_"+stage) || physicalValue(t, r.Ranges[0]) != 4 {
				t.Fatalf("isolated range=%+v", r.Ranges)
			}
			checkPhysicalOutput(t, p, stage, 1)
		})
	}
}

// BenchmarkPhysicalPipeline demonstrates a reusable application-owned execution
// hierarchy. It deliberately uses a synthetic pipeline; no DS4 parity is claimed.
// Example selection: -bench '^BenchmarkPhysicalPipeline$/^isolated$/^decode$'.
func BenchmarkPhysicalPipeline(b *testing.B) {
	b.Run("full", func(b *testing.B) {
		p := newPhysicalPipeline(b, 4)
		plan := physicalPlan(b, p, cuptibench.Regions, 1)
		cuptibench.Benchmark(b, plan, p.workload(""))
		checkPhysicalOutput(b, p, "", 1)
	})
	b.Run("isolated", func(b *testing.B) {
		for _, stage := range []string{"project", "decode"} {
			b.Run(stage, func(b *testing.B) {
				p := newPhysicalPipeline(b, 4)
				plan := physicalPlan(b, p, cuptibench.Kernels, 1)
				r := cuptibench.Benchmark(b, plan, p.workload(stage))
				if len(r.Ranges) != 1 {
					b.Fatalf("single-kernel claim captured %d ranges", len(r.Ranges))
				}
				checkPhysicalOutput(b, p, stage, 1)
			})
		}
	})
}

func TestPhysicalTwoStreamTiming(t *testing.T) {
	p := newPhysicalPipeline(t, 4)
	timer, err := cuda.NewEventTimer(p.dev, p.streams[:])
	if timer != nil {
		physicalClose(t, timer.Close)
	}
	physicalCheck(t, err)
	// Predeclared tolerances: event-vs-host completion within 25% plus 5 ms;
	// each fixed 20 ms kernel within [15,100] ms; overlapping span <85% of sum.
	// Failure remains a failed qualification gate; do not tune these after a run.
	for _, dependent := range []bool{false, true} {
		t.Run(fmt.Sprintf("dependent=%t", dependent), func(t *testing.T) {
			starts, ends := [2]*cuda.Event{}, [2]*cuda.Event{}
			for i := range starts {
				starts[i], err = p.dev.NewEvent()
				physicalCheck(t, err)
				physicalClose(t, starts[i].Destroy)
				ends[i], err = p.dev.NewEvent()
				physicalCheck(t, err)
				physicalClose(t, ends[i].Destroy)
			}
			physicalCheck(t, p.synchronize())
			begin := time.Now()
			duration, err := timer.Measure(func() error {
				for i, s := range p.streams {
					if dependent && i == 1 {
						if err := s.WaitEvent(ends[0]); err != nil {
							return err
						}
					}
					if err := starts[i].Record(s); err != nil {
						return err
					}
					if err := cudalaunch.Kernel(p.wait).Block(physicalBlock).On(s).Launch(p.out, uint32(i), uint32(i+1)); err != nil {
						return err
					}
					if err := ends[i].Record(s); err != nil {
						return err
					}
				}
				return nil
			})
			wall := time.Since(begin)
			physicalCheck(t, err)
			durations := [2]time.Duration{}
			for i := range durations {
				durations[i], err = ends[i].ElapsedSince(starts[i])
				physicalCheck(t, err)
				if durations[i] < 15*time.Millisecond || durations[i] > 100*time.Millisecond {
					t.Fatalf("fixed kernel duration outside admission: %v", durations)
				}
			}
			sum := durations[0] + durations[1]
			if duration <= 0 || math.Abs(float64(wall-duration)) > float64(wall)/4+float64(5*time.Millisecond) {
				t.Fatalf("completion mismatch event=%v host=%v", duration, wall)
			}
			if dependent {
				if duration < sum*9/10 {
					t.Fatalf("dependent work omitted: span=%v sum=%v", duration, sum)
				}
			} else if duration >= sum*85/100 {
				t.Fatalf("overlap not established: span=%v component sum=%v", duration, sum)
			}
			output := make([]uint32, physicalCapacity)
			physicalCheck(t, p.out.CopyToHost(output))
			if output[0] != 1 || output[1] != 2 {
				t.Fatalf("incomplete multi-stream outputs %v", output[:2])
			}
			t.Logf("dependent=%t event=%v host=%v stream-events=%v", dependent, duration, wall, durations)
		})
	}
}

func TestPhysicalCuptibenchInsufficientRangeCapacity(t *testing.T) {
	p := newPhysicalPipeline(t, 4)
	// The positive capture proves the two-kernel workload is available before
	// reducing capacity. Native rejection may occur before Decode; only a decoded
	// DroppedRanges value is evidence of actual loss.
	good := physicalPlan(t, p, cuptibench.Kernels, 1)
	baseline := physicalResult(t, p, good, "", 1)
	if len(baseline.Ranges) != 2 {
		t.Fatalf("positive kernel count=%d", len(baseline.Ranges))
	}
	bounded, err := cuptibench.New(p.dev, cuptibench.Config{CachePolicy: cuptirange.CachePreserve, Metrics: []string{physicalMetric}, Mode: cuptibench.Kernels, Operations: 1, MaxConfigBytes: 16 << 20, Limits: cuptirange.Limits{MaxRanges: 1, MaxRangeTreeNodes: 1, MaxCounterBytes: 32 << 20, MaxPasses: 128}})
	if bounded != nil {
		physicalClose(t, bounded.Close)
	}
	physicalCheck(t, err)
	// Exact capacity is admissible when the independent launch count and native
	// identity agree. Preparation's reset kernel must remain excluded.
	exact := physicalResult(t, p, bounded, "project", 1)
	if len(exact.Ranges) != 1 || !strings.Contains(exact.Ranges[0].Name, "fixture_project") || exact.Ranges[0].NativeName != "0" || physicalValue(t, exact.Ranges[0]) != 4 {
		t.Fatalf("exact-capacity identity/counters=%+v", exact.Ranges)
	}
	checkPhysicalOutput(t, p, "project", 1)
	rejected, err := bounded.Collect(p.workload(""))
	if err == nil {
		t.Fatalf("insufficient capacity admitted: %+v", rejected)
	}
	if rejected != nil && len(rejected.Ranges) != 0 {
		t.Fatalf("failed collection retained admissible counters: %+v", rejected)
	}
	t.Logf("capacity rejection=%v diagnostic result=%+v", err, rejected)
}

func TestPhysicalCuptibenchProfile(t *testing.T) {
	if os.Getenv("CUPTIBENCH_PHYSICAL_TEST") != "1" {
		t.Skip("unqualified: CUPTIBENCH_PHYSICAL_TEST=1 required")
	}
	// Run the real benchmark adapter and process Finish in a fresh test process;
	// unit-created pprof samples cannot establish physical counter provenance.
	home, err := os.UserHomeDir()
	physicalCheck(t, err)
	root := filepath.Join(home, "tmp")
	physicalCheck(t, os.MkdirAll(root, 0700))
	dir, err := os.MkdirTemp(root, "cuptibench-physical-")
	physicalCheck(t, err)
	t.Logf("retained physical profile receipt directory=%s", dir)
	exe, err := os.Executable()
	physicalCheck(t, err)
	path := filepath.Join(dir, "pipeline.pprof")
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe, "-test.run=^$", "-test.bench=^BenchmarkPhysicalPipeline$", "-test.benchtime=1x", "-test.count=1", "-cuptiprofile="+path)
	cmd.Env = append(os.Environ(), "CUPTIBENCH_EXPORT_TEST=1")
	output, runErr := cmd.CombinedOutput()
	physicalCheck(t, os.WriteFile(filepath.Join(dir, "runner.txt"), output, 0600))
	if runErr != nil {
		t.Fatalf("physical profile child failed: %v\n%s", runErr, output)
	}
	data, err := os.ReadFile(path + ".json")
	physicalCheck(t, err)
	var manifest struct{ Results []*cuptibench.Result }
	physicalCheck(t, json.Unmarshal(data, &manifest))
	if len(manifest.Results) != 3 {
		t.Fatalf("profile results=%d want=3", len(manifest.Results))
	}
	captures := make(map[string]*cuptibench.Result)
	for _, r := range manifest.Results {
		if r.Operations != 1 || r.Timing == nil || r.Timing.Operations != 1 || r.Timing.Total <= 0 {
			t.Fatalf("profile accounting=%+v", r)
		}
		view := r.Metadata["view"]
		if strings.HasSuffix(r.Metadata["benchmark"], "/full") {
			if view != "full" {
				t.Fatalf("full view=%q", view)
			}
		} else if view != "isolated" {
			t.Fatalf("isolated view=%q", view)
		}
		if r.Metadata["capture_id"] == "" || r.Metadata["run_id"] == "" {
			t.Fatal("missing profile experiment identities")
		}
		captures[r.Metadata["capture_id"]] = r
	}
	encoded, err := os.ReadFile(path)
	physicalCheck(t, err)
	parsed, err := profile.ParseData(encoded)
	physicalCheck(t, err)
	physicalCheck(t, parsed.CheckValid())
	if len(parsed.SampleType) != 1 || parsed.SampleType[0].Type != physicalMetric || parsed.DefaultSampleType != physicalMetric {
		t.Fatalf("profile schema/default=%+v/%s", parsed.SampleType, parsed.DefaultSampleType)
	}
	sum := int64(0)
	if len(parsed.Sample) != 4 {
		t.Fatalf("pprof samples=%d want=4 disjoint full leaves plus isolated kernels", len(parsed.Sample))
	}
	for _, sample := range parsed.Sample {
		if len(sample.Value) != 1 || sample.Value[0] != 4 {
			t.Fatalf("physical profile CTA value=%v want=4", sample.Value)
		}
		sum += sample.Value[0]
		ids := sample.Label["capture_id"]
		if len(ids) != 1 || captures[ids[0]] == nil {
			t.Fatalf("sample lost source capture: %v", sample.Label)
		}
		r := captures[ids[0]]
		if v := sample.Label["view"]; len(v) != 1 || v[0] != r.Metadata["view"] {
			t.Fatalf("sample view=%v", v)
		}
		if v := sample.Label["benchmark"]; len(v) != 1 || v[0] != r.Metadata["benchmark"] {
			t.Fatalf("sample benchmark=%v", v)
		}
		if r.Metadata["view"] == "full" {
			if len(sample.Label["kernel"]) != 0 {
				t.Fatal("algorithmic region mislabeled as kernel")
			}
		} else if len(sample.Label["kernel"]) != 1 || !strings.Contains(sample.Label["kernel"][0], "fixture_") {
			t.Fatalf("isolated native kernel identity=%v", sample.Label)
		}
	}
	if sum != 16 {
		t.Fatalf("profile sum=%d want=16 across three separately labelled experiments", sum)
	}
}

const physicalTrafficRead = "lts__t_sectors_srcunit_tex_op_read.sum"
const physicalTrafficMiss = "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum"
const physicalTrafficSource = `
extern "C" __global__ void traffic_initialize(unsigned *data, unsigned count) {
 for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
      i += gridDim.x * blockDim.x) data[i] = i * 1664525u + 1013904223u;
}
extern "C" __global__ void traffic_read(const unsigned *data, unsigned *out, unsigned count) {
 __shared__ unsigned partial[256];
 unsigned sum = 0;
 for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
      i += gridDim.x * blockDim.x) {
  unsigned value;
  // Bypass L1 so a warmed L1 cannot suppress the L2 read-counter denominator.
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(data + i));
  sum += value;
 }
 partial[threadIdx.x] = sum;
 __syncthreads();
 for (unsigned offset = 128; offset != 0; offset >>= 1) {
  if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
  __syncthreads();
 }
 if (threadIdx.x == 0) out[blockIdx.x] = partial[0];
}
`

// TestPhysicalFilteredL2Traffic qualifies this particular read path and cache
// regime. It does not establish DRAM traffic, a cross-chip conversion factor,
// bandwidth, or universal upper/lower bounds. Raw TEX-source read and read-miss
// sector counts remain distinct. The 32-byte sector assumption is documented at
// https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference
// and is challenged by the exact read-count control rather than silently used
// to relabel a measured value as bytes.
func TestPhysicalFilteredL2Traffic(t *testing.T) {
	p := newPhysicalPipeline(t, 4)
	var l2 int32
	physicalCheck(t, p.dev.WithContext(func() error { var err error; l2, err = p.dev.Raw().GetAttribute(driver.DevAttrL2CacheSize); return err }))
	if l2 < 256<<10 {
		t.Fatalf("unqualified L2 capacity %d: fixture requires at least 256 KiB", l2)
	}
	targetBytes := int64(l2) / 16
	if targetBytes > 2<<20 {
		targetBytes = 2 << 20
	}
	targetBytes = targetBytes / 1024 * 1024
	evictionBytes := 4 * int64(l2)
	if evictionBytes+targetBytes > 512<<20 {
		t.Fatalf("L2=%d requires allocation above the fixed 512 MiB fixture bound", l2)
	}
	target, err := cudamem.Alloc[uint32](p.dev, int(targetBytes/4))
	physicalCheck(t, err)
	physicalClose(t, target.Free)
	eviction, err := cudamem.Alloc[uint32](p.dev, int(evictionBytes/4))
	physicalCheck(t, err)
	physicalClose(t, eviction.Free)
	cm, err := nvrtcx.Compile(physicalTrafficSource, nvrtcx.WithDeviceArch(p.dev), nvrtcx.WithNoCache())
	physicalCheck(t, err)
	module, err := cm.Load(p.dev)
	physicalCheck(t, err)
	physicalClose(t, func() error { return p.dev.WithContext(module.Unload) })
	var initialize, read driver.Function
	physicalCheck(t, p.dev.WithContext(func() error {
		var err error
		initialize, err = module.GetFunction("traffic_initialize")
		if err != nil {
			return err
		}
		read, err = module.GetFunction("traffic_read")
		return err
	}))
	launch := func(data cudamem.Slice[uint32], words uint32) error {
		return cudalaunch.Kernel(read).Grid(128).Block(256).On(p.streams[0]).Launch(data, p.out, words)
	}
	for _, data := range []cudamem.Slice[uint32]{target, eviction} {
		physicalCheck(t, p.dev.WithContext(func() error {
			return cudalaunch.Kernel(initialize).Grid(128).Block(256).On(p.streams[0]).Launch(data, uint32(data.Len()))
		}))
	}
	physicalCheck(t, p.synchronize())
	t.Logf("traffic source sha256=%x L2_attribute_bytes=%d target_bytes=%d eviction_bytes=%d; ld.global.cg.u32; sector assumption=32; cache-control=preserve; no clock/access-policy changes", sha256.Sum256([]byte(physicalTrafficSource)), l2, targetBytes, evictionBytes)
	// Device-specific admission: twice reading a <=L2/16 target must produce
	// <=10% filtered L2 misses; sweeping a disjoint 4xL2 buffer must produce >=90%
	// misses on the following target read. These are preregistered fixture gates,
	// not promises that eviction or residency holds on every architecture.
	type observation struct {
		read, miss float64
		ops        int
		sectors    float64
	}
	observed := make(map[string]observation)
	for _, tc := range []struct {
		name         string
		warm         bool
		ops, divisor int
	}{
		{"evicted", false, 1, 1}, {"warm", true, 1, 1}, {"warm-repeat", true, 1, 1},
		{"warm-sequence", true, 3, 1}, {"evicted-sequence", false, 3, 1}, {"warm-half", true, 1, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			words := uint32(target.Len() / tc.divisor)
			expectedPerOp := float64(words) / 8 // Eight contiguous uint32 elements per sector.
			plan, err := cuptibench.New(p.dev, cuptibench.Config{CachePolicy: cuptirange.CachePreserve, Metrics: []string{physicalTrafficRead, physicalTrafficMiss}, Mode: cuptibench.Whole, Operations: tc.ops, MaxConfigBytes: 16 << 20, Limits: cuptirange.Limits{MaxRanges: 4, MaxRangeTreeNodes: 4, MaxCounterBytes: 16 << 20, MaxPasses: 64}})
			if plan != nil {
				physicalClose(t, plan.Close)
			}
			physicalCheck(t, err)
			prepares, runs := 0, 0
			preparation := fmt.Sprintf("candidate evicted: disjoint %d-byte cg read sweep before each replay sequence; driver L2=%d; target=%d; cache-control=preserve", evictionBytes, l2, words*4)
			if tc.warm {
				preparation = fmt.Sprintf("candidate warm: two cg reads of %d-byte target before each replay sequence; driver L2=%d; cache-control=preserve", words*4, l2)
			}
			w := cuptibench.Workload{View: "isolated", Streams: p.streams[:], Preparation: preparation, Synchronize: p.synchronize,
				Prepare: func() error {
					prepares++
					if !tc.warm {
						return launch(eviction, uint32(eviction.Len()))
					}
					if err := launch(target, words); err != nil {
						return err
					}
					return launch(target, words)
				},
				Run: func(*cuptibench.Scope) error { runs++; return launch(target, words) },
			}
			result, err := plan.Collect(w)
			if result != nil {
				result.Metadata["fixture_source_sha256"] = fmt.Sprintf("%x", sha256.Sum256([]byte(physicalTrafficSource)))
				result.Metadata["fixture_sector_bytes_assumption"] = "32; exact filtered read-count oracle required; no DRAM interpretation"
				encoded, err := json.Marshal(result)
				physicalCheck(t, err)
				t.Logf("raw filtered traffic result=%s", encoded)
			}
			physicalCheck(t, err)
			if result == nil || len(result.Ranges) != 1 || len(result.Ranges[0].Metrics) != 2 {
				t.Fatalf("invalid traffic result: %+v", result)
			}
			q := result.Quality
			if !q.Decoded || !q.AllPassesSubmitted || q.DroppedRanges != 0 || q.PassesStarted == 0 || q.PassesStarted != q.PassesStopped || prepares != int(q.PassesStarted) || runs != tc.ops*int(q.PassesStarted) || result.Operations != tc.ops {
				t.Fatalf("traffic replay accounting prepare=%d run=%d result=%+v", prepares, runs, result)
			}
			if result.Metadata["cache_control_requested"] != "preserve" || result.Metadata["cache_control_effective"] != "none" {
				t.Fatalf("unstated cache control: %v", result.Metadata)
			}
			got := make([]uint32, physicalCapacity)
			physicalCheck(t, p.out.CopyToHost(got))
			want := make([]uint32, 128)
			for i := uint32(0); i < words; i++ {
				want[(i/256)%128] += i*1664525 + 1013904223
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("read output block%d=%d want=%d", i, got[i], want[i])
				}
			}
			metrics := result.Ranges[0].Metrics
			if metrics[0].Name != physicalTrafficRead || metrics[1].Name != physicalTrafficMiss {
				t.Fatalf("filtered metric identity changed: %+v", metrics)
			}
			reads, misses := metrics[0].Value, metrics[1].Value
			if reads != expectedPerOp*float64(tc.ops) || misses < 0 || misses > reads {
				t.Fatalf("read-count/filter oracle: reads=%g misses=%g want reads=%g", reads, misses, expectedPerOp*float64(tc.ops))
			}
			if tc.warm {
				if misses > reads/10 {
					t.Fatalf("warm-state inverse control failed: reads=%g misses=%g", reads, misses)
				}
			} else {
				// Preparation is once per SEQUENCE. Later operations should reuse the first
				// operation's cache contents; multiplying cold misses by ops is a mutant.
				if misses < expectedPerOp*0.9 || misses > expectedPerOp*1.1 {
					t.Fatalf("evicted sequence oracle: misses=%g expected one cold read=%g operations=%d", misses, expectedPerOp, tc.ops)
				}
			}

			observed[tc.name] = observation{reads, misses, tc.ops, expectedPerOp}
		})
	}
	if t.Failed() {
		return
	}
	cold, warm, sequence, half := observed["evicted"], observed["warm"], observed["warm-sequence"], observed["warm-half"]
	if cold.read != warm.read || cold.miss <= warm.miss || sequence.read != 3*warm.read || 2*half.read != warm.read {
		t.Fatalf("cache/workload differential failed: %v", observed)
	}
	if sequence.read/float64(sequence.ops) != sequence.sectors || sequence.read/float64(sequence.ops+1) == sequence.sectors {
		t.Fatal("traffic normalization admitted incorrect logical operation count")
	}
}

// TestPhysicalReplayPassCounts requires configurations witnessed to need one,
// two, and four native passes on this device/backend. MaxPasses is the same
// budget (eight) in every case and is never used to force a replay count.
// CUPTIBENCH_REPLAY_METRICS_2 and _4 contain explicit comma-separated metric
// names selected through native host configuration queries. Supply all three
// lists together; with no lists this optional metric-schedule test is skipped.
// The CTA control is always first and is appended to each supplied selection if
// absent. A list found on another chip/version must pass planning again here.
func TestPhysicalReplayPassCounts(t *testing.T) {
	if os.Getenv("CUPTIBENCH_PHYSICAL_TEST") != "1" {
		t.Skip("unqualified: CUPTIBENCH_PHYSICAL_TEST=1 required")
	}
	supplied := false
	for _, passes := range []uint{1, 2, 4} {
		if _, ok := os.LookupEnv(fmt.Sprintf("CUPTIBENCH_REPLAY_METRICS_%d", passes)); ok {
			supplied = true
		}
	}
	if !supplied {
		t.Skip("optional metric-schedule probe: no CUPTIBENCH_REPLAY_METRICS_1/_2/_4 supplied; required actual replay-count gate is TestPhysicalReplayNestingPassCounts")
	}
	selections := make(map[uint][]string)
	for _, passes := range []uint{1, 2, 4} {
		key := fmt.Sprintf("CUPTIBENCH_REPLAY_METRICS_%d", passes)
		raw := strings.TrimSpace(os.Getenv(key))
		if raw == "" {
			t.Fatalf("missing %s: select native metrics with Plan.Configuration reporting exactly %d passes; MaxPasses cannot force this", key, passes)
		}
		names := []string{physicalMetric}
		seen := map[string]bool{physicalMetric: true}
		if raw != "" {
			for _, name := range strings.Split(raw, ",") {
				name = strings.TrimSpace(name)
				if name == "" {
					t.Fatalf("empty metric in %s", key)
				}
				if !seen[name] {
					seen[name] = true
					names = append(names, name)
				}
			}
		}
		if len(names) > 128 {
			t.Fatalf("%s exceeds the fixture limit of 128 metrics", key)
		}
		selections[passes] = names
	}
	p := newPhysicalPipeline(t, 4)
	var chip string
	physicalCheck(t, p.dev.WithContext(func() error { var err error; chip, err = cuptisass.DeviceChipName(uint32(p.dev.Ordinal())); return err }))
	availability, err := cuptirange.CounterAvailability(p.dev, 16<<20)
	physicalCheck(t, err)
	t.Logf("replay-plan chip=%s availability_sha256=%x; configured budget=8; logical operations=3", chip, sha256.Sum256(availability))
	counts := make(map[uint]float64)
	for _, passes := range []uint{1, 2, 4} {
		t.Run(fmt.Sprintf("passes%d", passes), func(t *testing.T) {
			names := selections[passes]
			query, err := cuptirange.NewPlan(chip, availability, names, 16<<20)
			if query != nil {
				physicalClose(t, query.Close)
			}
			physicalCheck(t, err)
			configuration, required := query.Configuration()
			physicalCheck(t, query.Close())
			t.Logf("requested_passes=%d planned_passes=%d metric_names=%q configuration_sha256=%x", passes, required, names, sha256.Sum256(configuration))
			if required != passes {
				t.Fatalf("metric selection requires %d native passes, wanted %d; choose a different selection, do not substitute a replay budget", required, passes)
			}
			plan, err := cuptibench.New(p.dev, cuptibench.Config{CachePolicy: cuptirange.CachePreserve, Metrics: names, Mode: cuptibench.Whole, Operations: 3, MaxConfigBytes: 16 << 20, Limits: cuptirange.Limits{MaxRanges: 8, MaxRangeTreeNodes: 8, MaxCounterBytes: 32 << 20, MaxPasses: 8}})
			if plan != nil {
				physicalClose(t, plan.Close)
			}
			physicalCheck(t, err)
			result := physicalResult(t, p, plan, "", 3)
			checkPhysicalOutput(t, p, "", 3)
			if result.Quality.PassesStarted != passes || result.Quality.PassesStopped != passes {
				t.Fatalf("planned=%d actual=%+v: no actual %d-pass qualification", required, result.Quality, passes)
			}
			if len(result.Ranges) != 1 || len(result.Ranges[0].Metrics) != len(names) || result.Ranges[0].Metrics[0].Name != physicalMetric {
				t.Fatalf("missing CTA control: %+v", result.Ranges)
			}
			ctas := result.Ranges[0].Metrics[0].Value
			// Three logical operations launch two four-block kernels each. Changing the
			// metric schedule must not change this 24-block numerator or its 8-block/op
			// normalization. Deliberately dividing by any selected pass count is wrong.
			if ctas != 24 || ctas/float64(result.Operations) != 8 || ctas/float64(passes) == 8 {
				t.Fatalf("normalization failure CTAs=%g operations=%d passes=%d", ctas, result.Operations, passes)
			}
			counts[passes] = ctas
		})
	}
	if t.Failed() {
		return
	}
	if counts[1] != counts[2] || counts[2] != counts[4] {
		t.Fatalf("metric schedule changed logical operation counts: %v", counts)
	}
}

// TestPhysicalReplayNestingPassCounts uses the native nesting schedule to
// exercise actual one/two/four-pass collection with one unchanged CTA metric.
// This is separate from metric-schedule qualification above. MaxPasses stays
// eight in all cases; only observed Start/Stop completion counts establish that
// the requested native replay regime occurred.
func TestPhysicalReplayNestingPassCounts(t *testing.T) {
	p := newPhysicalPipeline(t, 4)
	counts := make(map[uint16]float64)
	for _, depth := range []uint16{1, 2, 4} {
		t.Run(fmt.Sprintf("depth%d", depth), func(t *testing.T) {
			plan, err := cuptibench.New(p.dev, cuptibench.Config{CachePolicy: cuptirange.CachePreserve, Metrics: []string{physicalMetric}, Mode: cuptibench.Regions, MaxDepth: depth, Operations: 3, MaxConfigBytes: 16 << 20, Limits: cuptirange.Limits{MaxRanges: 32, MaxRangeTreeNodes: 32, MaxCounterBytes: 32 << 20, MaxPasses: 8}})
			if plan != nil {
				physicalClose(t, plan.Close)
			}
			physicalCheck(t, err)
			beforePrepare, beforeRun := p.prepares, p.runs
			w := cuptibench.Workload{View: "full", Streams: p.streams[:], Prepare: p.prepare, Synchronize: p.synchronize, Preparation: "GPU reset once per three-operation sequence; CUPTI no-reset policy; native nesting depth controls pass schedule"}
			w.Run = func(scope *cuptibench.Scope) error {
				p.runs++
				var nested func(uint16) error
				nested = func(level uint16) error {
					if level == depth {
						if err := p.projectStage(); err != nil {
							return err
						}
						return p.decodeStage()
					}
					return scope.Region(fmt.Sprintf("level%d", level), func() error { return nested(level + 1) })
				}
				return nested(1)
			}
			result, err := plan.Collect(w)
			if result != nil {
				data, encodeErr := json.Marshal(result)
				physicalCheck(t, encodeErr)
				t.Logf("native nesting depth=%d common MaxPasses=8 result=%s", depth, data)
			}
			physicalCheck(t, err)
			checkPhysicalOutput(t, p, "", 3)
			q := result.Quality
			if !q.Decoded || !q.AllPassesSubmitted || q.DroppedRanges != 0 || q.PassesStarted != uint(depth) || q.PassesStopped != uint(depth) || p.prepares-beforePrepare != int(depth) || p.runs-beforeRun != 3*int(depth) || result.Operations != 3 {
				t.Fatalf("native nesting schedule not witnessed: depth=%d quality=%+v prepares=%d runs=%d operations=%d", depth, q, p.prepares-beforePrepare, p.runs-beforeRun, result.Operations)
			}
			roots := 0
			for _, rg := range result.Ranges {
				if rg.Kind == "operation" && rg.ParentID == "" {
					roots++
					counts[depth] = physicalValue(t, rg)
				}
			}
			ctas := counts[depth]
			if roots != 1 || ctas != 24 || ctas/float64(result.Operations) != 8 || ctas/float64(q.PassesStarted) == 8 {
				t.Fatalf("native replay normalization: roots=%d CTAs=%g operations=%d passes=%d", roots, ctas, result.Operations, q.PassesStarted)
			}
		})
	}
	if t.Failed() {
		return
	}
	if counts[1] != counts[2] || counts[2] != counts[4] {
		t.Fatalf("native pass count changed CTA numerator: %v", counts)
	}
}

// Graph capture preserves the real project->decode dependency. Preparation is
// outside both graph execution and native counter collection on every user pass.
func TestPhysicalGraphWholeAndRegions(t *testing.T) {
	p := newPhysicalPipeline(t, 4)
	var launchErr error
	graph, err := cudagraph.Capture(p.streams[0], func() {
		launchErr = cudalaunch.Kernel(p.project).Grid(p.grid).Block(physicalBlock).On(p.streams[0]).Launch(p.state, p.mid)
		if launchErr == nil {
			launchErr = cudalaunch.Kernel(p.decode).Grid(p.grid).Block(physicalBlock).On(p.streams[0]).Launch(p.state, p.mid, p.out)
		}
	})
	physicalCheck(t, err)
	physicalClose(t, graph.Destroy)
	physicalCheck(t, launchErr)
	executable, err := graph.Instantiate()
	physicalCheck(t, err)
	physicalClose(t, executable.Destroy)
	// The executable survives source graph destruction.
	physicalCheck(t, graph.Destroy())
	for _, mode := range []cuptibench.Mode{cuptibench.Whole, cuptibench.Regions} {
		for _, ops := range []int{1, 3} {
			t.Run(fmt.Sprintf("mode%d/ops%d", mode, ops), func(t *testing.T) {
				plan, err := cuptibench.New(p.dev, cuptibench.Config{Mode: mode, Operations: ops, MaxDepth: func() uint16 {
					if mode == cuptibench.Regions {
						return 2
					}
					return 1
				}(), CachePolicy: cuptirange.CachePreserve, Metrics: []string{physicalMetric}, MaxConfigBytes: 16 << 20, Limits: cuptirange.Limits{MaxRanges: 64, MaxRangeTreeNodes: 128, MaxCounterBytes: 32 << 20, MaxPasses: 8}})
				physicalCheck(t, err)
				physicalClose(t, plan.Close)
				w := cuptibench.Workload{View: "full", Streams: p.streams[:1], Prepare: p.prepare, Synchronize: p.synchronize, Preparation: "reset state, mid and out before every pass; same-stream captured project then decode; CUPTI cache preserve", Run: func(s *cuptibench.Scope) error {
					p.runs++
					return s.Region("graph", func() error { return executable.Launch(p.streams[0]) })
				}}
				for repeat := 0; repeat < 2; repeat++ {
					beforePrepare, beforeRun := p.prepares, p.runs
					r, err := plan.Collect(w)
					physicalCheck(t, err)
					raw, err := json.Marshal(r)
					physicalCheck(t, err)
					t.Logf("graph result=%s", raw)
					if !r.Quality.Decoded || !r.Quality.AllPassesSubmitted || r.Quality.DroppedRanges != 0 || r.Quality.PassesStarted == 0 || r.Quality.PassesStarted != r.Quality.PassesStopped {
						t.Fatalf("graph quality=%+v", r.Quality)
					}
					if p.prepares-beforePrepare != int(r.Quality.PassesStarted) || p.runs-beforeRun != ops*int(r.Quality.PassesStarted) {
						t.Fatal("graph callback accounting differs from native passes")
					}
					roots, children := 0, 0
					for _, v := range r.Ranges {
						if v.Kind == "operation" {
							roots++
							if physicalValue(t, v) != float64(8*ops) {
								t.Fatalf("root=%+v", v)
							}
						} else {
							children++
							if v.Name != "graph" || v.Kind != "region" || physicalValue(t, v) != 8 {
								t.Fatalf("graph region=%+v", v)
							}
						}
					}
					wantChildren := 0
					if mode == cuptibench.Regions {
						wantChildren = ops
					}
					if roots != 1 || children != wantChildren {
						t.Fatalf("roots=%d children=%d", roots, children)
					}
					checkPhysicalOutput(t, p, "", ops)
				}
			})
		}
	}
}

// Unequal block counts make swapped name/ordinal associations observable. Both
// orders start with valid mid inputs prepared outside collection. A retained
// graph proof must preserve this order after destruction of the source graph.
func TestPhysicalKernelIdentityAssociation(t *testing.T) {
	p := newPhysicalPipeline(t, 5)
	for _, graphMode := range []bool{false, true} {
		for _, reverse := range []bool{false, true} {
			t.Run(fmt.Sprintf("graph%v/reverse%v", graphMode, reverse), func(t *testing.T) {
				project := func() error {
					return cudalaunch.Kernel(p.project).Grid(2).Block(physicalBlock).On(p.streams[0]).Launch(p.state, p.mid)
				}
				decode := func() error {
					return cudalaunch.Kernel(p.decode).Grid(5).Block(physicalBlock).On(p.streams[0]).Launch(p.state, p.mid, p.out)
				}
				sequence := func() error {
					first, second := project, decode
					if reverse {
						first, second = decode, project
					}
					if err := first(); err != nil {
						return err
					}
					return second()
				}
				replayMode := cuptirange.UserReplay
				var graphs []*cudagraph.Exec
				if graphMode {
					var launchErr error
					graph, err := cudagraph.Capture(p.streams[0], func() { launchErr = sequence() })
					physicalCheck(t, err)
					physicalClose(t, graph.Destroy)
					physicalCheck(t, launchErr)
					executable, err := graph.InstantiateWithOptions(cudagraph.InstantiateOptions{RetainKernelOrder: true})
					physicalCheck(t, err)
					physicalClose(t, executable.Destroy)
					physicalCheck(t, graph.Destroy())
					graphs = []*cudagraph.Exec{executable}
					sequence = func() error { return executable.Launch(p.streams[0]) }
					replayMode = cuptirange.KernelReplay
				}
				plan, err := cuptibench.New(p.dev, cuptibench.Config{Mode: cuptibench.Kernels, ReplayMode: replayMode, Operations: 1, CachePolicy: cuptirange.CachePreserve, Metrics: []string{physicalMetric}, MaxConfigBytes: 16 << 20, Limits: cuptirange.Limits{MaxRanges: 8, MaxRangeTreeNodes: 16, MaxCounterBytes: 32 << 20, MaxPasses: 8}})
				physicalCheck(t, err)
				physicalClose(t, plan.Close)
				prepares, runs := 0, 0
				w := cuptibench.Workload{View: "full", Graphs: graphs, Streams: p.streams[:1], Synchronize: p.synchronize, Preparation: "reset all buffers then project all five blocks before profiling; measured project two blocks and decode five; cache preserve", Prepare: func() error {
					prepares++
					if err := p.prepare(); err != nil {
						return err
					}
					return cudalaunch.Kernel(p.project).Grid(5).Block(physicalBlock).On(p.streams[0]).Launch(p.state, p.mid)
				}, Run: func(*cuptibench.Scope) error { runs++; return sequence() }}
				r, err := plan.Collect(w)
				physicalCheck(t, err)
				raw, err := json.Marshal(r)
				physicalCheck(t, err)
				t.Logf("identity result=%s", raw)
				if len(r.Ranges) != 2 {
					t.Fatalf("ranges=%+v", r.Ranges)
				}
				names := []string{"fixture_project", "fixture_decode"}
				counts := []float64{2, 5}
				if reverse {
					names[0], names[1] = names[1], names[0]
					counts[0], counts[1] = counts[1], counts[0]
				}
				for i, v := range r.Ranges {
					if !strings.Contains(v.Name, names[i]) || v.NativeName != fmt.Sprint(i) || physicalValue(t, v) != counts[i] {
						t.Fatalf("identity[%d]=%+v want=%s/%d/%g", i, v, names[i], i, counts[i])
					}
				}
				if prepares != int(r.Quality.PassesStarted) || runs != prepares || !r.Quality.AllPassesSubmitted || !r.Quality.Decoded || r.Quality.DroppedRanges != 0 {
					t.Fatalf("callbacks=%d/%d quality=%+v", prepares, runs, r.Quality)
				}
				if graphMode && (prepares != 1 || runs != 1 || r.Quality.PassesStopped != 1) {
					t.Fatal("kernel replay repeated host callbacks")
				}
				physicalCheck(t, p.synchronize())
				state, mid, out := make([]uint32, physicalCapacity), make([]uint32, physicalCapacity), make([]uint32, physicalCapacity)
				physicalCheck(t, p.state.CopyToHost(state))
				physicalCheck(t, p.mid.CopyToHost(mid))
				physicalCheck(t, p.out.CopyToHost(out))
				for i := 0; i < physicalCapacity; i++ {
					wantState, wantMid, wantOut := uint32(i), uint32(0), uint32(0)
					if i < 5*physicalBlock {
						wantState++
						wantMid = uint32(i + 3)
						wantOut = 2 * wantMid
					}
					if reverse && i < 2*physicalBlock {
						wantMid++
					}
					if state[i] != wantState || mid[i] != wantMid || out[i] != wantOut {
						t.Fatalf("output[%d]=%d/%d/%d want=%d/%d/%d", i, state[i], mid[i], out[i], wantState, wantMid, wantOut)
					}
				}
			})
		}
	}
}
