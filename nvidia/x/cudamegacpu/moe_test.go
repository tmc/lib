package cudamegacpu

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

func moeFixture(i, h, layer int) cudamega.MoEInput {
	c := cudamega.MoEWeightConfig{Layers: 3, Experts: 2, Intermediate: i, Hidden: h}
	rows := 3 * 2 * 2 * i
	in := cudamega.MoEInput{Config: c, Layer: layer, Weights: make([]byte, rows*h), Scales: make([]float32, rows), Hidden: make([]float32, h), Routing: [2]float32{0.25, 0.75}}
	for row := range in.Scales {
		in.Scales[row] = 0.5 + float32(row%11)/32
		for col := range in.Hidden {
			in.Weights[row*h+col] = 0x20 + byte((row*3+col)%16)
		}
	}
	for col := range in.Hidden {
		in.Hidden[col] = float32(col%7+1) / 64
	}
	return in
}

func moeProgram(t *testing.T, c cudamega.MoEWeightConfig, groups, threads uint32) *cudamega.Program {
	t.Helper()
	p, err := cudamega.NewMoEMixtureProgram(c, cudamega.Grid{CTAs: groups, ThreadsPerCTA: threads, Policy: cudamega.CountedPolicy, MaxElements: uint32(c.Intermediate)}, cudamega.SM121a)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func moeExecutor(t *testing.T, groups, threads uint32) *Executor {
	t.Helper()
	e, err := New(cudamega.Admission{ResidentCTAs: groups, WorkerThreads: threads, SharedBytes: 1})
	if err != nil {
		t.Fatal(err)
	}
	return e
}

// ordinaryMoE compiles the actual separately emitted ordinary source. Only
// CUDA spelling and launch indices are supplied by the host harness; no
// persistent-worker source, synchronization erasure, or Go evaluator is used.
func ordinaryMoE(t *testing.T, p *cudamega.Program, in cudamega.MoEInput) []float32 {
	t.Helper()
	source, err := p.MoEOrdinarySource()
	if err != nil {
		t.Fatal(err)
	}
	layout, err := p.Storage(uint64(in.Config.Intermediate))
	if err != nil {
		t.Fatal(err)
	}
	inv, err := p.Invocation(uint64(in.Config.Intermediate))
	if err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	b.WriteString(`#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#define __device__
#define __global__
struct Dim { unsigned x,y,z; };
static Dim blockIdx={0,0,0},blockDim={1,1,1},threadIdx={0,0,0},gridDim={1,1,1};
static float __fadd_rn(float a,float b) { volatile float v=a+b;return v; }
static float __fmul_rn(float a,float b) { volatile float v=a*b;return v; }
static float __int_as_float(unsigned a) { float v;std::memcpy(&v,&a,4);return v; }
static unsigned __float_as_int(float a) { unsigned v;std::memcpy(&v,&a,4);return v; }
`)
	b.WriteString(source)
	b.WriteString("\nint main() {\nstatic const unsigned char weights[]={")
	for _, v := range in.Weights {
		fmt.Fprintf(&b, "%d,", v)
	}
	b.WriteString("};\n")
	for _, a := range []struct {
		name   string
		values []float32
	}{{"scales", in.Scales}, {"hidden", in.Hidden}, {"routing", in.Routing[:]}} {
		fmt.Fprintf(&b, "float %s[]={", a.name)
		for _, v := range a.values {
			fmt.Fprintf(&b, "__int_as_float(%du),", math.Float32bits(v))
		}
		b.WriteString("};\n")
	}
	fmt.Fprintf(&b, "std::vector<float> scratch(%d,0.0f);blockDim.x=%d;\n", inv.ScratchWords, in.Config.Intermediate)
	for i, s := range p.Steps() {
		d, ok := s.Operation.MoE()
		if !ok {
			t.Fatal("ordinary oracle requires actual typed descriptors")
		}
		fmt.Fprintf(&b, "for(threadIdx.x=0;threadIdx.x<%du;++threadIdx.x) {", in.Config.Intermediate)
		if d.Kind == cudamega.MoEGateUp {
			fmt.Fprintf(&b, "moe_weight_slab(weights,scales,hidden,scratch.data()+%d,%d,%d,2,%d,%d);", layout[i].Offset, in.Layer, d.Expert, in.Config.Intermediate, in.Config.Hidden)
		} else {
			fmt.Fprintf(&b, "moe_mixture_ordinary(scratch.data(),routing,%du);", i)
		}
		b.WriteString("}\n")
	}
	b.WriteString("for(float v:scratch) {unsigned bits=__float_as_int(v);if(std::fwrite(&bits,4,1,stdout)!=1)return 2;}return 0;}\n")
	dir := t.TempDir()
	path := filepath.Join(dir, "ordinary.cpp")
	bin := filepath.Join(dir, "ordinary")
	if err := os.WriteFile(path, []byte(b.String()), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "/usr/bin/clang++", "-std=c++17", "-O0", "-ffp-contract=off", "-fno-fast-math", path, "-o", bin)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("compile actual ordinary source: %v\n%s", err, output)
	}
	cmd = exec.CommandContext(ctx, bin)
	output, err := cmd.Output()
	if err != nil {
		t.Fatalf("ordinary source execution: %v", err)
	}
	if len(output) != int(inv.ScratchWords)*4 {
		t.Fatalf("ordinary bytes=%d, want %d", len(output), inv.ScratchWords*4)
	}
	want := make([]float32, len(output)/4)
	for i := range want {
		want[i] = math.Float32frombits(binary.LittleEndian.Uint32(output[i*4:]))
	}
	return want
}

func equalMoE(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i, x := range a {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || math.Abs(float64(x-b[i])) > 2e-6+2e-6*math.Abs(float64(b[i])) {
			return false
		}
	}
	return true
}

func TestMoECPUBackendOrdinarySource(t *testing.T) {
	for _, i := range []int{8, 16} {
		for _, h := range []int{17, 32} {
			for _, layer := range []int{0, 2} {
				t.Run(fmt.Sprintf("I%d_H%d_layer%d", i, h, layer), func(t *testing.T) {
					in := moeFixture(i, h, layer)
					binding, err := cudamega.BindMoE(in)
					if err != nil {
						t.Fatal(err)
					}
					p := moeProgram(t, in.Config, 1, 1)
					want := ordinaryMoE(t, p, in)
					for _, groups := range []uint32{1, 2, 4, 8} {
						p := moeProgram(t, in.Config, groups, 1)
						e := moeExecutor(t, groups, 1)
						got, err := e.RunMoE(p, binding, 1, 0)
						if err != nil {
							t.Fatal(err)
						}
						if got.Abort != 0 || got.Threads != uint64(groups) || !equalMoE(got.Scratch, want) {
							t.Fatalf("%d true one-thread groups: got %+v want %v", groups, got, want)
						}
						if got.StartOrder[len(got.StartOrder)-2] != uint64(groups) || got.StartOrder[len(got.StartOrder)-1] != 1 {
							t.Fatal("entry protocol was not executed by all groups")
						}
					}
					p = moeProgram(t, in.Config, 2, 4)
					got, err := moeExecutor(t, 2, 4).RunMoE(p, binding, 1, 0)
					if err != nil || got.Threads != 8 || !equalMoE(got.Scratch, want) {
						t.Fatalf("two four-thread groups: %+v %v", got, err)
					}
				})
			}
		}
	}
}

func TestMoEDataEdgeChangesOutput(t *testing.T) {
	in := moeFixture(8, 17, 2)
	binding, err := cudamega.BindMoE(in)
	if err != nil {
		t.Fatal(err)
	}
	full := moeProgram(t, in.Config, 2, 1)
	want := ordinaryMoE(t, full, in)
	steps := full.Steps()
	tasks := make([]cudataskgraph.Task, len(steps))
	ops := make(map[string]cudamega.Operation)
	for i, s := range steps {
		tasks[i] = s.Task
		ops[s.Task.ID] = s.Operation
		if s.Operation.Kind == cudamega.MoECombine {
			tasks[i].Inputs = []string{"expert-0"}
			tasks[i].Deps = []string{"expert-0"}
		}
	}
	g, err := cudataskgraph.New(tasks)
	if err != nil {
		t.Fatal(err)
	}
	cut, err := cudamega.NewProgram(g, ops, full.Grid(), full.Target())
	if err != nil {
		t.Fatalf("valid one-expert omission refused instead of executed: %v", err)
	}
	got, err := moeExecutor(t, 2, 1).RunMoE(cut, binding, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if equalMoE(got.Scratch, want) {
		t.Fatal("removing expert-1 data/dependency edge did not change output")
	}
	if !equalMoE(got.Scratch, ordinaryMoE(t, cut, in)) {
		t.Fatal("omitted graph does not match its actual ordinary source")
	}
}

type moeReadyProbe struct {
	held, decided, release chan struct{}
	omit                   bool
	once                   sync.Once
}
type moeReadyWorker struct {
	*worker
	probe *moeReadyProbe
}

func (w *moeReadyWorker) Ready(i, d int, s cudataskgraph.Scope) error {
	if w.probe.omit && i == 2 && d == 1 {
		return nil
	}
	return w.worker.Ready(i, d, s)
}
func (w *moeReadyWorker) Take(i int) (bool, error) {
	ok, err := w.worker.Take(i)
	if i == 2 && !ok {
		w.probe.once.Do(func() { close(w.probe.decided) })
	}
	return ok, err
}
func (w *moeReadyWorker) Compute(i int, s []int, o cudamega.Operation) error {
	if i == 1 {
		close(w.probe.held)
		<-w.probe.release
	}
	err := w.worker.Compute(i, s, o)
	if i == 2 {
		w.probe.once.Do(func() { close(w.probe.decided) })
	}
	return err
}

func TestMoEReadyOrderingControlsPoisonedRead(t *testing.T) {
	in := moeFixture(8, 17, 2)
	binding, err := cudamega.BindMoE(in)
	if err != nil {
		t.Fatal(err)
	}
	p := moeProgram(t, in.Config, 2, 1)
	want := ordinaryMoE(t, p, in)
	for _, omit := range []bool{false, true} {
		t.Run(fmt.Sprint("omitReady=", omit), func(t *testing.T) {
			e := moeExecutor(t, 2, 1)
			probe := &moeReadyProbe{make(chan struct{}), make(chan struct{}), make(chan struct{}), omit, sync.Once{}}
			e.wrap = func(b cudamega.Backend) cudamega.Backend {
				w := b.(*worker)
				r := w.r.layout[1]
				for i := 0; i < r.Elements; i++ {
					w.r.scratch[r.Offset+i] = 0x7fc00001
				}
				return &moeReadyWorker{w, probe}
			}
			control := make(chan error, 1)
			go func() {
				defer close(probe.release)
				for _, c := range []<-chan struct{}{probe.held, probe.decided} {
					select {
					case <-c:
					case <-time.After(5 * time.Second):
						control <- fmt.Errorf("ordering control failed to reach its channel barrier")
						return
					}
				}
				control <- nil
			}()
			got, err := e.RunMoE(p, binding, 1, 0)
			controlErr := <-control
			if err != nil || controlErr != nil {
				t.Fatalf("run=%v control=%v", err, controlErr)
			}
			if equalMoE(got.Scratch, want) == omit {
				t.Fatalf("ordering omission=%v did not discriminate poisoned read: %v", omit, got.Scratch)
			}
		})
	}
}

func TestMoEBindingAndStartupRefusals(t *testing.T) {
	in := moeFixture(8, 17, 0)
	binding, err := cudamega.BindMoE(in)
	if err != nil {
		t.Fatal(err)
	}
	p := moeProgram(t, in.Config, 1, 1)
	e := moeExecutor(t, 1, 1)
	starts := 0
	e.wrap = func(b cudamega.Backend) cudamega.Backend { starts++; return b }
	for _, tc := range []struct {
		name  string
		p     *cudamega.Program
		b     cudamega.MoEBinding
		epoch uint64
	}{
		{"nil program", nil, binding, 1}, {"zero binding", p, cudamega.MoEBinding{}, 1}, {"zero epoch", p, binding, 0}, {"config mismatch", moeProgram(t, cudamega.MoEWeightConfig{Layers: 3, Experts: 2, Intermediate: 16, Hidden: 17}, 1, 1), binding, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := e.RunMoE(tc.p, tc.b, tc.epoch, 0); err == nil {
				t.Fatal("accepted invalid startup")
			}
			if starts != 0 {
				t.Fatal("invalid binding reached worker binding")
			}
		})
	}
	if _, err := e.Run(p, make([]uint32, 8), 1, 0); err == nil {
		t.Fatal("scalar Run accepted typed graph")
	}
	if _, err := Reference(p, make([]uint32, 8)); err == nil {
		t.Fatal("scalar Reference accepted typed graph")
	}
	e.progress = cudamega.NoForwardProgress
	if _, err := e.RunMoE(p, binding, 1, 0); err == nil || starts != 0 {
		t.Fatal("progress was not refused before startup")
	}
	if _, err := New(cudamega.Admission{ResidentCTAs: 1025, WorkerThreads: 1, SharedBytes: 1}); err == nil {
		t.Fatal("unbounded OS thread count accepted")
	}
	if _, err := (*Executor)(nil).RunMoE(p, binding, 1, 0); err == nil {
		t.Fatal("nil executor accepted")
	}
	for _, mutate := range []func(*cudamega.MoEInput){func(x *cudamega.MoEInput) { x.Layer = 1 }, func(x *cudamega.MoEInput) { x.Weights = x.Weights[:1] }, func(x *cudamega.MoEInput) { x.Weights[len(x.Weights)-1] = 127 }, func(x *cudamega.MoEInput) { x.Scales[0] = float32(math.NaN()) }, func(x *cudamega.MoEInput) { x.Hidden[0] = float32(math.Inf(1)) }} {
		x := moeFixture(8, 17, 0)
		mutate(&x)
		if _, err := cudamega.BindMoE(x); err == nil {
			t.Fatal("invalid complete slab/finite/layer binding accepted")
		}
	}
}

func TestMoEImmutableInputsAndAbort(t *testing.T) {
	in := moeFixture(8, 17, 2)
	p := moeProgram(t, in.Config, 1, 1)
	want := ordinaryMoE(t, p, in)
	binding, err := cudamega.BindMoE(in)
	if err != nil {
		t.Fatal(err)
	}
	in.Weights[0] = 127
	in.Hidden[0] = 999
	in.Scales[0] = 999
	in.Routing = [2]float32{0, 0}
	snapshot, err := binding.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	snapshot.Hidden[0] = 999
	snapshot.Weights[0] = 127
	e := moeExecutor(t, 1, 1)
	got, err := e.RunMoE(p, binding, 1, 0)
	if err != nil || !equalMoE(got.Scratch, want) {
		t.Fatalf("binding aliases caller slices: %v %v", got.Scratch, err)
	}
	got.Scratch[0] = 999
	again, err := e.RunMoE(p, binding, 2, 0)
	if err != nil || !equalMoE(again.Scratch, want) {
		t.Fatal("returned result aliases next run")
	}
	aborted, err := e.RunMoE(p, binding, 3, cudamega.AbortHostRequest)
	if err != nil || aborted.Abort != cudamega.AbortHostRequest || aborted.StartOrder[0] != 0 {
		t.Fatalf("preabort executed tasks: %+v %v", aborted, err)
	}
}

type moeFailWorker struct {
	*worker
	cause error
}

func (w *moeFailWorker) Compute(i int, s []int, o cudamega.Operation) error {
	if i == 0 && w.tid == 1 {
		return w.cause
	}
	return w.worker.Compute(i, s, o)
}
func TestMoELaneErrorReleasesBarriers(t *testing.T) {
	in := moeFixture(8, 17, 0)
	b, err := cudamega.BindMoE(in)
	if err != nil {
		t.Fatal(err)
	}
	p := moeProgram(t, in.Config, 1, 4)
	e := moeExecutor(t, 1, 4)
	cause := errors.New("lane failed")
	e.wrap = func(b cudamega.Backend) cudamega.Backend { return &moeFailWorker{b.(*worker), cause} }
	done := make(chan error, 1)
	go func() { _, err := e.RunMoE(p, b, 1, 0); done <- err }()
	select {
	case err := <-done:
		if !errors.Is(err, cause) {
			t.Fatalf("primary lane error lost: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("one lane error stranded a group barrier")
	}
}

// This bounded property checks the decoder/binding boundary and finite
// execution, not numerical equivalence. The ordinary-source test is the
// independent numerical oracle.
func FuzzMoEFiniteBinding(f *testing.F) {
	for _, raw := range []byte{0, 1, 126, 127, 128, 129, 254, 255} {
		f.Add(raw, int8(17), byte(0))
	}
	f.Fuzz(func(t *testing.T, raw byte, hidden int8, layer byte) {
		in := moeFixture(8, 17, int(layer%2)*2)
		for i := range in.Weights {
			in.Weights[i] = raw
		}
		for i := range in.Hidden {
			in.Hidden[i] = float32(hidden) / 128
		}
		b, err := cudamega.BindMoE(in)
		if raw&127 == 127 {
			if err == nil {
				t.Fatal("NaN FP8 storage admitted")
			}
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		p := moeProgram(t, in.Config, 1, 1)
		got, err := moeExecutor(t, 1, 1).RunMoE(p, b, 1, 0)
		if err != nil || got.Abort != 0 || len(got.Scratch) != 24 {
			t.Fatalf("finite binding failed: %+v %v", got, err)
		}
		for _, x := range got.Scratch {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				t.Fatal("finite bounded inputs produced a nonfinite output")
			}
		}
	})
}
