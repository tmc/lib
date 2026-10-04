//go:build !windows

package moonep

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
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

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func dedupFixture(width int) DedupSpec {
	return DedupSpec{Tokens: 2, Width: width, K: 3, Slots: 8, Groups: [][3]int32{{0, 0, 2}, {4, 2, 2}}, DupLoffs: []int32{1, 2, 5, 6}, Counts: [2]int32{2, 4}, RouteSlots: []int32{0, 1, 2, 4, 5, 6}, ExpertIDs: []int32{0, 1, 2, 2, 0, 1}, Padding: [][2]int32{{3, 4}, {7, 8}}, SourceID: "local-fixed-route-source-v1", RankMap: []int32{0}, BufferGeneration: 1}
}

func dedupFixtureInput(s DedupSpec, epoch int) ([]float32, []float32) {
	x := make([]float32, s.Slots*s.Width)
	w := make([]float32, s.Slots)
	for i := range x {
		x[i] = math.Float32frombits(0x7fc07902)
	}
	secondary := map[int32]bool{}
	for _, d := range s.DupLoffs[:s.Counts[1]] {
		secondary[d] = true
	}
	for i, slot := range s.RouteSlots {
		w[slot] = float32(i+1) * 0.1875
		if !secondary[slot] {
			for c := 0; c < s.Width; c++ {
				x[int(slot)*s.Width+c] = float32(epoch*7+i/s.K+1) + float32(c)*0.03125
			}
		}
	}
	return x, w
}

// Reference derives each weighted route directly from its original token.
// Group traversal is used only to place intermediate expected reductions.
func dedupReference(s DedupSpec, epoch int, weights []float32) DedupResult {
	out := DedupResult{make([]float32, s.Slots*s.Width), make([]float32, s.Slots*s.Width), make([]float32, s.Slots*s.Width), make([]float32, s.Tokens*s.Width)}
	owner := map[int32]int32{}
	for _, g := range s.Groups[:s.Counts[0]] {
		owner[g[0]] = g[0]
		for _, d := range s.DupLoffs[g[1] : g[1]+g[2]] {
			owner[d] = g[0]
		}
	}
	sums := make([]float64, len(out.Reduced))
	combined := make([]float64, len(out.Combined))
	for r, slot := range s.RouteSlots {
		p, ok := owner[slot]
		if !ok {
			p = slot
		}
		token := r / s.K
		expert := float64(s.ExpertIDs[r])
		for c := 0; c < s.Width; c++ {
			x := float64(float32(epoch*7+token+1) + float32(c)*0.03125)
			v := (x*(expert+1) + expert*.125) * float64(weights[slot])
			out.Expanded[int(slot)*s.Width+c] = float32(x)
			out.Expert[int(slot)*s.Width+c] = float32(v)
			sums[int(p)*s.Width+c] += v
			combined[token*s.Width+c] += v
		}
	}
	for i, x := range sums {
		out.Reduced[i] = float32(x)
	}
	for i, x := range combined {
		out.Combined[i] = float32(x)
	}
	return out
}

func dedupCompare(got, want DedupResult, source, target, inputDigest string) error {
	for i, pair := range [][2][]float32{{got.Expanded, want.Expanded}, {got.Expert, want.Expert}, {got.Reduced, want.Reduced}, {got.Combined, want.Combined}} {
		policy := cudatest.CUDAOutputPolicy{Absolute: 3e-6, Relative: 3e-5, NaN: "reject", Infinity: "reject", Zero: "equal", Elements: len(pair[1]), Target: target, DType: "fp32", MathMode: "sequential fp32 fmad=false", SourceDigest: fmt.Sprintf("%x", sha256.Sum256([]byte(source))), InputDigest: inputDigest}
		if e := cudatest.CompareCUDAOutputFP32(pair[0], pair[1], policy); e != nil {
			return fmt.Errorf("stage %d: %w", i, e)
		}
	}
	return nil
}

func dedupInputDigest(s DedupSpec, epoch int) string {
	x, w := dedupFixtureInput(s, epoch)
	bits := func(xs []float32) []uint32 {
		out := make([]uint32, len(xs))
		for i, x := range xs {
			out[i] = math.Float32bits(x)
		}
		return out
	}
	raw, _ := json.Marshal(struct {
		Spec           DedupSpec
		Input, Weights []uint32
	}{s, bits(x), bits(w)})
	return fmt.Sprintf("%x", sha256.Sum256(raw))
}
func dedupJoinedSource(k [4]DedupKernel) string {
	out := ""
	for _, s := range k {
		out += s.Source
	}
	return out
}

func dedupBits(xs []float32) []uint32 {
	out := make([]uint32, len(xs))
	for i, x := range xs {
		out[i] = math.Float32bits(x)
	}
	return out
}

type dedupPacketReceipt struct {
	Spec                   DedupSpec
	Epoch                  int
	Receipts               []DedupReceipt
	Expected               DedupResult
	ActualBits             [][]uint32
	Rejected               bool
	Comparison             string
	PlanReuse, InputGuards bool
}

func dedupMakeReceipt(s DedupSpec, epoch int, receipts []DedupReceipt, actual, want DedupResult, rejected bool, comparison string) dedupPacketReceipt {
	return dedupPacketReceipt{s, epoch, receipts, want, [][]uint32{dedupBits(actual.Expanded), dedupBits(actual.Expert), dedupBits(actual.Reduced), dedupBits(actual.Combined)}, rejected, comparison, epoch > 0, true}
}
func TestDedupReceiptNonfiniteBits(t *testing.T) {
	got := DedupResult{Expanded: []float32{math.Float32frombits(0x7fc07902)}, Combined: []float32{float32(math.Inf(-1))}}
	r := dedupMakeReceipt(dedupFixture(1), 0, nil, got, DedupResult{}, true, "nonfinite mismatch")
	b, e := json.Marshal(r)
	if e != nil {
		t.Fatal(e)
	}
	var decoded dedupPacketReceipt
	if e = json.Unmarshal(b, &decoded); e != nil {
		t.Fatal(e)
	}
	if decoded.ActualBits[0][0] != 0x7fc07902 || decoded.ActualBits[3][0] != 0xff800000 {
		t.Fatal("nonfinite bits lost")
	}
}

const dedupHostHeader = `#include <cstdio>
#include <vector>
#define __device__
#define __constant__
#define __global__
struct dim {unsigned int x=1,y=1,z=1;};dim blockIdx,blockDim,threadIdx,gridDim;
`

func dedupHost(t *testing.T, s DedupSpec, kernels [4]DedupKernel, input, weight []float32) DedupResult {
	t.Helper()
	clang, e := exec.LookPath("clang++")
	if e != nil {
		t.Fatal(e)
	}
	dir := t.TempDir()
	source := dedupHostHeader
	// Each stage has identical constant metadata, so share the first definition.
	for i, k := range kernels {
		body := k.Source
		if i > 0 {
			j := strings.Index(body, "extern \"C\"")
			body = body[j:]
		}
		source += body + "\n"
	}
	source += fmt.Sprintf(`int main(){blockDim.x=256;std::vector<float>x(%d),w(%d);if(std::fread(x.data(),4,x.size(),stdin)!=x.size()||std::fread(w.data(),4,w.size(),stdin)!=w.size())return 2;
float poison;unsigned int p=0x7fc07901;__builtin_memcpy(&poison,&p,4);
std::vector<float>a(%d+2,poison),b(%d+2,poison),c(%d+2,poison),d(%d+2,poison);
for(unsigned int i=0;i<%d;++i){blockIdx.x=i/256;threadIdx.x=i%%256;dedup_expand(x.data(),a.data()+1,%d);}
for(unsigned int i=0;i<%d;++i){blockIdx.x=i/256;threadIdx.x=i%%256;dedup_expert(a.data()+1,w.data(),b.data()+1,%d);}
for(unsigned int i=0;i<%d;++i){blockIdx.x=i/256;threadIdx.x=i%%256;dedup_reduce(b.data()+1,c.data()+1,%d);}
for(unsigned int i=0;i<%d;++i){blockIdx.x=i/256;threadIdx.x=i%%256;dedup_combine(c.data()+1,d.data()+1,%d);}
for(auto*v:{&a,&b,&c,&d})std::fwrite(v->data(),4,v->size(),stdout);return std::ferror(stdout)?3:0;}`, len(input), len(weight), s.Slots*s.Width, s.Slots*s.Width, s.Slots*s.Width, s.Tokens*s.Width, s.Slots*s.Width, s.Slots*s.Width, s.Slots*s.Width, s.Slots*s.Width, s.Slots*s.Width, s.Slots*s.Width, s.Tokens*s.Width, s.Tokens*s.Width)
	cu := filepath.Join(dir, "host.cc")
	bin := filepath.Join(dir, "host")
	if e = os.WriteFile(cu, []byte(source), 0600); e != nil {
		t.Fatal(e)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, clang, "-O0", "-ffp-contract=off", cu, "-o", bin)
	if raw, e := cmd.CombinedOutput(); e != nil {
		t.Fatalf("compile: %v context=%v %s", e, ctx.Err(), raw)
	}
	var in bytes.Buffer
	binary.Write(&in, binary.LittleEndian, input)
	binary.Write(&in, binary.LittleEndian, weight)
	ctx2, cancel2 := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel2()
	cmd = exec.CommandContext(ctx2, bin)
	cmd.Stdin = &in
	raw, e := cmd.Output()
	if e != nil {
		t.Fatalf("run: %v context=%v", e, ctx2.Err())
	}
	reader := bytes.NewReader(raw)
	var out DedupResult
	for i, dst := range []*[]float32{&out.Expanded, &out.Expert, &out.Reduced, &out.Combined} {
		n := s.Slots * s.Width
		if i == 3 {
			n = s.Tokens * s.Width
		}
		xs := make([]float32, n+2)
		if e = binary.Read(reader, binary.LittleEndian, xs); e != nil {
			t.Fatal(e)
		}
		if math.Float32bits(xs[0]) != 0x7fc07901 || math.Float32bits(xs[n+1]) != 0x7fc07901 {
			t.Fatal("guard overwritten")
		}
		*dst = xs[1 : n+1]
	}
	if reader.Len() != 0 {
		t.Fatal("extra output")
	}
	return out
}

func TestDedupExpansionReference(t *testing.T) {
	for _, width := range []int{1, 17, 128} {
		for _, kind := range []string{"groups", "zero", "permuted"} {
			t.Run(fmt.Sprintf("w%d/%s", width, kind), func(t *testing.T) {
				s := dedupFixture(width)
				if kind == "zero" {
					s.Counts = [2]int32{}
					s.Groups = nil
					s.DupLoffs = nil
				}
				if kind == "permuted" {
					s.Groups[0], s.Groups[1] = s.Groups[1], s.Groups[0]
				}
				x, w := dedupFixtureInput(s, 0)
				k, e := DedupSources(s)
				if e != nil {
					t.Fatal(e)
				}
				got := dedupHost(t, s, k, x, w)
				if e = dedupCompare(got, dedupReference(s, 0, w), dedupJoinedSource(k), "host-emulation", dedupInputDigest(s, 0)); e != nil {
					t.Fatal(e)
				}
			})
		}
	}
}

func TestDedupInverseReductionReference(t *testing.T) {
	s := dedupFixture(17)
	m, e := dedupAdmit(s)
	if e != nil {
		t.Fatal(e)
	}
	x, _ := dedupFixtureInput(s, 0)
	expanded := make([]float64, len(x))
	for row, p := range m.primary {
		if p >= 0 {
			for c := 0; c < s.Width; c++ {
				expanded[row*s.Width+c] = float64(x[int(p)*s.Width+c])
			}
		}
	}
	for _, g := range s.Groups[:s.Counts[0]] {
		for c := 0; c < s.Width; c++ {
			v := expanded[int(g[0])*s.Width+c]
			for _, d := range s.DupLoffs[g[1] : g[1]+g[2]] {
				v += expanded[int(d)*s.Width+c]
			}
			if v != float64(g[2]+1)*float64(x[int(g[0])*s.Width+c]) {
				t.Fatal("bare reduction must preserve multiplicity")
			}
		}
	}
	k := mustDedupSources(t, s)
	old := "(input[i]*(float)(expert[row]+1)+(float)expert[row]*0.125f)*weight[row]"
	if !strings.Contains(k[1].Source, old) {
		t.Fatal("identity expert seam did not match")
	}
	k[1].Source = strings.Replace(k[1].Source, old, "input[i]", 1)
	_, w := dedupFixtureInput(s, 0)
	got := dedupHost(t, s, k, x, w)
	for _, g := range s.Groups[:s.Counts[0]] {
		for c := 0; c < s.Width; c++ {
			if got.Reduced[int(g[0])*s.Width+c] != float32(g[2]+1)*x[int(g[0])*s.Width+c] {
				t.Fatal("emitted reduction lost multiplicity")
			}
		}
	}
}

func TestDedupDistinctRouteWeights(t *testing.T) {
	s := dedupFixture(1)
	_, w := dedupFixtureInput(s, 0)
	a := dedupReference(s, 0, w)
	for _, d := range s.DupLoffs {
		w[d] = 0
	}
	b := dedupReference(s, 0, w)
	if a.Combined[0] == b.Combined[0] {
		t.Fatal("secondary weights erased")
	}
}
func TestDedupPadding(t *testing.T) {
	s := dedupFixture(17)
	m, e := dedupAdmit(s)
	if e != nil {
		t.Fatal(e)
	}
	for _, r := range s.Padding {
		for i := r[0]; i < r[1]; i++ {
			if m.primary[i] != -1 {
				t.Fatal("padding routed")
			}
		}
	}
	for _, k := range mustDedupSources(t, s) {
		if e = k.Schema.Validate(); e != nil {
			t.Fatal(e)
		}
	}
}
func mustDedupSources(t *testing.T, s DedupSpec) [4]DedupKernel {
	t.Helper()
	k, e := DedupSources(s)
	if e != nil {
		t.Fatal(e)
	}
	return k
}

func TestDedupInvalidGroups(t *testing.T) {
	for _, c := range []struct {
		name   string
		mutate func(*DedupSpec)
	}{{"count", func(s *DedupSpec) { s.Counts[0] = 9 }}, {"dupcount", func(s *DedupSpec) { s.Counts[1] = 9 }}, {"primary-as-dup", func(s *DedupSpec) { s.DupLoffs[0] = 0 }}, {"two-writers", func(s *DedupSpec) { s.DupLoffs[2] = 1 }}, {"overlap-primary", func(s *DedupSpec) { s.Groups[1][0] = 1 }}, {"offset", func(s *DedupSpec) { s.Groups[0][1] = 9 }}, {"cross-token", func(s *DedupSpec) { s.DupLoffs[0] = 5 }}, {"padding", func(s *DedupSpec) { s.Padding[0] = [2]int32{0, 4} }}, {"remote", func(s *DedupSpec) { s.RankMap = []int32{0, 1} }}, {"generation", func(s *DedupSpec) { s.BufferGeneration = 0 }}, {"overflow", func(s *DedupSpec) { s.Width = int32Max }}, {"route", func(s *DedupSpec) { s.RouteSlots[1] = 0 }}} {
		t.Run(c.name, func(t *testing.T) {
			s := dedupFixture(1)
			c.mutate(&s)
			if _, e := dedupAdmit(s); e == nil {
				t.Fatal("invalid metadata admitted")
			}
		})
	}
}

func TestDedupPlanReuse(t *testing.T) {
	s := dedupFixture(1)
	a := dedupDigest(s)
	s.BufferGeneration++
	if a == dedupDigest(s) {
		t.Fatal("generation omitted")
	}
	for _, mut := range []func(*DedupSpec){func(s *DedupSpec) { s.SourceID = "changed" }, func(s *DedupSpec) { s.Padding[0][1]++ }, func(s *DedupSpec) { s.Width++ }, func(s *DedupSpec) { s.RankMap[0] = 1 }, func(s *DedupSpec) { s.ExpertIDs[0]++ }} {
		s := dedupFixture(1)
		a := dedupDigest(s)
		mut(&s)
		if a == dedupDigest(s) {
			t.Fatal("reuse identity omitted")
		}
	}
}

func TestDedupPlanLifetime(t *testing.T) {
	p := &CommPlan{}
	snap := dedupSnapshot(p)
	release, e := dedupAcquire(p, snap)
	if e != nil {
		t.Fatal(e)
	}
	if p.Close() == nil {
		t.Fatal("early plan free")
	}
	if _, e = dedupAcquire(p, snap); e == nil {
		t.Fatal("incomplete plan reused")
	}
	release()
	release()
	if e = p.Close(); e != nil {
		t.Fatal(e)
	}
	if _, e = dedupAcquire(p, snap); e == nil {
		t.Fatal("closed plan reused")
	}
}

func TestDedupPlanConcurrentBorrow(t *testing.T) {
	p := &CommPlan{}
	snap := dedupSnapshot(p)
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if r, e := dedupAcquire(p, snap); e == nil {
				r()
			}
		}()
	}
	wg.Wait()
	if p.active != 0 {
		t.Fatal("borrow leaked")
	}
}

func TestDedupDrainRecovery(t *testing.T) {
	p := &CommPlan{}
	release, _ := dedupAcquire(p, dedupSnapshot(p))
	fail := true
	freed, released := 0, 0
	d := &DedupInvocation{attempted: true, releases: []func(){release, func() { released++ }}, ops: dedupOps{wait: func(context.Context) error {
		if fail {
			return errors.New("fault")
		}
		return nil
	}}, buffers: []*dedupBuffer{{free: func() error { freed++; return nil }}}}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if d.Wait(ctx) == nil || d.Close() == nil || released != 0 || freed != 0 || p.Close() == nil {
		t.Fatal("failed drain released owners")
	}
	fail = false
	if e := d.Close(); e != nil || released != 1 || freed != 1 {
		t.Fatal("recovered drain", e)
	}
	if e := p.Close(); e != nil {
		t.Fatal(e)
	}
}

func TestDedupInvocationFaults(t *testing.T) {
	for _, name := range []string{"cancel-before-launch", "compile-fault", "launch-fault", "failed-drain"} {
		t.Run(name, func(t *testing.T) {
			s := dedupFixture(1)
			released, freed, closed, launched, waited := 0, 0, 0, 0, 0
			fault := true
			ops := dedupOps{alloc: func(n int) (*dedupBuffer, error) {
				return &dedupBuffer{ptr: 1, free: func() error { freed++; return nil }}, nil
			}, wait: func(context.Context) error {
				waited++
				if name == "failed-drain" && fault {
					return errors.New("drain")
				}
				return nil
			}, compile: func(k DedupKernel) (*dedupLoaded, error) {
				m := &dedupLoaded{close: func() error { closed++; return nil }, launch: func([]driver.DevicePtr, int) error {
					launched++
					if name == "launch-fault" {
						return errors.New("launch")
					}
					return nil
				}}
				if name == "compile-fault" {
					return m, errors.New("compile")
				}
				return m, nil
			}}
			ctx := context.Background()
			if name == "cancel-before-launch" {
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			}
			v, e := dedupRun(ctx, s, 1, 2, ops, []func(){func() { released++ }})
			if e == nil {
				t.Fatal("fault succeeded")
			}
			if name == "failed-drain" {
				if released != 0 || freed != 0 || closed != 0 {
					t.Fatal("failed drain freed owners")
				}
				fault = false
				if e = v.Close(); e != nil {
					t.Fatal(e)
				}
			}
			if released != 1 || freed != 4 {
				t.Fatal("cleanup count", released, freed)
			}
			if name == "cancel-before-launch" && (launched != 0 || waited != 0) {
				t.Fatal("prelaunch cancellation reached device")
			}
			if name == "launch-fault" && (launched != 1 || waited != 1 || closed != 1) {
				t.Fatal("ambiguous launch was not drained")
			}
		})
	}
}

type dedupMutant struct {
	name     string
	stage    int
	old, new string
}

func dedupMutants() []dedupMutant {
	return []dedupMutant{{"missing-expansion", 0, "input[primary[row]*WIDTH+c]", "input[i]"}, {"stale-payload", 0, "input[primary[row]*WIDTH+c]", "0.0f"}, {"secondary-weight", 1, "weight[row]", "(primary[row]==(int)row?weight[row]:0.0f)"}, {"missing-reduction", 2, "if(primary[j]==(int)row)", "if(primary[j]==(int)row&&j==(int)row)"}, {"primary-offset", 0, "input[primary[row]*WIDTH+c]", "input[((primary[row]+1)%SLOTS)*WIDTH+c]"}}
}
func dedupMutate(t *testing.T, k [4]DedupKernel, m dedupMutant) [4]DedupKernel {
	t.Helper()
	s := k[m.stage].Source
	k[m.stage].Source = strings.Replace(s, m.old, m.new, 1)
	if s == k[m.stage].Source {
		t.Fatal("mutant did not match")
	}
	return k
}
func TestDedupSourceMutants(t *testing.T) {
	s := dedupFixture(17)
	x, w := dedupFixtureInput(s, 0)
	k := mustDedupSources(t, s)
	for _, m := range dedupMutants() {
		t.Run(m.name, func(t *testing.T) {
			mutated := dedupMutate(t, k, m)
			got := dedupHost(t, s, mutated, x, w)
			e := dedupCompare(got, dedupReference(s, 0, w), dedupJoinedSource(mutated), "host-emulation", dedupInputDigest(s, 0))
			var mismatch *cudatest.CUDAOutputMismatch
			if !errors.As(e, &mismatch) {
				t.Fatal("source mutant did not produce numerical mismatch", e)
			}
		})
	}
}

func FuzzDedupCounts(f *testing.F) {
	f.Add(int32(2), int32(4))
	f.Add(int32(-1), int32(9))
	f.Fuzz(func(t *testing.T, g, d int32) {
		s := dedupFixture(1)
		s.Counts = [2]int32{g, d}
		_, e := dedupAdmit(s)
		if (g < 0 || g > 2 || d < 0 || d > 4) && e == nil {
			t.Fatal("invalid count")
		}
	})
}
