package cudamega

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"reflect"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudamoe"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

func moeCountedProgram(config MoEWeightConfig, target Target) (*Program, error) {
	if err := config.validate(); err != nil {
		return nil, err
	}
	if target != "sm_121" {
		return nil, errors.New("cudamega: counted weight consumer requires sm_121")
	}
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "expert-0", Scope: cudataskgraph.ScopeDevice, Extent: uint64(config.Intermediate)},
		{ID: "expert-1", Scope: cudataskgraph.ScopeDevice, Extent: uint64(config.Intermediate)},
	})
	if err != nil {
		return nil, err
	}
	policy, err := PolicyFor(g)
	if err != nil || policy != CountedPolicy {
		return nil, errors.New("cudamega: expert graph is not counted")
	}
	return newProgram(g, map[string]Operation{
		"expert-0": {Kind: MoEGateUp, moe: &config},
		"expert-1": {Kind: MoEGateUp, moe: &config},
	}, Grid{CTAs: 2, ThreadsPerCTA: 32, Policy: policy, MaxElements: uint32(config.Intermediate)}, target, true)
}

type moeCountedCUDA struct {
	*cudaSource
	config MoEWeightConfig
}

func (c *moeCountedCUDA) Begin(p *Program) error {
	if p != c.p || len(p.steps) != 2 {
		return errors.New("cudamega: different counted weight program")
	}
	c.defs.WriteString("constexpr unsigned task_count=2, grid_ctas=2, grid_threads=32;\n")
	return nil
}

func (c *moeCountedCUDA) Compute(i int, sources []int, op Operation) error {
	if i < 0 || i >= 2 || len(sources) != 1 || sources[0] != -1 || op.Kind != MoEGateUp || op.moe == nil || *op.moe != c.config {
		return errors.New("cudamega: different counted payload")
	}
	// This is the real slab addressing and SiLU(gate)*up payload. The task
	// index selects an expert; it is never a synthetic Add/Multiply opcode.
	fmt.Fprintf(&c.body, `        for (unsigned neuron=threadIdx.x; neuron<intermediate; neuron+=blockDim.x) {
          unsigned long long base=((unsigned long long)layer*2+%du)*(2ULL*intermediate);
          unsigned long long gate_row=base+(neuron/8)*16+neuron%%8, up_row=gate_row+8;
          float gate=0.0f, up=0.0f;
          for (unsigned column=0; column<hidden; ++column) {
            gate=__fadd_rn(gate,__fmul_rn(weight_decode(weights[gate_row*hidden+column]),input[column]));
            up=__fadd_rn(up,__fmul_rn(weight_decode(weights[up_row*hidden+column]),input[column]));
          }
          gate=__fmul_rn(gate,scales[gate_row]);
          up=__fmul_rn(up,scales[up_row]);
          output[%du*intermediate+neuron]=__fmul_rn(gate/__fadd_rn(1.0f,expf(-gate)),up);
        }
`, i, i)
	return nil
}

func (c *moeCountedCUDA) End() error { return nil }

// MoECountedWeightSource emits a real two-expert gate/up graph of rank width
// two. Each task owns one 32-thread CTA and one disjoint Intermediate-element
// output. Both read the same hidden vector and independently retained ready
// expert ranges of the interleave8 E4M3FN slab. No expert-output reduction is
// implied. The graph uses the existing ready-only counted primitive runner.
//
// The translation unit also contains cudamoe's ordinary moe_weight_slab entry.
// Launch that entry once per expert into the corresponding output range for
// the ordinary reference. Source equivalence is not native qualification or
// evidence that two tasks physically overlap. Both compiled artifacts and
// their actual outputs need separate qualification.
func MoECountedWeightSource(config MoEWeightConfig, target Target) (string, error) {
	p, err := moeCountedProgram(config, target)
	if err != nil {
		return "", err
	}
	c := &moeCountedCUDA{cudaSource: &cudaSource{p: p}, config: config}
	r, err := NewRunner(p, c)
	if err != nil {
		return "", err
	}
	if err := r.Run(); err != nil {
		return "", err
	}
	ordinary, _, err := cudamoe.MoEWeightSource(cudamoe.MoEWeightSlabConsumer)
	if err != nil {
		return "", err
	}
	entry := fmt.Sprintf(`extern "C" __global__ void moe_weight_counted_worker(const unsigned char *weights,const float *scales,const float *input,float *output,unsigned *completed,unsigned long long *start_order,volatile unsigned *abort_code,int layer,int expert,unsigned n,unsigned long long epoch) {
  constexpr unsigned intermediate=%d, hidden=%d;
  if (gridDim.x!=2||gridDim.y!=1||gridDim.z!=1||blockDim.x!=32||blockDim.y!=1||blockDim.z!=1) { if(threadIdx.x==0) atomicCAS((unsigned*)abort_code,0u,%du); return; }
  if (n!=intermediate||!epoch||layer<0||layer>=3||layer==1||expert!=0) { if(threadIdx.x==0) atomicCAS((unsigned*)abort_code,0u,%du); return; }
  // Each independent task checks only its own slot, which no other CTA writes.
  __shared__ unsigned entry_bad;
  if (threadIdx.x==0) entry_bad=completed[blockIdx.x]!=0;
  __syncthreads();
  if (entry_bad) { if(threadIdx.x==0) atomicCAS((unsigned*)abort_code,0u,%du); return; }
`, config.Intermediate, config.Hidden, AbortBadLaunch, AbortBadExtent, AbortStaleCompletion)
	return ordinary + c.defs.String() + entry + c.body.String() + "}\n", nil
}

func checkedMoECountedArtifact(config MoEWeightConfig, a *cudakernel.Artifact) (*cudakernel.Artifact, error) {
	source, err := MoECountedWeightSource(config, "sm_121")
	if err != nil {
		return nil, err
	}
	if err := a.Validate(); err != nil {
		return nil, err
	}
	schema, _ := MoEWeightSchema()
	if a.Kind != cudakernel.KindCUBIN || a.Target != "sm_121" || a.EntryPoint != "moe_weight_counted_worker" || !reflect.DeepEqual(a.Schema, schema) || !reflect.DeepEqual(a.CompilerOpts, []string{"--gpu-architecture=sm_121", "--std=c++17", "--fmad=false", "--ftz=false"}) {
		return nil, errors.New("cudamega: counted weight artifact identity differs")
	}
	want, count := fmt.Sprintf("%x", sha256.Sum256([]byte(source))), 0
	for _, dep := range a.Dependencies {
		if dep.Path == "cudamega_moe_counted.cu" {
			count++
			if !dep.Tracked || dep.Digest != want || dep.ErrorMsg != "" {
				return nil, errors.New("cudamega: counted source binding differs")
			}
		}
	}
	if count != 1 {
		return nil, errors.New("cudamega: counted source binding missing or duplicated")
	}
	b := *a
	b.Raw = append([]byte(nil), a.Raw...)
	b.Dependencies = append([]cudakernel.Dependency(nil), a.Dependencies...)
	b.CompilerOpts = append([]string(nil), a.CompilerOpts...)
	b.Schema.Params = append([]cudakernel.Param(nil), a.Schema.Params...)
	return &b, nil
}

// LoadMoECountedWeightConsumer loads the source-bound two-expert worker and
// admits two cooperative CTAs only after querying its compiled occupancy and
// the actual device SM count. Close every nonnil error result before releasing
// the device. Receipt association does not authenticate compiler semantics.
func LoadMoECountedWeightConsumer(dev *cuda.Device, config MoEWeightConfig, a *cudakernel.Artifact) (*MoEWeightConsumer, error) {
	owned, err := checkedMoECountedArtifact(config, a)
	if err != nil {
		return nil, err
	}
	c, err := loadMoEWeightImage(dev, config, owned)
	if c != nil {
		// A nonnil error result owns cleanup, but is not executable. Admit
		// the counted mode only after every additional device check succeeds.
		c.tasks = 0
	}
	if err != nil {
		return c, err
	}
	sms, err := dev.MultiprocessorCount()
	if err != nil {
		return c, err
	}
	cooperative, err := dev.Attribute(driver.DevAttrCooperativeLaunch)
	if err != nil {
		return c, err
	}
	if err := checkMoECountedResidency(c.blocksPerSM, sms, cooperative); err != nil {
		return c, err
	}
	c.tasks = 2
	return c, nil
}

func checkMoECountedResidency(blocks int32, sms, cooperative int) error {
	// loadMoEWeightImage queried this kernel at 32 threads and zero dynamic
	// shared bytes. Do not invent a shared-memory measurement for a nonzero
	// sentinel field in an unrelated host admission structure.
	if blocks <= 0 || sms <= 0 || uint64(sms) > math.MaxUint32 || cooperative == 0 || uint64(blocks)*uint64(sms) < 2 {
		return errors.New("cudamega: two cooperative CTAs exceed measured residency")
	}
	return nil
}

// ExecuteCounted independently borrows both expert generations, submits the
// two-CTA graph and waits for actual stream completion. views must name expert
// zero and one of the same active layer/slab/device. Output has 2*Intermediate
// floats, Completed has two words and StartOrder has five words. Failure or
// cancellation during drain retains both expert leases and all buffers.
func (c *MoEWeightConsumer) ExecuteCounted(ctx context.Context, stream *cuda.Stream, views [2]*cudamoe.MoEWeightView, b MoEWeightBuffers, epoch uint64) error {
	if c == nil {
		return errors.New("cudamega: nil counted weight consumer")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.tasks != 2 || c.pending != nil || c.kernel == nil || c.blocksPerSM < 1 || stream == nil || stream.Device() != c.device || epoch == 0 {
		return errors.New("cudamega: counted consumer unavailable or still owns work")
	}
	if ctx != nil && ctx.Err() != nil {
		return ctx.Err()
	}
	owned := make([]cudamoe.MoEWeightBorrowed, 0, 2)
	releases := make([]func(), 0, 2)
	for _, view := range views {
		value, release, err := view.Borrow(c.device)
		if err != nil {
			for _, r := range releases {
				r()
			}
			return err
		}
		owned = append(owned, value)
		releases = append(releases, release)
	}
	if err := c.submitBorrowed(stream, owned, releases, b, epoch); err != nil {
		return err
	}
	if err := c.wait(ctx); err != nil {
		return err
	}
	return checkMoEPublication(b)
}

func checkMoECountedBorrow(config MoEWeightConfig, dev *cuda.Device, owned []cudamoe.MoEWeightBorrowed) error {
	if len(owned) != 2 {
		return errors.New("cudamega: counted task requires both expert owners")
	}
	for i, value := range owned {
		if err := checkMoEBorrow(config, dev, value); err != nil {
			return err
		}
		d, first := value.Descriptor, owned[0].Descriptor
		if d.Expert != i || d.Layer != first.Layer || d.WeightBase != first.WeightBase || d.ScaleBase != first.ScaleBase {
			return errors.New("cudamega: counted expert slab, layer or ordering differs")
		}
	}
	return nil
}
