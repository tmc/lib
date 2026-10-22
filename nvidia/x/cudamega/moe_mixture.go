package cudamega

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudamoe"
)

// MoEOrdinarySource emits the actual ordinary slab expert entry and a
// separate ordinary routing-combine entry for p. Launch moe_weight_slab once
// for each gate/up step into its Storage range, then moe_mixture_ordinary
// for each combine step, in program order. The latter takes scratch, routing
// and the step index. No persistent worker or scheduling primitive is used.
// This is source, not a compiler, launch or device qualification receipt.
func (p *Program) MoEOrdinarySource() (string, error) {
	if p == nil || !p.hasTypedMoE() {
		return "", fmt.Errorf("ordinary MoE source requires a typed program")
	}
	cfg := p.steps[0].Operation.typedMoE.Config
	layout, err := p.Storage(uint64(cfg.Intermediate))
	if err != nil {
		return "", err
	}
	ordinary, _, err := cudamoe.MoEWeightSource(cudamoe.MoEWeightSlabConsumer)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(ordinary)
	fmt.Fprintf(&b, `
extern "C" __global__ void moe_mixture_ordinary(float *scratch,const float *routing,unsigned task) {
  unsigned e=blockIdx.x*blockDim.x+threadIdx.x;
  if(gridDim.y!=1||gridDim.z!=1||blockDim.y!=1||blockDim.z!=1||e>=%du) return;
  switch(task) {
`, cfg.Intermediate)
	for i, s := range p.steps {
		if s.Operation.Kind != MoECombine {
			continue
		}
		fmt.Fprintf(&b, "  case %d: {\n    float sum=0.0f;\n", i)
		for _, d := range p.inputSources(i) {
			expert := p.steps[d].Operation.typedMoE.Expert
			fmt.Fprintf(&b, "    sum=__fadd_rn(sum,__fmul_rn(routing[%d],scratch[%d+e]));\n", expert, layout[d].Offset)
		}
		fmt.Fprintf(&b, "    scratch[%d+e]=sum; return;\n  }\n", layout[i].Offset)
	}
	b.WriteString("  default: return;\n  }\n}\n")
	return b.String(), nil
}

type moeMixtureCUDA struct {
	*cudaSource
	layout []TaskStorage
}

func (c *moeMixtureCUDA) Begin(p *Program) error {
	if p != c.p || !p.hasTypedMoE() {
		return fmt.Errorf("different typed mixture program")
	}
	var err error
	c.layout, err = p.Storage(uint64(p.grid.MaxElements))
	if err != nil {
		return err
	}
	fmt.Fprintf(&c.defs, "constexpr unsigned task_count=%d, grid_ctas=%d, grid_threads=%d;\n", len(p.steps), p.grid.CTAs, p.grid.ThreadsPerCTA)
	if p.needSpin {
		fmt.Fprintf(&c.defs, "constexpr unsigned long long spin_limit=%dull;\n", p.grid.spinLimit())
	}
	used := make([]bool, len(p.steps))
	for _, s := range p.steps {
		for _, dep := range s.Deps {
			used[dep] = true
		}
	}
	for i, yes := range used {
		if yes {
			fmt.Fprintf(&c.defs, "constexpr unsigned producers_%d=%du;\n", i, p.steps[i].Owner.Count())
		}
	}
	return nil
}

func (c *moeMixtureCUDA) Compute(i int, sources []int, op Operation) error {
	d, ok := op.MoE()
	if !ok || i < 0 || i >= len(c.p.steps) {
		return fmt.Errorf("unbound typed mixture operation")
	}
	s := c.p.steps[i]
	fmt.Fprintf(&c.body, "        for(unsigned e=(blockIdx.x-%du)*blockDim.x+threadIdx.x;e<%du;e+=%du*blockDim.x) {\n", s.Owner.Lo, d.Config.Intermediate, s.Owner.Count())
	switch d.Kind {
	case MoEGateUp:
		if len(sources) != 0 {
			return fmt.Errorf("expert has graph data operands")
		}
		fmt.Fprintf(&c.body, `          unsigned long long row=((unsigned long long)layer*2+%du)*%du+(e/8)*16+e%%8;
          float gate=0.0f,up=0.0f;
          for(unsigned h=0;h<%du;h++) {
            gate=__fadd_rn(gate,__fmul_rn(weight_decode(weights[row*%du+h]),hidden[h]));
            up=__fadd_rn(up,__fmul_rn(weight_decode(weights[(row+8)*%du+h]),hidden[h]));
          }
          gate=__fmul_rn(gate,scales[row]);
          up=__fmul_rn(up,scales[row+8]);
          scratch[%d+e]=__float_as_int(__fmul_rn(gate/__fadd_rn(1.0f,expf(-gate)),up));
`, d.Expert, 2*d.Config.Intermediate, d.Config.Hidden, d.Config.Hidden, d.Config.Hidden, c.layout[i].Offset)
	case MoECombine:
		c.body.WriteString("          float result=0.0f;\n")
		for _, dep := range sources {
			expert := c.p.steps[dep].Operation.typedMoE.Expert
			fmt.Fprintf(&c.body, "          result=__fadd_rn(result,__fmul_rn(__int_as_float(scratch[%d+e]),routing[%d]));\n", c.layout[dep].Offset, expert)
		}
		fmt.Fprintf(&c.body, "          scratch[%d+e]=__float_as_int(result);\n", c.layout[i].Offset)
	default:
		return fmt.Errorf("unsupported typed mixture opcode")
	}
	c.body.WriteString("        }\n")
	return nil
}

func (c *moeMixtureCUDA) End() error { return nil }

func (p *Program) moeMixtureSource() (string, error) {
	ordinary, err := p.MoEOrdinarySource()
	if err != nil {
		return "", err
	}
	c := &moeMixtureCUDA{cudaSource: &cudaSource{p: p}}
	r, err := NewRunner(p, c)
	if err != nil {
		return "", err
	}
	if err := r.Run(); err != nil {
		return "", err
	}
	entry := fmt.Sprintf(`
extern "C" __global__ void moe_mixture_worker(const unsigned char *weights,const float *scales,const float *hidden,const float *routing,unsigned *scratch,unsigned *completed,unsigned long long *start_order,volatile unsigned *abort_code,int layer,unsigned n,unsigned long long epoch) {
  if(gridDim.x!=grid_ctas||gridDim.y!=1||gridDim.z!=1||blockDim.x!=grid_threads||blockDim.y!=1||blockDim.z!=1) { if(threadIdx.x==0) atomicCAS((unsigned*)abort_code,0u,%du); return; }
  if(n!=%du||!epoch||(layer!=0&&layer!=2)) { if(threadIdx.x==0) atomicCAS((unsigned*)abort_code,0u,%du); return; }
`, AbortBadLaunch, p.grid.MaxElements, AbortBadExtent)
	return "#include <stddef.h>\n" + ordinary + c.defs.String() + entry + c.body.String() + "}\n", nil
}
