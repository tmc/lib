package specdecode

import (
	"context"
	"encoding/binary"
	"fmt"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// SpeculativeDecodeController configures a device-side speculative draft
// verification and KV state controller kernel designed to run as the tail node
// of a CUDA Conditional WHILE Graph (cudaGraphConditionalHandle).
//
// In each conditional WHILE iteration, the kernel compares K speculative draft
// token IDs (draft_tokens[0..K-1]) against K+1 greedy/sampled token IDs
// produced by the target model (target_greedy_tokens[0..K]), appends the
// longest matching prefix of length a (0 <= a <= K) plus the bonus token
// target_greedy_tokens[a] (a + 1 tokens total) to output_tokens starting at
// *cur_seq_len, updates *cur_seq_len, accepted_per_step[*step_idx], and
// *step_idx in place on the device without a CPU round-trip, and calls
// cudaGraphSetConditional(cond_handle, keep_going ? 1u : 0u).
type SpeculativeDecodeController struct {
	Name        string
	DraftTokens int // K: speculative draft tokens verified per step (1..64)
	MaxSteps    int // maximum conditional WHILE steps per burst (0 defaults to 64)
}

func (c SpeculativeDecodeController) norm() (SpeculativeDecodeController, error) {
	if c.Name == "" {
		c.Name = "specdecode_cond_controller"
	}
	if !cIdent(c.Name) {
		return c, fmt.Errorf("speculative decode controller: bad name %q", c.Name)
	}
	if c.DraftTokens < 1 || c.DraftTokens > 64 {
		return c, fmt.Errorf("speculative decode controller %s: DraftTokens %d outside 1..64", c.Name, c.DraftTokens)
	}
	if c.MaxSteps == 0 {
		c.MaxSteps = 64
	}
	if c.MaxSteps < 1 || c.MaxSteps > 4096 {
		return c, fmt.Errorf("speculative decode controller %s: MaxSteps %d outside 1..4096", c.Name, c.MaxSteps)
	}
	return c, nil
}

// Grid returns the single-CTA launch grid and 32-thread block dimensions for
// the device-side speculative decode controller.
func (c SpeculativeDecodeController) Grid() (grid, block [3]int) {
	return [3]int{1, 1, 1}, [3]int{32, 1, 1}
}

func (c SpeculativeDecodeController) params() []string {
	return []string{
		"const int32_t* __restrict__ draft_tokens",
		"const int32_t* __restrict__ target_greedy_tokens",
		"int32_t eos_token_id",
		"int32_t max_seq_len",
		"int32_t* __restrict__ output_tokens",
		"int32_t* __restrict__ cur_seq_len",
		"int32_t* __restrict__ accepted_per_step",
		"int32_t* __restrict__ step_idx",
		"cudaGraphConditionalHandle cond_handle",
	}
}

const specDecodeCondPrelude = `#if !defined(EG_CUDA_GRAPH_CONDITIONAL_HANDLE_DEFINED) && !defined(__DRIVER_TYPES_H__)
#define EG_CUDA_GRAPH_CONDITIONAL_HANDLE_DEFINED 1
typedef unsigned long long cudaGraphConditionalHandle;
#if !defined(EG_HOST_EMU) && !defined(EG_NATIVE_COND_SLOT)
extern "C" __device__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);
#endif
#endif
#if defined(EG_NATIVE_COND_SLOT) && !defined(EG_HOST_EMU)
__device__ __forceinline__ void eg_native_set_conditional(cudaGraphConditionalHandle handle, unsigned int value) {
    if (handle != 0ULL) {
        *reinterpret_cast<unsigned int*>(handle) = value;
    }
}
#define cudaGraphSetConditional eg_native_set_conditional
#endif
`

// CUDA returns the device kernel source (without [hopper.Prelude]).
func (c SpeculativeDecodeController) CUDA() (string, error) {
	c, err := c.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(specDecodeCondPrelude)
	fmt.Fprintf(&b, "// %s: device-side speculative decode conditional graph controller (K=%d, max_steps=%d).\n",
		c.Name, c.DraftTokens, c.MaxSteps)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(32) %s(\n    %s) {\n",
		c.Name, strings.Join(c.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int32_t K = %d, MAX_STEPS = %d;\n", c.DraftTokens, c.MaxSteps)
	b.WriteString(`    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int32_t seq = *cur_seq_len;
    int32_t step = *step_idx;
    if (seq >= max_seq_len || step >= MAX_STEPS) {
        cudaGraphSetConditional(cond_handle, 0u);
        return;
    }

    int32_t a = 0;
    for (int32_t i = 0; i < K; ++i) {
        if (draft_tokens[i] == target_greedy_tokens[i]) {
            ++a;
        } else {
            break;
        }
    }

    bool hit_eos = false;
    int32_t accepted_draft = 0;
    for (int32_t i = 0; i <= a; ++i) {
        if (seq >= max_seq_len) {
            break;
        }
        const int32_t tok = (i < a) ? draft_tokens[i] : target_greedy_tokens[a];
        output_tokens[seq++] = tok;
        if (i < a) {
            ++accepted_draft;
        }
        if (eos_token_id >= 0 && tok == eos_token_id) {
            hit_eos = true;
            break;
        }
    }

    accepted_per_step[step] = accepted_draft;
    ++step;
    *cur_seq_len = seq;
    *step_idx = step;

    const unsigned int keep_going = (!hit_eos && seq < max_seq_len && step < MAX_STEPS) ? 1u : 0u;
    cudaGraphSetConditional(cond_handle, keep_going);
}
`)
	return b.String(), nil
}

// Launcher returns a host function
//
//	void Name_launch(<kernel params...>, cudaStream_t stream)
//
// that launches the speculative decode conditional graph controller kernel.
func (c SpeculativeDecodeController) Launcher() (string, error) {
	c, err := c.norm()
	if err != nil {
		return "", err
	}
	params := c.params()
	names := paramNames(params)
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream) {\n", c.Name, strings.Join(params, ", "))
	b.WriteString("    const dim3 grid(1, 1, 1), block(32, 1, 1);\n")
	fmt.Fprintf(&b, "    EG_LAUNCH(%s, grid, block, 0, stream, %s);\n}\n", c.Name, strings.Join(names, ", "))
	return b.String(), nil
}

// Source returns [hopper.Prelude], the kernel, and its launcher.
func (c SpeculativeDecodeController) Source() (string, error) {
	src, err := c.CUDA()
	if err != nil {
		return "", err
	}
	l, err := c.Launcher()
	if err != nil {
		return "", err
	}
	return hopper.Prelude + "\n" + src + "\n" + l, nil
}

// SpeculativeDecodeStep holds the device inputs for one speculative decode
// verification pass: Draft[0..K-1] are the K draft token IDs proposed for the
// step, and TargetGreedy[0..K] are the K+1 greedy/sampled token IDs produced
// by the target model across the root position and the K draft positions.
type SpeculativeDecodeStep struct {
	Draft        []int32 // length K
	TargetGreedy []int32 // length K + 1
}

// SpeculativeDecodeInput holds the initial sequence state and per-step draft
// and target tokens for [SpeculativeDecodeController.Reference] and
// [SpeculativeDecodeController.RunEmulatedLoop].
type SpeculativeDecodeInput struct {
	InitialTokens []int32                 // initial prefix tokens in output_tokens[0..len(InitialTokens)-1]
	Steps         []SpeculativeDecodeStep // speculative verification steps
	EOSTokenID    int32                   // EOS token ID (< 0 disables early EOS termination)
	MaxSeqLen     int                     // capacity of output_tokens buffer
}

// SpeculativeDecodeResult holds the final sequence and step state produced by
// [SpeculativeDecodeController.Reference] or
// [SpeculativeDecodeController.RunEmulatedLoop].
type SpeculativeDecodeResult struct {
	OutputTokens    []int32  // output_tokens[:FinalSeqLen]
	FinalSeqLen     int      // final *cur_seq_len
	AcceptedPerStep []int32  // accepted_per_step[:StepsExecuted]
	StepsExecuted   int      // final *step_idx
	FinalCondition  uint32   // final condition value in cond_handle
	HandleUpdates   int      // number of cudaGraphSetConditional updates
	IterationTrace  []uint32 // condition value after each iteration
}

func (c SpeculativeDecodeController) validateInput(in SpeculativeDecodeInput) (SpeculativeDecodeController, int, error) {
	c, err := c.norm()
	if err != nil {
		return c, 0, err
	}
	if in.MaxSeqLen < 1 {
		return c, 0, fmt.Errorf("speculative decode controller %s: MaxSeqLen %d < 1", c.Name, in.MaxSeqLen)
	}
	if len(in.InitialTokens) > in.MaxSeqLen {
		return c, 0, fmt.Errorf("speculative decode controller %s: len(InitialTokens) %d > MaxSeqLen %d",
			c.Name, len(in.InitialTokens), in.MaxSeqLen)
	}
	if len(in.Steps) == 0 {
		return c, 0, fmt.Errorf("speculative decode controller %s: empty Steps", c.Name)
	}
	for i, st := range in.Steps {
		if len(st.Draft) != c.DraftTokens {
			return c, 0, fmt.Errorf("speculative decode controller %s: step %d len(Draft) = %d, want %d",
				c.Name, i, len(st.Draft), c.DraftTokens)
		}
		if len(st.TargetGreedy) != c.DraftTokens+1 {
			return c, 0, fmt.Errorf("speculative decode controller %s: step %d len(TargetGreedy) = %d, want %d",
				c.Name, i, len(st.TargetGreedy), c.DraftTokens+1)
		}
	}
	effectiveMaxSteps := min(c.MaxSteps, len(in.Steps))
	return c, effectiveMaxSteps, nil
}

// Reference computes the exact host reference for a multi-step speculative
// decode conditional loop.
func (c SpeculativeDecodeController) Reference(in SpeculativeDecodeInput) (*SpeculativeDecodeResult, error) {
	c, effectiveMaxSteps, err := c.validateInput(in)
	if err != nil {
		return nil, err
	}

	outTokens := make([]int32, in.MaxSeqLen)
	copy(outTokens, in.InitialTokens)
	seq := len(in.InitialTokens)
	step := 0
	acceptedPerStep := make([]int32, effectiveMaxSteps)
	var trace []uint32
	cond := uint32(1)
	updates := 0

	for cond != 0 {
		if seq >= in.MaxSeqLen || step >= effectiveMaxSteps {
			cond = 0
			updates++
			trace = append(trace, cond)
			break
		}
		st := in.Steps[step]
		a := 0
		for i := range c.DraftTokens {
			if st.Draft[i] == st.TargetGreedy[i] {
				a++
			} else {
				break
			}
		}
		hitEOS := false
		var acceptedDraft int32
		for i := 0; i <= a; i++ {
			if seq >= in.MaxSeqLen {
				break
			}
			tok := st.TargetGreedy[a]
			if i < a {
				tok = st.Draft[i]
			}
			outTokens[seq] = tok
			seq++
			if i < a {
				acceptedDraft++
			}
			if in.EOSTokenID >= 0 && tok == in.EOSTokenID {
				hitEOS = true
				break
			}
		}
		acceptedPerStep[step] = acceptedDraft
		step++
		if !hitEOS && seq < in.MaxSeqLen && step < effectiveMaxSteps {
			cond = 1
		} else {
			cond = 0
		}
		updates++
		trace = append(trace, cond)
	}

	return &SpeculativeDecodeResult{
		OutputTokens:    slices.Clone(outTokens[:seq]),
		FinalSeqLen:     seq,
		AcceptedPerStep: slices.Clone(acceptedPerStep[:step]),
		StepsExecuted:   step,
		FinalCondition:  cond,
		HandleUpdates:   updates,
		IterationTrace:  trace,
	}, nil
}

// RunEmulatedLoop executes the multi-step speculative decode conditional loop
// on the host emulator via [cudahostemu.RunConditionalWhile] with zero CPU
// round-trips between iterations.
func (c SpeculativeDecodeController) RunEmulatedLoop(ctx context.Context, in SpeculativeDecodeInput) (*SpeculativeDecodeResult, error) {
	c, effectiveMaxSteps, err := c.validateInput(in)
	if err != nil {
		return nil, err
	}
	c.MaxSteps = effectiveMaxSteps

	src, err := c.Source()
	if err != nil {
		return nil, err
	}

	stageKernel := c.Name + "_stage_step"
	var fullSrc strings.Builder
	fullSrc.WriteString(src)
	fmt.Fprintf(&fullSrc, `
__global__ void __launch_bounds__(32) %s(
    const int32_t* __restrict__ all_draft_tokens,
    const int32_t* __restrict__ all_target_tokens,
    const int32_t* __restrict__ step_idx,
    int32_t* __restrict__ draft_tokens,
    int32_t* __restrict__ target_greedy_tokens) {
    constexpr int32_t K = %d, MAX_STEPS = %d;
    const int32_t tid = threadIdx.x;
    const int32_t s = *step_idx;
    if (s < 0 || s >= MAX_STEPS) return;
    if (tid < K) {
        draft_tokens[tid] = all_draft_tokens[s * K + tid];
    }
    if (tid <= K) {
        target_greedy_tokens[tid] = all_target_tokens[s * (K + 1) + tid];
    }
}
`, stageKernel, c.DraftTokens, effectiveMaxSteps)

	K := c.DraftTokens
	allDraft := make([]int32, 0, len(in.Steps)*K)
	allTarget := make([]int32, 0, len(in.Steps)*(K+1))
	for _, st := range in.Steps {
		allDraft = append(allDraft, st.Draft...)
		allTarget = append(allTarget, st.TargetGreedy...)
	}

	outInit := make([]int32, in.MaxSeqLen)
	copy(outInit, in.InitialTokens)

	allDraftBuf := specI32Bytes(allDraft)
	allTargetBuf := specI32Bytes(allTarget)
	draftBuf := specI32Bytes(make([]int32, K))
	targetBuf := specI32Bytes(make([]int32, K+1))
	outBuf := specI32Bytes(outInit)
	curSeqBuf := specI32Bytes([]int32{int32(len(in.InitialTokens))})
	acceptedBuf := specI32Bytes(make([]int32, effectiveMaxSteps))
	stepIdxBuf := specI32Bytes([]int32{0})

	body := []cudahostemu.CondGraphStep{
		{
			Kernel: stageKernel,
			Grid:   cudahostemu.Dim3{X: 1},
			Block:  cudahostemu.Dim3{X: 32},
			Args:   []string{"all_draft_tokens", "all_target_tokens", "step_idx", "draft_tokens", "target_greedy_tokens"},
		},
		{
			Kernel: c.Name,
			Grid:   cudahostemu.Dim3{X: 1},
			Block:  cudahostemu.Dim3{X: 32},
			Args: []string{
				"draft_tokens",
				"target_greedy_tokens",
				"eos_token_id",
				"max_seq_len",
				"output_tokens",
				"cur_seq_len",
				"accepted_per_step",
				"step_idx",
				"cond_handle",
			},
		},
	}

	whileRes, err := cudahostemu.RunConditionalWhile(
		ctx,
		fullSrc.String(),
		cudahostemu.CondWhileConfig{
			Handle:                1,
			InitialCondition:      1,
			MaxIterations:         effectiveMaxSteps + 4,
			RequireExplicitUpdate: true,
			DetectDataRace:        true,
		},
		body,
		cudahostemu.BufferArg("all_draft_tokens", "int32_t", allDraftBuf),
		cudahostemu.BufferArg("all_target_tokens", "int32_t", allTargetBuf),
		cudahostemu.BufferArg("draft_tokens", "int32_t", draftBuf),
		cudahostemu.BufferArg("target_greedy_tokens", "int32_t", targetBuf),
		cudahostemu.Arg{Name: "eos_token_id", Expr: fmt.Sprintf("%d", in.EOSTokenID)},
		cudahostemu.Arg{Name: "max_seq_len", Expr: fmt.Sprintf("%d", in.MaxSeqLen)},
		cudahostemu.BufferArg("output_tokens", "int32_t", outBuf),
		cudahostemu.BufferArg("cur_seq_len", "int32_t", curSeqBuf),
		cudahostemu.BufferArg("accepted_per_step", "int32_t", acceptedBuf),
		cudahostemu.BufferArg("step_idx", "int32_t", stepIdxBuf),
		cudahostemu.CondHandleArg("cond_handle", 1),
	)
	if err != nil {
		return nil, err
	}

	finalSeq := int(specBytesI32(curSeqBuf)[0])
	stepsExec := int(specBytesI32(stepIdxBuf)[0])
	outWords := specBytesI32(outBuf)
	accWords := specBytesI32(acceptedBuf)
	if finalSeq < 0 || finalSeq > len(outWords) {
		return nil, fmt.Errorf("speculative decode controller %s: invalid final cur_seq_len %d (max %d)", c.Name, finalSeq, len(outWords))
	}
	if stepsExec < 0 || stepsExec > len(accWords) {
		return nil, fmt.Errorf("speculative decode controller %s: invalid final step_idx %d (max %d)", c.Name, stepsExec, len(accWords))
	}

	return &SpeculativeDecodeResult{
		OutputTokens:    slices.Clone(outWords[:finalSeq]),
		FinalSeqLen:     finalSeq,
		AcceptedPerStep: slices.Clone(accWords[:stepsExec]),
		StepsExecuted:   stepsExec,
		FinalCondition:  whileRes.FinalCondition,
		HandleUpdates:   whileRes.HandleUpdateCount,
		IterationTrace:  whileRes.IterationTrace,
	}, nil
}

func specI32Bytes(v []int32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(x))
	}
	return b
}

func specBytesI32(b []byte) []int32 {
	v := make([]int32, len(b)/4)
	for i := range v {
		v[i] = int32(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v
}
