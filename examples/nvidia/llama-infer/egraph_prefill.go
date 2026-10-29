package main

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// embedGatherCUDA is the prefill embedding lookup: one block per prompt
// row widens that token's BF16 table row to FP32. It is hand-written, not
// generated; the generator has no gather term.
const (
	embedGatherName = "llama_embed_gather"
	embedGatherCUDA = `// llama_embed_gather: hand-written, not generated.
extern "C" __global__ void llama_embed_gather(
    const uint16_t* __restrict__ table, const int32_t* __restrict__ tokens,
    float* __restrict__ out, int hidden) {
    const uint16_t* src = table + (size_t)tokens[blockIdx.x] * hidden;
    float* dst = out + (size_t)blockIdx.x * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        dst[i] = __uint_as_float((uint32_t)src[i] << 16);
    }
}
`
)

// An egraphPrefill runs a prompt through the generated prefill plans
// ([egraphPrefillPlans]) up to rows tokens per chunk. Per chunk it makes
// one embedding gather launch and, per layer, the attnIn plan's launches,
// one rotary launch that appends the chunk's keys and values to the
// decoder's BF16 cache, one attention launch over all the chunk's rows,
// and the post plan's launches.
type egraphPrefill struct {
	rows, qtok   int
	attnIn, post [][]egraphLaunch // per layer
	rope, attn   []egraphLaunch   // per layer; chunk arguments set per chunk
	gather       driver.Function

	tokens    cudamem.Slice[int32] // the prompt
	positions cudamem.Slice[int32] // positions[i] = i
	seqLens   cudamem.Slice[int32] // per attention group, for the whole prompt
	seqIDs    cudamem.Slice[int32] // zeros
	pages     cudamem.Slice[int32] // identity page table, one row per group
	x         [2]cudamem.Slice[float32]
	qkv, q    cudamem.Slice[uint16]
	attnOut   cudamem.Slice[uint16]
}

// prefillSeqLens returns the attention sequence lengths of a chunk of n
// prompt rows that starts at position start. Group g holds the qtok rows
// from g·qtok; [hopper.MTPPagedAttention] lets it attend every position
// before its first row and, causally, its own rows, so its length is the
// position after its last row. A last group of fewer than qtok rows keeps
// the full length so its real rows keep their positions; its missing rows
// are computed and ignored.
func prefillSeqLens(start, n, qtok int) []int32 {
	lens := make([]int32, (n+qtok-1)/qtok)
	for g := range lens {
		lens[g] = int32(start + (g+1)*qtok)
	}
	return lens
}

// bindPrefill fills the prefill tables newEgraphDecoder allocated in eg.pf
// and binds its launches.
func (eg *egraphDecoder) bindPrefill(dev *cuda.Device, cfg *ModelConfig, gpuW *GPUModelWeights) error {
	pl, pf := eg.plans.pf, eg.pf
	pos := make([]int32, cfg.MaxSeqLen)
	for i := range pos {
		pos[i] = int32(i)
	}
	if err := pf.positions.CopyFromHost(pos); err != nil {
		return fmt.Errorf("upload prefill positions: %w", err)
	}
	if err := pf.seqIDs.CopyFromHost(make([]int32, pf.seqIDs.Len())); err != nil {
		return fmt.Errorf("upload prefill seq ids: %w", err)
	}
	pages := make([]int32, pf.pages.Len())
	for i := range pages {
		pages[i] = int32(i % pl.attn.MaxPages)
	}
	if err := pf.pages.CopyFromHost(pages); err != nil {
		return fmt.Errorf("upload prefill page table: %w", err)
	}
	// A partial last group reads q rows past the chunk; zero them so the
	// rows it computes and ignores stay finite.
	if err := pf.q.CopyFromHost(make([]uint16, pf.q.Len())); err != nil {
		return fmt.Errorf("zero prefill q: %w", err)
	}

	cur, next := pf.x[0], pf.x[1]
	for i := range gpuW.Layers {
		lw := &gpuW.Layers[i]
		bufs := map[string]driver.DevicePtr{
			"x":                   cur.Ptr(),
			"attn_norm":           lw.InputNorm.Ptr(),
			"w_qkv":               eg.wqkv[i].Ptr(),
			pl.attnIn.Output.Name: pf.qkv.Ptr(),
			"attn":                pf.attnOut.Ptr(),
			"w_o":                 lw.WO.Ptr(),
			"ffn_norm":            lw.PostNorm.Ptr(),
			"w_gu":                eg.wgu[i].Ptr(),
			"w_down":              lw.WDown.Ptr(),
			pl.post.Output.Name:   next.Ptr(),
		}
		if len(eg.bqkv) > 0 {
			bufs["b_qkv"] = eg.bqkv[i].Ptr()
		}
		in, err := eg.bind(pl.attnIn, bufs)
		if err != nil {
			return err
		}
		post, err := eg.bind(pl.post, bufs)
		if err != nil {
			return err
		}
		pf.attnIn = append(pf.attnIn, in)
		pf.post = append(pf.post, post)
		cur, next = next, cur

		// qkv, cos_sin_table, positions, seq_ids, seqlens, kv_positions,
		// page_table, q_out, k_cache, v_cache
		rg, rb := pl.rope.Grid(1)
		pf.rope = append(pf.rope, egraphLaunch{
			fn: eg.fns[pl.rope.Name], grid: [3]uint32{0, uint32(rg[1]), 1}, block: [3]uint32{uint32(rb[0]), 1, 1}, epoch: -1,
			args: []any{pf.qkv.Ptr(), eg.cs.Ptr(), pf.positions.Ptr(), pf.seqIDs.Ptr(), eg.seqLen1.Ptr(), pf.positions.Ptr(),
				eg.pages.Ptr(), pf.q.Ptr(), eg.kc[i].Ptr(), eg.vc[i].Ptr()},
		})
		// q, k_cache, v_cache, page_table, seqlens, draft_mask, out
		_, block := pl.attn.Grid(1)
		pf.attn = append(pf.attn, egraphLaunch{
			fn: eg.fns[pl.attn.Name], block: [3]uint32{uint32(block[0]), uint32(block[1]), uint32(block[2])}, epoch: -1,
			args: []any{pf.q.Ptr(), eg.kc[i].Ptr(), eg.vc[i].Ptr(), pf.pages.Ptr(), pf.seqLens.Ptr(), driver.DevicePtr(0), pf.attnOut.Ptr()},
		})
	}
	// Tile gemms use more dynamic shared memory than a launch gets by
	// default.
	for _, ls := range [][]egraphLaunch{pf.attnIn[0], pf.post[0]} {
		for _, l := range ls {
			if err := cudalaunch.Kernel(l.fn).SharedMem(l.smem).PrepareSharedMem(dev); err != nil {
				return fmt.Errorf("prefill gemm shared memory: %w", err)
			}
		}
	}
	return nil
}

// launches returns the launches one chunk makes, before the head.
func (pf *egraphPrefill) launches() int {
	c := 1
	for i := range pf.attnIn {
		c += len(pf.attnIn[i]) + 2 + len(pf.post[i])
	}
	return c
}

// prefill runs prompt from position 0, filling the decoder's KV cache, and
// leaves the argmax of the last token's logits in sp.TokenOut, which it
// also returns.
func (eg *egraphDecoder) prefill(gpuW *GPUModelWeights, cfg *ModelConfig, sp *GPUScratchpad, sampleFn driver.Function, prompt []int32) (int32, error) {
	pf := eg.pf
	if len(prompt) == 0 {
		return 0, errors.New("e-graph prefill: empty prompt")
	}
	if err := checkPos(len(prompt)-1, cfg.MaxSeqLen); err != nil {
		return 0, err
	}
	for _, tok := range prompt {
		if err := checkToken(tok, cfg.VocabSize); err != nil {
			return 0, err
		}
	}
	// Upload the prompt and every chunk's attention lengths before the
	// first launch.
	if err := pf.tokens.CopyFromHost(prompt); err != nil {
		return 0, fmt.Errorf("upload prompt: %w", err)
	}
	var lens []int32
	var groupStart []int
	for start := 0; start < len(prompt); start += pf.rows {
		groupStart = append(groupStart, len(lens))
		lens = append(lens, prefillSeqLens(start, min(pf.rows, len(prompt)-start), pf.qtok)...)
	}
	if err := pf.seqLens.CopyFromHost(lens); err != nil {
		return 0, fmt.Errorf("upload attention lengths: %w", err)
	}

	h := cfg.HiddenSize
	launch := func(l egraphLaunch) error {
		k := cudalaunch.Kernel(l.fn).Grid(l.grid[0], l.grid[1], l.grid[2]).Block(l.block[0], l.block[1], l.block[2]).SharedMem(l.smem)
		if eg.programmatic(l.fn) {
			k = k.WithProgrammaticStreamSerialization(true)
		}
		return k.Launch(l.args...)
	}
	launchRows := func(ls []egraphLaunch, n int) error {
		for _, l := range ls {
			if err := launch(l.withRows(n)); err != nil {
				return err
			}
		}
		return nil
	}
	var last driver.DevicePtr
	for c, g0 := range groupStart {
		start := c * pf.rows
		n := min(pf.rows, len(prompt)-start)
		groups := (n + pf.qtok - 1) / pf.qtok
		if err := launch(egraphLaunch{fn: pf.gather, grid: [3]uint32{uint32(n), 1, 1}, block: [3]uint32{256, 1, 1},
			args: []any{gpuW.EmbedTokens, pf.tokens.Ptr() + driver.DevicePtr(4*start), pf.x[0].Ptr(), int32(h)}}); err != nil {
			return 0, err
		}
		positions := pf.positions.Ptr() + driver.DevicePtr(4*start)
		seqLens := pf.seqLens.Ptr() + driver.DevicePtr(4*g0)
		for i := range cfg.NumHiddenLayers {
			if err := launchRows(pf.attnIn[i], n); err != nil {
				return 0, err
			}
			rope := pf.rope[i]
			rope.args = append([]any(nil), rope.args...)
			rope.args[2], rope.args[5] = positions, positions
			rope.grid = [3]uint32{uint32(n), rope.grid[1], 1}
			if err := launch(rope); err != nil {
				return 0, err
			}
			a := pf.attn[i]
			a.args = append([]any(nil), a.args...)
			a.args[4] = seqLens
			a.grid = [3]uint32{uint32(groups), uint32(cfg.NumKeyValueHeads), 1}
			if err := launch(a); err != nil {
				return 0, err
			}
			if err := launchRows(pf.post[i], n); err != nil {
				return 0, err
			}
		}
		last = pf.x[cfg.NumHiddenLayers%2].Ptr() + driver.DevicePtr(4*h*(n-1))
	}
	norm, err := eg.bind(eg.plans.norm, map[string]driver.DevicePtr{
		"x":                       last,
		"norm":                    gpuW.Norm.Ptr(),
		eg.plans.norm.Output.Name: eg.xn.Ptr(),
	})
	if err != nil {
		return 0, err
	}
	head, err := eg.bind(eg.plans.head, map[string]driver.DevicePtr{
		"xn":                      eg.xn.Ptr(),
		"lm_head":                 gpuW.LMHead.Ptr(),
		eg.plans.head.Output.Name: sp.Logits.Ptr(),
	})
	if err != nil {
		return 0, err
	}
	for _, l := range append(norm, head...) {
		if err := launch(l); err != nil {
			return 0, err
		}
	}
	if err := cudalaunch.Kernel(sampleFn).Grid(1, 1, 1).Block(256, 1, 1).Launch(sp.Logits, sp.TokenOut, int32(cfg.VocabSize)); err != nil {
		return 0, err
	}
	if err := driver.CtxSynchronize(); err != nil {
		return 0, err
	}
	if err := sp.TokenOut.CopyToHost(sp.HostTokenOut); err != nil {
		return 0, err
	}
	return sp.HostTokenOut[0], nil
}

func (pf *egraphPrefill) free() error {
	return errors.Join(freeOwned(&pf.tokens), freeOwned(&pf.positions), freeOwned(&pf.seqLens), freeOwned(&pf.seqIDs),
		freeOwned(&pf.pages), freeOwned(&pf.x[0]), freeOwned(&pf.x[1]), freeOwned(&pf.qkv), freeOwned(&pf.q),
		freeOwned(&pf.attnOut))
}

// prefillTiles returns the names of the prefill plans' tile gemms.
func (eg *egraphDecoder) prefillTiles() []string {
	var names []string
	for _, p := range []*hopper.Plan{eg.plans.pf.attnIn, eg.plans.pf.post} {
		for _, k := range p.Kernels {
			if k.Tile != nil {
				names = append(names, k.Tile.Name)
			}
		}
	}
	return names
}
