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

// An egraphPrefill runs a prompt through the generated prefill plans
// ([egraphPrefillPlans]) up to egraphPrefillRows tokens at a time. Each
// layer's projections run once over all of a chunk's rows as tile gemms,
// rotary embedding appends the chunk's keys and values to the decoder's
// BF16 cache in one launch, and attention runs once per row, since the
// generated attention kernel takes one sequence length for its whole grid.
type egraphPrefill struct {
	rows         int
	attnIn, post [][]egraphLaunch // per layer
	rope         []egraphLaunch   // per layer; positions set per chunk
	qcast        []egraphLaunch
	sdpa         []egraphLaunch // per layer; q, out and seq_len set per row

	tokens    cudamem.Slice[int32]
	positions cudamem.Slice[int32] // positions[i] = i, for i < MaxSeqLen
	seqIDs    cudamem.Slice[int32] // zeros
	x         [2]cudamem.Slice[float32]
	qkv, qbf  cudamem.Slice[uint16]
	q, attn   cudamem.Slice[float32]
	host      []int32
}

func (eg *egraphDecoder) bindPrefill(dev *cuda.Device, cfg *ModelConfig, gpuW *GPUModelWeights) (*egraphPrefill, error) {
	pl := eg.plans.pf
	h := cfg.HiddenSize
	qDim := cfg.NumAttentionHeads * cfg.HeadDim()
	kvDim := cfg.NumKeyValueHeads * cfg.HeadDim()
	pf := &egraphPrefill{rows: pl.rows, host: make([]int32, pl.rows)}
	var err error
	alloc := func(what string, f func() error) {
		if err == nil {
			if err = f(); err != nil {
				err = fmt.Errorf("alloc prefill %s: %w", what, err)
			}
		}
	}
	alloc("tokens", func() (err error) { pf.tokens, err = cudamem.Alloc[int32](dev, pl.rows); return })
	alloc("positions", func() (err error) { pf.positions, err = cudamem.Alloc[int32](dev, cfg.MaxSeqLen); return })
	alloc("seq ids", func() (err error) { pf.seqIDs, err = cudamem.Alloc[int32](dev, pl.rows); return })
	alloc("residual", func() (err error) { pf.x[0], err = cudamem.Alloc[float32](dev, pl.rows*h); return })
	alloc("residual", func() (err error) { pf.x[1], err = cudamem.Alloc[float32](dev, pl.rows*h); return })
	alloc("qkv", func() (err error) { pf.qkv, err = cudamem.Alloc[uint16](dev, pl.rows*(qDim+2*kvDim)); return })
	alloc("q", func() (err error) { pf.qbf, err = cudamem.Alloc[uint16](dev, pl.rows*qDim); return })
	alloc("q", func() (err error) { pf.q, err = cudamem.Alloc[float32](dev, pl.rows*qDim); return })
	alloc("attention", func() (err error) { pf.attn, err = cudamem.Alloc[float32](dev, pl.rows*qDim); return })
	if err != nil {
		return nil, errors.Join(err, pf.free())
	}
	pos := make([]int32, cfg.MaxSeqLen)
	for i := range pos {
		pos[i] = int32(i)
	}
	if err := pf.positions.CopyFromHost(pos); err != nil {
		return nil, errors.Join(fmt.Errorf("upload prefill positions: %w", err), pf.free())
	}
	if err := pf.seqIDs.CopyFromHost(make([]int32, pl.rows)); err != nil {
		return nil, errors.Join(fmt.Errorf("upload prefill seq ids: %w", err), pf.free())
	}
	if err := eg.bindPrefillLaunches(dev, cfg, gpuW, pf); err != nil {
		return nil, errors.Join(err, pf.free())
	}
	return pf, nil
}

func (eg *egraphDecoder) bindPrefillLaunches(dev *cuda.Device, cfg *ModelConfig, gpuW *GPUModelWeights, pf *egraphPrefill) error {
	pl := eg.plans.pf
	cur, next := pf.x[0], pf.x[1]
	for i := range gpuW.Layers {
		lw := &gpuW.Layers[i]
		bufs := map[string]driver.DevicePtr{
			"x":                   cur.Ptr(),
			"attn_norm":           lw.InputNorm.Ptr(),
			"w_qkv":               eg.wqkv[i].Ptr(),
			pl.attnIn.Output.Name: pf.qkv.Ptr(),
			"attn":                pf.attn.Ptr(),
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
		pf.rope = append(pf.rope, egraphLaunch{
			fn: eg.fns[pl.rope.Name], block: [3]uint32{128, 1, 1}, epoch: -1,
			args: []any{pf.qkv.Ptr(), eg.cs.Ptr(), pf.positions.Ptr(), pf.seqIDs.Ptr(), eg.one, pf.positions.Ptr(),
				eg.pageTable, pf.qbf.Ptr(), eg.kc[i].Ptr(), eg.vc[i].Ptr()},
		})
		pf.sdpa = append(pf.sdpa, eg.sdpa[i])
	}
	var err error
	if pf.qcast, err = eg.bind(pl.qcast, map[string]driver.DevicePtr{
		"q":                  pf.qbf.Ptr(),
		pl.qcast.Output.Name: pf.q.Ptr(),
	}); err != nil {
		return err
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

// prefill runs prompt from position 0, filling the decoder's KV cache, and
// leaves the argmax of the last token's logits in sp.TokenOut, which it
// also returns.
func (eg *egraphDecoder) prefill(gpuW *GPUModelWeights, cfg *ModelConfig, sp *GPUScratchpad, gm *GPUModules, sampleFn driver.Function, prompt []int32) (int32, error) {
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
	h := cfg.HiddenSize
	qDim := driver.DevicePtr(4 * cfg.NumAttentionHeads * cfg.HeadDim())
	launch := func(l egraphLaunch) error {
		return cudalaunch.Kernel(l.fn).Grid(l.grid[0], l.grid[1], l.grid[2]).Block(l.block[0], l.block[1], l.block[2]).SharedMem(l.smem).Launch(l.args...)
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
	for start := 0; start < len(prompt); start += pf.rows {
		chunk := prompt[start:min(start+pf.rows, len(prompt))]
		n := len(chunk)
		copy(pf.host, chunk)
		if err := pf.tokens.CopyFromHost(pf.host[:n]); err != nil {
			return 0, fmt.Errorf("upload prompt: %w", err)
		}
		for i := range n {
			if err := cudalaunch.Kernel(gm.embedBF16Fn).Grid(1, 1, 1).Block(256, 1, 1).Launch(
				gpuW.EmbedTokens, pf.tokens.Ptr()+driver.DevicePtr(4*i), pf.x[0].Ptr()+driver.DevicePtr(4*h*i), int32(h)); err != nil {
				return 0, err
			}
		}
		positions := pf.positions.Ptr() + driver.DevicePtr(4*start)
		for i := range cfg.NumHiddenLayers {
			if err := launchRows(pf.attnIn[i], n); err != nil {
				return 0, err
			}
			rope := pf.rope[i]
			rope.args = append([]any(nil), rope.args...)
			rope.args[2], rope.args[5] = positions, positions
			rope.grid = [3]uint32{uint32(n), 1, 1}
			if err := launch(rope); err != nil {
				return 0, err
			}
			if err := launchRows(pf.qcast, n); err != nil {
				return 0, err
			}
			for r := range n {
				a := pf.sdpa[i]
				a.args = append([]any(nil), a.args...)
				a.args[0] = pf.q.Ptr() + driver.DevicePtr(r)*qDim
				a.args[6] = pf.attn.Ptr() + driver.DevicePtr(r)*qDim
				a.args[7] = int32(start + r + 1)
				if err := launch(a); err != nil {
					return 0, err
				}
			}
			if err := launchRows(pf.post[i], n); err != nil {
				return 0, err
			}
		}
		last = pf.x[cfg.NumHiddenLayers%2].Ptr() + driver.DevicePtr(4*h*(n-1))
	}
	head, err := eg.bind(eg.plans.head, map[string]driver.DevicePtr{
		"x":                       last,
		"norm":                    gpuW.Norm.Ptr(),
		"lm_head":                 gpuW.LMHead.Ptr(),
		eg.plans.head.Output.Name: sp.Logits.Ptr(),
	})
	if err != nil {
		return 0, err
	}
	for _, l := range head {
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
	return errors.Join(freeOwned(&pf.tokens), freeOwned(&pf.positions), freeOwned(&pf.seqIDs),
		freeOwned(&pf.x[0]), freeOwned(&pf.x[1]), freeOwned(&pf.qkv), freeOwned(&pf.qbf),
		freeOwned(&pf.q), freeOwned(&pf.attn))
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
