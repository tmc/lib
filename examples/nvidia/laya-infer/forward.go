package main

// GPU forward pass: the ModernBERT encoder and the decision head, ported from
// laya.c's forward(). The scorer and action head that follow are small enough
// to stay on the host, in head.go.

import (
	"context"
	"fmt"
	"math"

	"unsafe"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// headEps is the epsilon torch's nn.LayerNorm defaults to; the decision head
// and scorer use it regardless of the encoder's configured norm_eps.
const headEps = 1e-5

// flashTile is FlashAttention's query tile, and flashHeadDim the head width it
// is specialized for. Geometry outside those falls back to the three-kernel
// attention path.
const (
	flashTile    = 64
	flashHeadDim = 64
)

// scoreBudget caps the attention score buffer at 256 MB of float32. The scores
// are the only O(T^2) allocation, so a batch too wide for the budget is run
// through attention a few sequences at a time.
const scoreBudget = 64 << 20

// blasScalars holds the alpha and beta values for the batched attention GEMMs.
// cublasGemmStridedBatchedEx takes them as untyped host pointers rather than
// *float32, so they live in one heap allocation whose address stays valid for
// the life of the model instead of on a stack that may move.
type blasScalars struct{ scale, one, zero float32 }

// precision selects how the matrix products are computed. It is the only
// accuracy knob: the residual stream and all elementwise arithmetic are
// float32 in every mode, and the LayerNorm statistics are double as laya.c
// has them, except float32 where half is about to round the normalized row
// to 11 bits. Every mode accumulates its products in float32; what they
// trade is the mantissa the tensor cores see in the operands, and for half
// also the width of the operands in memory.
//
// fast is the default because the answers are the same decisions -- the
// difference lands in the 4th decimal of a probability, well inside the
// spread between a tiled CPU GEMM and cuBLAS that the port already has. exact
// is what reproduces laya.c byte for byte, at the cost of running without
// tf32 tensor cores or the fused attention kernel. half stores the weight
// matrices and the activations that feed a product as IEEE half. The
// checkpoint is already half, so its weights lose nothing; the activations
// lose three bits of mantissa against tf32.
type precision int

const (
	fast  precision = iota // tf32 tensor cores, and the fused attention kernel
	exact                  // float32 throughout, byte-identical to laya.c
	half                   // half operands, f16 tensor cores and fused attention
)

// model is a loaded Laya model ready to answer questions. It is not safe
// for concurrent use.
type model struct {
	cfg    *config
	tok    *tokenizer
	dev    *cuda.Device
	stream *cuda.Stream
	k      *kernels
	w      *weights
	blas   cublas.Handle
	scalar *blasScalars
	prec   precision
	fused  bool // run attention as one fused kernel instead of QK/softmax/PV

	// scratch is the device memory every pass carves its working set out
	// of. It grows to the largest batch seen and is kept, so a pass costs
	// no allocation once one as large has run.
	scratch cudamem.Slice[byte]
}

func newModel(dir string, prec precision) (*model, error) {
	return openModel(dir, prec, compileKernels)
}

// openModel loads the model, building its kernels with compile. newModel
// passes compileKernels; the --lines path passes one that keeps the
// disassembly of the module it loads, so a sampled program counter can be
// named.
func openModel(dir string, prec precision, compile func(*cuda.Device) (*kernels, error)) (*model, error) {
	cfg, err := loadConfig(dir)
	if err != nil {
		return nil, err
	}
	tok, err := loadTokenizer(dir)
	if err != nil {
		return nil, err
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		return nil, fmt.Errorf("open CUDA device: %w", err)
	}
	m := &model{cfg: cfg, tok: tok, dev: dev, scalar: &blasScalars{one: 1},
		prec: prec, fused: prec != exact}
	// half has only the fused attention kernel, so every attention in the
	// pass must be one it is specialized for.
	if prec == half && (cfg.HeadDim() != flashHeadDim || cfg.D/cfg.HeadHeads != flashHeadDim) {
		dev.Close()
		return nil, fmt.Errorf("half precision needs %d-wide attention heads, have %d and %d",
			flashHeadDim, cfg.HeadDim(), cfg.D/cfg.HeadHeads)
	}
	err = dev.WithContext(func() error {
		if m.k, err = compile(dev); err != nil {
			return err
		}
		if m.w, err = loadWeights(dir, dev, cfg, prec == half); err != nil {
			return err
		}
		if m.stream, err = dev.NewStream(); err != nil {
			return err
		}
		if m.blas, err = cublas.Create(); err != nil {
			return err
		}
		if err := m.blas.SetStream(cublas.Stream(m.stream.Raw())); err != nil {
			return err
		}
		// Weight loading is not what anyone profiles. Opening the range here
		// leaves it to inference, and lets nsys --capture-range=cudaProfilerApi
		// arm itself fully before any kernel runs.
		return driver.ProfilerStart()
	})
	if err != nil {
		dev.Close()
		return nil, err
	}
	return m, nil
}

func (m *model) Close() error {
	return m.dev.WithContext(func() error {
		// Close the range newModel opened. Without this the Go runtime exits
		// through exit_group, the injected profiler never runs its C atexit
		// handler, and the report comes out empty.
		if err := driver.ProfilerStop(); err != nil {
			return err
		}
		m.blas.Destroy()
		if err := m.scratch.Free(); err != nil {
			return err
		}
		if err := m.stream.Destroy(); err != nil {
			return err
		}
		return m.k.Close()
	})
}

// batch is one padded group of questions evaluated together.
type batch struct {
	T       int     // padded sequence length
	ids     []int32 // [B, T] token ids, zero-padded
	lens    []int32 // [B] real lengths
	types   []int32 // [B] question type per sequence
	markers [][]int32
}

// newBatch pads a set of tokenized questions to a common length.
func newBatch(seqs [][]int32, types []int32, markers [][]int32) *batch {
	b := &batch{types: types, markers: markers, lens: make([]int32, len(seqs))}
	for i, s := range seqs {
		b.lens[i] = int32(len(s))
		b.T = max(b.T, len(s))
	}
	b.ids = make([]int32, len(seqs)*b.T)
	for i, s := range seqs {
		copy(b.ids[i*b.T:], s)
	}
	return b
}

// A tensor is device memory holding float32 or half values, whichever the
// precision stores it as. At most one of f32 and f16 is set.
type tensor struct {
	f32 cudamem.Slice[float32]
	f16 cudamem.Slice[uint16] // IEEE binary16 bit patterns
}

func (t tensor) half() bool { return t.f16.Len() > 0 }

func (t tensor) Len() int {
	if t.half() {
		return t.f16.Len()
	}
	return t.f32.Len()
}

// Ptr returns the device address, which is what a kernel launch passes.
func (t tensor) Ptr() driver.DevicePtr {
	if t.half() {
		return t.f16.Ptr()
	}
	return t.f32.Ptr()
}

func (t tensor) Free() error {
	if t.half() {
		return t.f16.Free()
	}
	return t.f32.Free()
}

// dtype is the tensor's element type as cuBLAS names it.
func (t tensor) dtype() cublas.DataType {
	if t.half() {
		return cublas.R16F
	}
	return cublas.R32F
}

// f32 wraps a float32 slice as a tensor.
func f32(s cudamem.Slice[float32]) tensor { return tensor{f32: s} }

// scratch is the device working set for one batch, as views of
// model.scratch. The residual h, the products added into it and the gathered
// rows are float32 in every precision; the tensors are the operands of a
// matrix product or attention.
type scratch struct {
	ids, lens, types cudamem.Slice[int32]
	h, tmp           cudamem.Slice[float32] // [B*T, d]
	xn, ao           tensor                 // [B*T, d]
	qkv              tensor                 // [B*T, 3d]
	q, k, v          tensor                 // [B, nh, T, hd]
	sc               cudamem.Slice[float32] // [scChunk, nh, T, T] attention scores
	scChunk          int                    // sequences per attention chunk
	wi               tensor                 // [B*T, 2*dff]
	g                tensor                 // [B*T, dff]
	hf               tensor                 // [B*T, head_ff]
	rows             cudamem.Slice[float32] // gathered output rows
	rowIdx           cudamem.Slice[int32]
	cg, sg, cl, sl   cudamem.Slice[float32] // RoPE tables, global then local
}

// A carver hands out consecutive views of buf, each aligned to 256 bytes as a
// driver allocation would be. With a zero buf it hands out nothing and only
// counts the bytes it would need.
type carver struct {
	buf cudamem.Slice[byte]
	off int
	err error
}

func carve[T any](c *carver, n int) cudamem.Slice[T] {
	size := n * int(unsafe.Sizeof(*new(T)))
	off := c.off
	c.off += (size + 255) &^ 255
	if c.buf.Len() == 0 || c.err != nil {
		return cudamem.Slice[T]{}
	}
	v, err := cudamem.Reinterpret[T](c.buf.Slice(off, off+size))
	if err != nil {
		c.err = err
	}
	return v
}

// alloc returns the working set for b, growing m.scratch first if b needs
// more than it holds. The views stay valid until the next alloc.
func (m *model) alloc(b *batch, nRows int) (*scratch, error) {
	var c carver
	m.carveScratch(&c, b, nRows)
	if c.off > m.scratch.Len() {
		if err := m.scratch.Free(); err != nil {
			return nil, err
		}
		m.scratch = cudamem.Slice[byte]{}
		buf, err := cudamem.Alloc[byte](m.dev, c.off)
		if err != nil {
			return nil, fmt.Errorf("allocate %d bytes of device scratch: %w", c.off, err)
		}
		m.scratch = buf
	}
	c = carver{buf: m.scratch}
	s := m.carveScratch(&c, b, nRows)
	return s, c.err
}

// carveScratch lays b's working set out in c.
func (m *model) carveScratch(c *carver, b *batch, nRows int) *scratch {
	cfg := m.cfg
	B, T, d := len(b.lens), b.T, cfg.D
	rows, hd := B*T, cfg.HeadDim()
	s := &scratch{}
	t := func(n int) tensor {
		if m.prec == half {
			return tensor{f16: carve[uint16](c, n)}
		}
		return tensor{f32: carve[float32](c, n)}
	}
	s.ids = carve[int32](c, rows)
	s.lens = carve[int32](c, B)
	s.types = carve[int32](c, B)
	s.rowIdx = carve[int32](c, nRows)
	s.h = carve[float32](c, rows*d)
	s.tmp = carve[float32](c, rows*d)
	s.xn = t(rows * d)
	s.ao = t(rows * d)
	s.qkv = t(rows * 3 * d)
	s.q = t(rows * d)
	s.k = t(rows * d)
	s.v = t(rows * d)
	s.wi = t(rows * 2 * cfg.DFF)
	s.g = t(rows * cfg.DFF)
	s.hf = t(rows * cfg.HeadFF)
	s.rows = carve[float32](c, nRows*d)
	// The scores are the one O(T^2) buffer, and only the unfused attention
	// path reads them.
	if !m.fused || hd != flashHeadDim || d/cfg.HeadHeads != flashHeadDim {
		perSeq := max(cfg.NumHeads, cfg.HeadHeads) * T * T
		s.scChunk = min(max(scoreBudget/perSeq, 1), B)
		s.sc = carve[float32](c, s.scChunk*perSeq)
	}
	for _, r := range []*cudamem.Slice[float32]{&s.cg, &s.sg, &s.cl, &s.sl} {
		*r = carve[float32](c, T*hd/2)
	}
	return s
}

// ropeTables fills cos/sin tables for positions 0..T-1, computed in float64 and
// rounded once, as laya.c does.
func ropeTables(T, hd int, theta float64) (cs, sn []float32) {
	half := hd / 2
	cs, sn = make([]float32, T*half), make([]float32, T*half)
	for j := range half {
		inv := 1.0 / math.Pow(theta, float64(2*j)/float64(hd))
		for t := range T {
			cs[t*half+j] = float32(math.Cos(float64(t) * inv))
			sn[t*half+j] = float32(math.Sin(float64(t) * inv))
		}
	}
	return cs, sn
}

// gemm computes C[M,N] = A[M,K] * W[N,K]^T for row-major operands. cuBLAS is
// column-major, so the row-major product is expressed as its transpose:
// C^T[N,M] = W[N,K] * A^T[K,M], which needs no data movement. A and W have
// one element type; C may be float32 when they are half.
func (m *model) gemm(c, a, w tensor, M, N, K int) error {
	return m.gemmBeta(c, a, w, M, N, K, &m.scalar.zero)
}

// gemmAcc adds A[M,K] * W[N,K]^T into C[M,N] instead of overwriting it, so a
// residual add rides along in the GEMM's epilogue. C must not alias A or W.
func (m *model) gemmAcc(c, a, w tensor, M, N, K int) error {
	return m.gemmBeta(c, a, w, M, N, K, &m.scalar.one)
}

// gemmBeta is gemm with a caller-chosen beta. cuBLAS takes alpha and beta by
// address, so they come from m.scalar, whose lifetime matches the model.
//
// This is GemmEx rather than SgemmEx so that the operand precision arrives the
// same way it does for the batched attention products, as the compute type in
// computeType, rather than as a math mode set once on the handle.
func (m *model) gemmBeta(c, a, w tensor, M, N, K int, beta *float32) error {
	return m.blas.GemmEx(cublas.OpT, cublas.OpN, int32(N), int32(M), int32(K),
		uintptr(unsafe.Pointer(&m.scalar.one)),
		uintptr(w.Ptr()), w.dtype(), int32(K),
		uintptr(a.Ptr()), a.dtype(), int32(K),
		uintptr(unsafe.Pointer(beta)),
		uintptr(c.Ptr()), c.dtype(), int32(N),
		m.computeType(), cublas.GemmDefault)
}

// attnScores computes S[bh,i,j] = dot(Q[bh,i,:], K[bh,j,:]) / sqrt(hd) for nbh
// consecutive (sequence, head) pairs. Q and K are [T, hd] row-major per pair,
// which cuBLAS reads column-major as their transposes, so the row-major product
// Q K^T is the column-major product K Q^T.
func (m *model) attnScores(sc, q, k cudamem.Slice[float32], nbh, T, hd int) error {
	m.scalar.scale = float32(1 / math.Sqrt(float64(hd)))
	return m.blas.GemmStridedBatchedEx(cublas.OpT, cublas.OpN,
		int32(T), int32(T), int32(hd),
		uintptr(unsafe.Pointer(&m.scalar.scale)),
		uintptr(k.Ptr()), cublas.R32F, int32(hd), int64(T*hd),
		uintptr(q.Ptr()), cublas.R32F, int32(hd), int64(T*hd),
		uintptr(unsafe.Pointer(&m.scalar.zero)),
		uintptr(sc.Ptr()), cublas.R32F, int32(T), int64(T*T),
		int32(nbh), m.computeType(), cublas.GemmDefault)
}

// attnContext computes out[i, h*hd+c] = sum_j S[h,i,j] * V[h,j,c] for one
// sequence's heads, writing token-major into a [T, d] row-major block so the
// output projection is a plain row GEMM. The batch runs over heads, whose
// outputs are hd columns apart within a row of stride d.
func (m *model) attnContext(out, v, sc cudamem.Slice[float32], heads, T, hd, d int) error {
	return m.blas.GemmStridedBatchedEx(cublas.OpN, cublas.OpN,
		int32(hd), int32(T), int32(T),
		uintptr(unsafe.Pointer(&m.scalar.one)),
		uintptr(v.Ptr()), cublas.R32F, int32(hd), int64(T*hd),
		uintptr(sc.Ptr()), cublas.R32F, int32(T), int64(T*T),
		uintptr(unsafe.Pointer(&m.scalar.zero)),
		uintptr(out.Ptr()), cublas.R32F, int32(d), int64(hd),
		int32(heads), m.computeType(), cublas.GemmDefault)
}

// computeType is the accumulation mode for every matrix product in the pass.
// The accumulator is float32 in all three: what fast gives up is the
// operands, which the tensor cores round to tf32, and what half gives up is
// already gone by the time the operands are in memory.
//
// There is no bfloat16 mode here because asking for one changes nothing.
// Compute32FFast16BF permits cuBLAS to narrow the operands to bfloat16 or to
// tf32, as it prefers; with float32 operands in memory on sm_121 it selects
// the same nvjet tf32 kernels this returns, and the pass produces identical
// bytes in identical time. Reaching the bfloat16 tensor cores means storing
// the operands as bfloat16, not relabelling the compute type.
func (m *model) computeType() cublas.ComputeType {
	if m.prec == fast {
		return cublas.Compute32FFastTF32
	}
	return cublas.Compute32F
}

// grid returns a 1-D grid covering n elements at the given block size.
func grid(n, block int) uint32 {
	return uint32(min((n+block-1)/block, 65535))
}

func nullPtr(s cudamem.Slice[float32]) driver.DevicePtr {
	if s.Len() == 0 {
		return driver.DevicePtr(0)
	}
	return s.Ptr()
}

// pass is one prepared forward pass: a padded batch, its device scratch and the
// rows to read back. Launching the same pass twice recomputes the same answer,
// which is what a profiler's replay needs. A pass is valid until the next
// prepare, which reuses its scratch.
type pass struct {
	b      *batch
	s      *scratch
	rowIdx []int32
}

// prepare lays out device scratch for b and uploads the inputs that do not
// change between launches.
func (m *model) prepare(b *batch) (*pass, error) {
	c := m.cfg
	T, hd := b.T, c.HeadDim()

	// One gathered row per option marker, plus the CLS row of each sequence.
	rowIdx := make([]int32, 0, len(b.ids))
	for i := range b.markers {
		for _, mk := range b.markers[i] {
			rowIdx = append(rowIdx, int32(i*T)+mk)
		}
		rowIdx = append(rowIdx, int32(i*T))
	}
	for _, id := range b.ids {
		if id < 0 || int(id) >= c.Vocab {
			return nil, fmt.Errorf("token id %d outside embedding table", id)
		}
	}

	s, err := m.alloc(b, len(rowIdx))
	if err != nil {
		return nil, err
	}
	cg, sg := ropeTables(T, hd, c.ThetaGlobal)
	cl, sl := ropeTables(T, hd, c.ThetaLocal)
	for _, e := range []struct {
		dst cudamem.Slice[float32]
		src []float32
	}{{s.cg, cg}, {s.sg, sg}, {s.cl, cl}, {s.sl, sl}} {
		if err := e.dst.CopyFromHost(e.src); err != nil {
			return nil, err
		}
	}
	for _, e := range []struct {
		dst cudamem.Slice[int32]
		src []int32
	}{{s.ids, b.ids}, {s.lens, b.lens}, {s.types, b.types}, {s.rowIdx, rowIdx}} {
		if err := e.dst.CopyFromHost(e.src); err != nil {
			return nil, err
		}
	}
	return &pass{b: b, s: s, rowIdx: rowIdx}, nil
}

// launch enqueues the encoder and decision head on m's stream and returns
// without waiting. region, when non-nil, brackets each named phase; profilers
// pass their own, and ordinary callers pass nil.
func (m *model) launch(p *pass, region func(string, func() error) error) error {
	c, s, b := m.cfg, p.s, p.b
	B, T, d, dff := len(b.lens), b.T, c.D, c.DFF
	rows, nh := B*T, c.NumHeads

	in := func(name string, run func() error) error {
		if region == nil {
			return run()
		}
		return region(name, run)
	}
	run := func(fn driver.Function, gx, gy, gz, bx uint32, args ...any) error {
		return cudalaunch.Kernel(fn).Grid(gx, gy, gz).Block(bx).On(m.stream).Launch(args...)
	}
	norm := func(dst tensor, src, w, bias cudamem.Slice[float32], n int, eps float32) error {
		fn := m.k.layerNorm
		if dst.half() {
			fn = m.k.layerNormHalf
		}
		return run(fn, uint32(n), 1, 1, 256,
			src, w, nullPtr(bias), dst, int32(n), int32(d), eps)
	}
	add := func(x, y cudamem.Slice[float32], n int) error {
		return run(m.k.addResid, grid(n, 256), 1, 1, 256, x, y, int32(n))
	}
	bias := func(x tensor, bs cudamem.Slice[float32], n, w int) error {
		fn := m.k.addBias
		if x.half() {
			fn = m.k.addBiasHalf
		}
		return run(fn, grid(n*w, 256), 1, 1, 256, x, bs, int32(n), int32(w))
	}
	attend := func(useRope bool, heads, win int, cs, sn cudamem.Slice[float32]) error {
		rope := int32(0)
		if useRope {
			rope = 1
		}
		whd := d / heads
		split, flash := m.k.splitQKV, m.k.flashAttn
		if s.qkv.half() {
			split, flash = m.k.splitQKVHalf, m.k.flashAttnHalf
		}
		if err := in("split", func() error {
			return run(split, uint32(T), uint32(heads), uint32(B), 64,
				s.qkv, nullPtr(cs), nullPtr(sn), s.q, s.k, s.v,
				int32(B), int32(T), int32(d), int32(heads), int32(whd), rope)
		}); err != nil {
			return err
		}
		if m.fused && whd == flashHeadDim {
			return in("sdpa", func() error {
				return run(flash, uint32((T+flashTile-1)/flashTile), uint32(heads), uint32(B), 128,
					s.q, s.k, s.v, s.lens, s.ao,
					int32(B), int32(T), int32(heads), int32(win),
					float32(1/math.Sqrt(float64(whd))))
			})
		}
		return in("sdpa", func() error {
			seq := heads * T * whd // Q/K/V elements per sequence
			for b0 := 0; b0 < B; b0 += s.scChunk {
				n := min(s.scChunk, B-b0)
				if err := m.attnScores(s.sc,
					s.q.f32.Slice(b0*seq, s.q.Len()), s.k.f32.Slice(b0*seq, s.k.Len()),
					n*heads, T, whd); err != nil {
					return err
				}
				if err := run(m.k.attnSoftmax, uint32(T), uint32(n*heads), 1, 256,
					s.sc, s.lens.Slice(b0, B), int32(T), int32(heads), int32(win)); err != nil {
					return err
				}
				// PV batches over heads, so each sequence needs its own call.
				for b := range n {
					if err := m.attnContext(
						s.ao.f32.Slice((b0+b)*T*d, s.ao.Len()),
						s.v.f32.Slice((b0+b)*seq, s.v.Len()),
						s.sc.Slice(b*heads*T*T, s.sc.Len()),
						heads, T, whd, d); err != nil {
						return err
					}
				}
			}
			return nil
		})
	}

	if err := in("embed", func() error {
		embed := m.k.embed
		if m.w.Emb.half() {
			embed = m.k.embedHalf
		}
		if err := run(embed, uint32(rows), 1, 1, 256, m.w.Emb, s.ids, s.h, int32(rows), int32(d)); err != nil {
			return err
		}
		return norm(f32(s.h), s.h, m.w.EmbNorm, cudamem.Slice[float32]{}, rows, c.Eps)
	}); err != nil {
		return err
	}

	for i := range m.w.Layers {
		L := &m.w.Layers[i]
		x := f32(s.h)
		switch {
		case L.AttnNorm.Len() > 0: // layer 0 has no attn_norm
			if err := norm(s.xn, s.h, L.AttnNorm, cudamem.Slice[float32]{}, rows, c.Eps); err != nil {
				return err
			}
			x = s.xn
		case s.xn.half():
			if err := run(m.k.toHalf, grid(rows*d, 256), 1, 1, 256, s.h, s.xn, int32(rows*d)); err != nil {
				return err
			}
			x = s.xn
		}
		if err := in("enc_qkv", func() error {
			return m.gemm(s.qkv, x, L.Wqkv, rows, 3*d, d)
		}); err != nil {
			return err
		}
		win, cs, sn := -1, s.cg, s.sg
		if c.Local[i] {
			win, cs, sn = c.Window, s.cl, s.sl
		}
		if err := in("enc_attn", func() error {
			return attend(true, nh, win, cs, sn)
		}); err != nil {
			return err
		}
		if err := in("enc_proj", func() error {
			if err := m.gemmAcc(f32(s.h), s.ao, L.Wo, rows, d, d); err != nil {
				return err
			}
			return norm(s.xn, s.h, L.MLPNorm, cudamem.Slice[float32]{}, rows, c.Eps)
		}); err != nil {
			return err
		}
		if err := in("enc_mlp", func() error {
			if err := m.gemm(s.wi, s.xn, L.Wi, rows, 2*dff, d); err != nil {
				return err
			}
			geglu := m.k.geglu
			if s.g.half() {
				geglu = m.k.gegluHalf
			}
			if err := run(geglu, grid(rows*dff, 256), 1, 1, 256, s.wi, s.g, int32(rows), int32(dff)); err != nil {
				return err
			}
			return m.gemmAcc(f32(s.h), s.g, L.Wo2, rows, d, dff)
		}); err != nil {
			return err
		}
	}

	hff := c.HeadFF
	if err := in("head", func() error {
		if err := norm(f32(s.h), s.h, m.w.FinalNorm, cudamem.Slice[float32]{}, rows, c.Eps); err != nil {
			return err
		}
		if err := run(m.k.addTypeEmb, grid(rows*d, 256), 1, 1, 256,
			s.h, m.w.TypeEmb, s.types, int32(B), int32(T), int32(d)); err != nil {
			return err
		}
		for i := range m.w.Head {
			H := &m.w.Head[i]
			if err := norm(s.xn, s.h, H.Norm1W, H.Norm1B, rows, headEps); err != nil {
				return err
			}
			if err := m.gemm(s.qkv, s.xn, H.InW, rows, 3*d, d); err != nil {
				return err
			}
			if err := bias(s.qkv, H.InB, rows, 3*d); err != nil {
				return err
			}
			if err := attend(false, c.HeadHeads, -1, cudamem.Slice[float32]{}, cudamem.Slice[float32]{}); err != nil {
				return err
			}
			if err := m.gemm(f32(s.tmp), s.ao, H.OutW, rows, d, d); err != nil {
				return err
			}
			if err := bias(f32(s.tmp), H.OutB, rows, d); err != nil {
				return err
			}
			if err := add(s.h, s.tmp, rows*d); err != nil {
				return err
			}
			if err := norm(s.xn, s.h, H.Norm2W, H.Norm2B, rows, headEps); err != nil {
				return err
			}
			if err := m.gemm(s.hf, s.xn, H.L1W, rows, hff, d); err != nil {
				return err
			}
			if err := bias(s.hf, H.L1B, rows, hff); err != nil {
				return err
			}
			relu := m.k.relu
			if s.hf.half() {
				relu = m.k.reluHalf
			}
			if err := run(relu, grid(rows*hff, 256), 1, 1, 256, s.hf, int32(rows*hff)); err != nil {
				return err
			}
			if err := m.gemm(f32(s.tmp), s.hf, H.L2W, rows, d, hff); err != nil {
				return err
			}
			if err := bias(f32(s.tmp), H.L2B, rows, d); err != nil {
				return err
			}
			if err := add(s.h, s.tmp, rows*d); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		return err
	}

	return in("gather", func() error {
		return run(m.k.gatherRows, uint32(len(p.rowIdx)), 1, 1, 256,
			s.h, s.rowIdx, s.rows, int32(len(p.rowIdx)), int32(d))
	})
}

// gather waits for the launched work and copies back, for every sequence, the
// option marker rows followed by the CLS row.
func (m *model) gather(p *pass) ([][]float32, error) {
	d := m.cfg.D
	if err := m.stream.Synchronize(context.Background()); err != nil {
		return nil, err
	}
	flat := make([]float32, len(p.rowIdx)*d)
	if err := p.s.rows.CopyToHost(flat); err != nil {
		return nil, err
	}
	var out [][]float32
	for i, n := 0, 0; i < len(p.b.markers); i++ {
		k := len(p.b.markers[i]) + 1
		out = append(out, flat[n*d:(n+k)*d])
		n += k
	}
	return out, nil
}

// forward runs the encoder and decision head over b and returns, for every
// sequence, the option marker rows followed by the CLS row.
func (m *model) forward(b *batch) ([][]float32, error) {
	var out [][]float32
	err := m.dev.WithContext(func() error {
		p, err := m.prepare(b)
		if err != nil {
			return err
		}
		if err := m.launch(p, nil); err != nil {
			return err
		}
		out, err = m.gather(p)
		return err
	})
	return out, err
}
