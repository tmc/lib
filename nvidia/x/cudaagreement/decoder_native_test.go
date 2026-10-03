package cudaagreement_test

import (
	_ "embed"
	"fmt"
	"math"
	"os"
	"slices"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaagreement"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

//go:embed testdata/decoder.cu
var decoderSource string

// Per forward: ten chain launches and seven cuBLAS calls (qkv, scores, pv,
// o, gate_up, down, lm_head).
const (
	decoderChainLaunches = 10
	decoderGEMMCalls     = 7
	decoderBlock         = 256
)

func isChainKernel(name string) bool { return strings.HasPrefix(name, "chain_") }

// decoderDevice is what every rig on one device shares: the chain kernels and
// a cuBLAS handle bound to the harness stream.
type decoderDevice struct {
	h    *harness
	fn   map[string]*cuda.Kernel
	blas cublas.Handle
}

func newDecoderDevice(t *testing.T, h *harness) *decoderDevice {
	t.Helper()
	image, err := nvccCUBIN(t, "decoder", decoderSource, os.Getenv("CUDAAGREEMENT_NATIVE_TARGET"))
	if err != nil {
		t.Fatal(err)
	}
	mod := loadModule(t, h.dev, image)
	d := &decoderDevice{h: h, fn: map[string]*cuda.Kernel{}}
	for _, name := range []string{"chain_embed", "chain_rmsnorm", "chain_rope_kv", "chain_softmax", "chain_merge_heads", "chain_residual_add", "chain_silu_mul"} {
		fn, err := mod.Function(name)
		if err != nil {
			t.Fatal(err)
		}
		d.fn[name] = fn
	}
	err = h.dev.WithContext(func() error {
		var err error
		if d.blas, err = cublas.Create(); err != nil {
			return err
		}
		version, err := d.blas.Version()
		if err != nil {
			return err
		}
		t.Logf("cuBLAS version=%d math=DefaultMath compute=Compute32F types=R32F algo=GemmDefault", version)
		if err := d.blas.SetMathMode(cublas.DefaultMath); err != nil {
			return err
		}
		return d.blas.SetStream(h.stream.Raw())
	})
	if err != nil {
		t.Fatalf("cuBLAS: %v", err)
	}
	t.Cleanup(func() {
		synchronize(t, h.stream)
		if err := h.dev.WithContext(d.blas.Close); err != nil {
			t.Error(err)
		}
	})
	return d
}

// decoderRig is one shape's weights and activations on the device.
type decoderRig struct {
	d   *decoderDevice
	s   decoderShape
	in  *decoderInputs
	w   map[string]cudamem.Slice[float32]
	ids cudamem.Slice[int32]
	act map[string]cudamem.Slice[float32]
	// kcInit and vcInit are the caches as uploaded before each forward: the
	// prefix at positions before L, NaN after it.
	kcInit, vcInit []float32
}

func newDecoderRig(t *testing.T, d *decoderDevice, s decoderShape, seed uint64) *decoderRig {
	t.Helper()
	r := &decoderRig{d: d, s: s, in: newDecoderInputs(s, seed), w: map[string]cudamem.Slice[float32]{}, act: map[string]cudamem.Slice[float32]{}}
	alloc := func(n int) cudamem.Slice[float32] {
		b, err := cudamem.Alloc[float32](d.h.dev, n)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := b.Free(); err != nil {
				t.Error(err)
			}
		})
		return b
	}
	for name, host := range map[string][]float32{
		"emb": r.in.emb, "wn1": r.in.wn1, "wqkv": r.in.wqkv, "wo": r.in.wo, "wn2": r.in.wn2,
		"wgu": r.in.wgu, "wd": r.in.wd, "wnf": r.in.wnf, "wlm": r.in.wlm, "inv_freq": r.in.invFreq,
	} {
		b := alloc(len(host))
		if err := b.CopyFromHost(host); err != nil {
			t.Fatal(err)
		}
		r.w[name] = b
	}
	var err error
	if r.ids, err = cudamem.Alloc[int32](d.h.dev, s.T); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.ids.Free(); err != nil {
			t.Error(err)
		}
	})
	if err := r.ids.CopyFromHost(r.in.ids); err != nil {
		t.Fatal(err)
	}
	for name, n := range decoderBuffers(s) {
		r.act[name] = alloc(n)
	}
	hd, S := s.HD(), s.S()
	nan := float32(math.NaN())
	r.kcInit = make([]float32, s.H*S*hd)
	r.vcInit = make([]float32, s.H*S*hd)
	for i := range r.kcInit {
		r.kcInit[i], r.vcInit[i] = nan, nan
	}
	for h := range s.H {
		copy(r.kcInit[h*S*hd:], r.in.kPrefix[h*s.L*hd:(h+1)*s.L*hd])
		copy(r.vcInit[h*S*hd:], r.in.vPrefix[h*s.L*hd:(h+1)*s.L*hd])
	}
	return r
}

// reset poisons every activation with NaN and restores the caches, so that a
// stage that did not run fails the oracle rather than reading the previous
// forward's values.
func (r *decoderRig) reset(t *testing.T) {
	t.Helper()
	for name, b := range r.act {
		switch name {
		case "kc":
			if err := b.CopyFromHost(r.kcInit); err != nil {
				t.Fatal(err)
			}
		case "vc":
			if err := b.CopyFromHost(r.vcInit); err != nil {
				t.Fatal(err)
			}
		default:
			if err := b.MemsetAsync(0xff, r.d.h.stream); err != nil {
				t.Fatal(err)
			}
		}
	}
	synchronize(t, r.d.h.stream)
}

func (r *decoderRig) launch(name string, grid int, args ...any) error {
	err := cudalaunch.OwnedKernel(r.d.fn[name]).Grid(uint32(grid), 1, 1).Block(decoderBlock, 1, 1).On(r.d.h.stream).Launch(args...)
	if err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	return nil
}

// gemm is one column-major fp32 cuBLAS call, strided-batched when batch > 1.
func (r *decoderRig) gemm(ta, tb cublas.Operation, m, n, k int, a cudamem.Slice[float32], lda int, sa int, b cudamem.Slice[float32], ldb int, sb int, c cudamem.Slice[float32], ldc int, sc int, batch int) error {
	return r.d.h.dev.WithContext(func() error {
		alpha, beta := cublas.Float32Scalar(1), cublas.Float32Scalar(0)
		if batch == 1 {
			return r.d.blas.GemmEx(ta, tb, int32(m), int32(n), int32(k),
				alpha, a.Ptr(), cublas.R32F, int32(lda),
				b.Ptr(), cublas.R32F, int32(ldb),
				beta, c.Ptr(), cublas.R32F, int32(ldc), cublas.Compute32F, cublas.GemmDefault)
		}
		return r.d.blas.GemmStridedBatchedEx(ta, tb, int32(m), int32(n), int32(k),
			alpha, a.Ptr(), cublas.R32F, int32(lda), int64(sa),
			b.Ptr(), cublas.R32F, int32(ldb), int64(sb),
			beta, c.Ptr(), cublas.R32F, int32(ldc), int64(sc), int32(batch), cublas.Compute32F, cublas.GemmDefault)
	})
}

// linear is y = W·x with W out×in column-major and one column per token.
func (r *decoderRig) linear(w string, out, in int, x, y string) error {
	return r.gemm(cublas.OpN, cublas.OpN, out, r.s.T, in, r.w[w], out, 0, r.act[x], in, 0, r.act[y], out, 0, 1)
}

// forward submits one forward on the harness stream. causal=false is the
// planted defect: softmax attends to every cached position.
func (r *decoderRig) forward(causal bool) error {
	s, a, w := r.s, r.act, r.w
	D, H, F, hd, S, T, L := int32(s.D), int32(s.H), int32(s.F), int32(s.HD()), int32(s.S()), int32(s.T), int32(s.L)
	blocks := func(n int32) int { return int((n + decoderBlock - 1) / decoderBlock) }
	var c int32
	if causal {
		c = 1
	}
	steps := []func() error{
		func() error { return r.launch("chain_embed", blocks(T*D), r.ids, w["emb"], a["x0"], T, D) },
		func() error { return r.launch("chain_rmsnorm", int(T), a["x0"], w["wn1"], a["xn1"], D, decoderEps) },
		func() error { return r.linear("wqkv", 3*s.D, s.D, "xn1", "qkv") },
		func() error {
			return r.launch("chain_rope_kv", blocks(T*H*hd/2), a["qkv"], w["inv_freq"], a["q"], a["kc"], a["vc"], T, L, S, H, hd)
		},
		// scores[h] (S×T) = kc[h]ᵀ (S×hd) · q[h] (hd×T)
		func() error {
			return r.gemm(cublas.OpT, cublas.OpN, s.S(), s.T, s.HD(), a["kc"], s.HD(), s.S()*s.HD(), a["q"], s.HD(), s.T*s.HD(), a["scores"], s.S(), s.T*s.S(), s.H)
		},
		func() error {
			return r.launch("chain_softmax", int(H*T), a["scores"], a["probs"], T, L, S, r.in.scale, c)
		},
		// o[h] (hd×T) = vc[h] (hd×S) · probs[h] (S×T)
		func() error {
			return r.gemm(cublas.OpN, cublas.OpN, s.HD(), s.T, s.S(), a["vc"], s.HD(), s.S()*s.HD(), a["probs"], s.S(), s.T*s.S(), a["o"], s.HD(), s.T*s.HD(), s.H)
		},
		func() error { return r.launch("chain_merge_heads", blocks(T*D), a["o"], a["attn"], T, H, hd) },
		func() error { return r.linear("wo", s.D, s.D, "attn", "ao") },
		func() error { return r.launch("chain_residual_add", blocks(T*D), a["x0"], a["ao"], a["x1"], T*D) },
		func() error { return r.launch("chain_rmsnorm", int(T), a["x1"], w["wn2"], a["xn2"], D, decoderEps) },
		func() error { return r.linear("wgu", 2*s.F, s.D, "xn2", "gu") },
		func() error { return r.launch("chain_silu_mul", blocks(T*F), a["gu"], a["h"], T, F) },
		func() error { return r.linear("wd", s.D, s.F, "h", "dn") },
		func() error { return r.launch("chain_residual_add", blocks(T*D), a["x1"], a["dn"], a["x2"], T*D) },
		func() error { return r.launch("chain_rmsnorm", int(T), a["x2"], w["wnf"], a["xnf"], D, decoderEps) },
		func() error { return r.linear("wlm", s.V, s.D, "xnf", "logits") },
	}
	for i, step := range steps {
		if err := step(); err != nil {
			return fmt.Errorf("forward step %d: %w", i, err)
		}
	}
	return nil
}

// readback copies every activation to the host.
func (r *decoderRig) readback(t *testing.T) map[string][]float32 {
	t.Helper()
	st := make(map[string][]float32)
	for name, b := range r.act {
		v := make([]float32, b.Len())
		if err := b.CopyToHost(v); err != nil {
			t.Fatal(err)
		}
		st[name] = v
	}
	return st
}

// run resets, submits one forward, drains the stream, and returns the
// number of stream launches the forward issued through cudalaunch.
func (r *decoderRig) run(t *testing.T, causal bool) uint64 {
	t.Helper()
	r.reset(t)
	before := cudalaunch.TotalLaunches()
	if err := r.forward(causal); err != nil {
		t.Fatal(err)
	}
	synchronize(t, r.d.h.stream)
	return cudalaunch.TotalLaunches() - before
}

// TestDecoderLayerHarnessLaunches runs the decoder layer at both correctness
// shapes, checks every stage of every token against the stagewise oracle,
// and then plants the causal-mask defect, which must fail softmax and only
// softmax.
func TestDecoderLayerHarnessLaunches(t *testing.T) {
	h := newHarness(t)
	d := newDecoderDevice(t, h)
	for _, s := range decoderCorrectness {
		t.Run(s.Name, func(t *testing.T) {
			r := newDecoderRig(t, d, s, 1)
			t.Logf("fixture: %s", s)
			n := r.run(t, true)
			if n != decoderChainLaunches {
				t.Fatalf("forward issued %d chain launches, want %d", n, decoderChainLaunches)
			}
			st := r.readback(t)
			rep := checkDecoder(s, r.in, st, s.rows(true))
			t.Logf("stagewise oracle, all %d tokens:\n%s", s.T, rep)
			if f := rep.failed(); len(f) != 0 {
				t.Fatalf("stages %v fail the oracle", f)
			}
			logits := st["logits"]
			t.Logf("logits[0][0:4] = %v, logits[T-1][V-4:] = %v", logits[:4], logits[len(logits)-4:])

			r.run(t, false)
			mrep := checkDecoder(s, r.in, r.readback(t), s.rows(true))
			t.Logf("planted defect (causal mask removed):\n%s", mrep)
			if f := mrep.failed(); !slices.Equal(f, []string{"softmax"}) {
				t.Fatalf("causal-mask defect failed stages %v, want [softmax]", f)
			}
		})
	}
}

// startKernelActivity starts a CUPTI activity session collecting kernel
// executions only.
func startKernelActivity(t *testing.T) *cuptiactivity.Session {
	t.Helper()
	s, err := cuptiactivity.Start(cuptiactivity.Config{
		Kinds:       []cupti.ActivityKind{cupti.ActivityKindConcurrentKernel},
		BufferBytes: 8 << 20,
		MaxBuffers:  16,
		MaxRecords:  1 << 16,
	})
	if err != nil {
		if s != nil {
			s.Close()
		}
		t.Fatalf("cuptiactivity.Start: %v", err)
	}
	return s
}

// lossless reports an error unless res is a closed collection that lost
// nothing. It does not require records: an empty region has none.
func lossless(res cuptiactivity.Result) error {
	switch {
	case !res.Closed:
		return fmt.Errorf("session not closed")
	case res.PendingBuffers != 0 || res.RefusedBuffers != 0 || res.IncompleteRecords != 0 ||
		res.DiscardedRecords != 0 || res.DroppedRecords != 0 || res.CallbackErrors != 0 || res.FirstError != nil:
		return fmt.Errorf("collection lost data: pending=%d refused=%d incomplete=%d discarded=%d dropped=%d callback_errors=%d first=%v",
			res.PendingBuffers, res.RefusedBuffers, res.IncompleteRecords, res.DiscardedRecords, res.DroppedRecords, res.CallbackErrors, res.FirstError)
	}
	return nil
}

// TestSpanOfEmptyRegionNative holds a collection open over a region that runs
// no kernel. The span is zero however long the region was open, and no
// fraction is computed. A positive control in the same process, one kernel in
// an otherwise identical session, shows the collector would have seen one.
func TestSpanOfEmptyRegionNative(t *testing.T) {
	h := newHarness(t)

	sess := startKernelActivity(t)
	opened := time.Now()
	time.Sleep(50 * time.Millisecond)
	synchronize(t, h.stream)
	open := time.Since(opened)
	if err := sess.Close(); err != nil {
		t.Fatal(err)
	}
	res := sess.Result()
	if err := lossless(res); err != nil {
		t.Fatal(err)
	}
	ks, err := cudaagreement.KernelsFromActivity(res.Records, isChainKernel)
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudaagreement.Account(ks)
	if err != nil {
		t.Fatal(err)
	}
	_, ferr := a.Fractions()
	t.Logf("empty region: open=%v records=%d kernels=%d account: %v; Fractions: %v", open, len(res.Records), len(ks), a, ferr)
	if len(ks) != 0 || a != (cudaagreement.SpanAccount{}) {
		t.Fatalf("empty region accounts as %v, want zero", a)
	}
	if ferr == nil {
		t.Fatal("Fractions divided by an empty region")
	}

	// Positive control: one launch of the integer fixture's first step.
	x, w := fixtureInputs(0)
	if err := h.slices["x"].CopyFromHost(x); err != nil {
		t.Fatal(err)
	}
	if err := h.slices["w"].CopyFromHost(w); err != nil {
		t.Fatal(err)
	}
	one := &cudaagreement.Plan{Name: "one", Steps: h.plan.Steps[:1]}
	sess = startKernelActivity(t)
	opened = time.Now()
	if _, err := (&cudaagreement.LaunchExecutor{Modules: h.mods}).Execute(h.stream, one, h.bufs); err != nil {
		t.Fatal(err)
	}
	time.Sleep(50 * time.Millisecond)
	synchronize(t, h.stream)
	open = time.Since(opened)
	if err := sess.Close(); err != nil {
		t.Fatal(err)
	}
	res = sess.Result()
	if err := lossless(res); err != nil {
		t.Fatal(err)
	}
	ks, err = cudaagreement.KernelsFromActivity(res.Records, func(string) bool { return true })
	if err != nil {
		t.Fatal(err)
	}
	a, err = cudaagreement.Account(ks)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("one-kernel control: open=%v kernels=%d account: %v", open, len(ks), a)
	if len(ks) != 1 || a.SpanNs == 0 || a.SpanNs >= uint64(open.Nanoseconds()) {
		t.Fatalf("one-kernel control: %d kernels, span %d ns over a %v session; want 1 kernel whose span is nonzero and shorter than the session", len(ks), a.SpanNs, open)
	}
}

// TestSpanFractionsNative measures, at each measured shape, the share of one
// forward's kernel launches and device span that the chain kernels account
// for. Each repetition is one CUPTI session around exactly one forward; the
// warmups that load cuBLAS kernels and pick its algorithms run outside any
// session.
func TestSpanFractionsNative(t *testing.T) {
	const warmups, reps = 3, 7
	h := newHarness(t)
	d := newDecoderDevice(t, h)
	for _, s := range decoderMeasured {
		t.Run(s.Name, func(t *testing.T) {
			r := newDecoderRig(t, d, s, 2)
			t.Logf("fixture: %s", s)
			for range warmups {
				r.run(t, true)
			}
			var accounts []cudaagreement.SpanAccount
			for rep := range reps {
				r.reset(t)
				before := cudalaunch.TotalLaunches()
				sess := startKernelActivity(t)
				if err := r.forward(true); err != nil {
					t.Fatal(err)
				}
				synchronize(t, h.stream)
				if err := sess.Close(); err != nil {
					t.Fatal(err)
				}
				launched := cudalaunch.TotalLaunches() - before
				res := sess.Result()
				if err := lossless(res); err != nil {
					t.Fatalf("rep %d: %v", rep, err)
				}
				ks, err := cudaagreement.KernelsFromActivity(res.Records, isChainKernel)
				if err != nil {
					t.Fatalf("rep %d: %v", rep, err)
				}
				a, err := cudaagreement.Account(ks)
				if err != nil {
					t.Fatalf("rep %d: %v", rep, err)
				}
				f, err := a.Fractions()
				if err != nil {
					t.Fatalf("rep %d: %v", rep, err)
				}
				// Counted control: the classifier's chain count is the
				// number of chain launches the host actually made.
				if uint64(a.ChainLaunches) != launched || launched != decoderChainLaunches {
					t.Fatalf("rep %d: %d chain kernel records, %d chain launches counted, want %d of each", rep, a.ChainLaunches, launched, decoderChainLaunches)
				}
				if a.OtherLaunches < decoderGEMMCalls {
					t.Fatalf("rep %d: %d non-chain kernel records for %d cuBLAS calls", rep, a.OtherLaunches, decoderGEMMCalls)
				}
				if a.ChainNs+a.OtherNs+a.OverlapNs+a.IdleNs != a.SpanNs || a.ChainLaunches+a.OtherLaunches != a.Launches {
					t.Fatalf("rep %d: parts do not sum: %v", rep, a)
				}
				t.Logf("rep %d: %v", rep, a)
				t.Logf("rep %d: fractions launches chain=%.4f other=%.4f sum=%.4f (of %d kernel records); span chain=%.4f other=%.4f overlap=%.4f idle=%.4f sum=%.4f (of %d ns)",
					rep, f.ChainLaunches, f.OtherLaunches, f.ChainLaunches+f.OtherLaunches, a.Launches,
					f.ChainSpan, f.OtherSpan, f.OverlapSpan, f.IdleSpan, f.ChainSpan+f.OtherSpan+f.OverlapSpan+f.IdleSpan, a.SpanNs)
				t.Logf("rep %d kernels:\n%s", rep, kernelTable(ks))
				accounts = append(accounts, a)
			}
			logMedians(t, accounts)

			// The measured forward must be a correct one.
			rep := checkDecoder(s, r.in, r.readback(t), s.rows(false))
			t.Logf("stagewise oracle, tokens %v of the last measured forward:\n%s", s.rows(false), rep)
			if f := rep.failed(); len(f) != 0 {
				t.Fatalf("stages %v fail the oracle", f)
			}
		})
	}
}

// kernelTable lists kernels in start order with their durations.
func kernelTable(ks []cudaagreement.Kernel) string {
	ks = slices.Clone(ks)
	sort.Slice(ks, func(i, j int) bool { return ks[i].Start < ks[j].Start })
	var b strings.Builder
	for i, k := range ks {
		gap := uint64(0)
		if i > 0 && k.Start > ks[i-1].End {
			gap = k.Start - ks[i-1].End
		}
		fmt.Fprintf(&b, "  %2d chain=%-5v dur_ns=%-9d gap_before_ns=%-7d %s\n", i, k.Chain, k.End-k.Start, gap, k.Name)
	}
	return b.String()
}

// logMedians reports each span part's median over the repetitions, with its
// range. Medians are per part and are not expected to sum.
func logMedians(t *testing.T, as []cudaagreement.SpanAccount) {
	t.Helper()
	med := func(get func(cudaagreement.SpanAccount) float64) (m, lo, hi float64) {
		v := make([]float64, len(as))
		for i, a := range as {
			v[i] = get(a)
		}
		slices.Sort(v)
		return v[len(v)/2], v[0], v[len(v)-1]
	}
	for _, p := range []struct {
		name string
		get  func(cudaagreement.SpanAccount) float64
	}{
		{"span_ns", func(a cudaagreement.SpanAccount) float64 { return float64(a.SpanNs) }},
		{"chain_span_fraction", func(a cudaagreement.SpanAccount) float64 { return float64(a.ChainNs) / float64(a.SpanNs) }},
		{"other_span_fraction", func(a cudaagreement.SpanAccount) float64 { return float64(a.OtherNs) / float64(a.SpanNs) }},
		{"overlap_span_fraction", func(a cudaagreement.SpanAccount) float64 { return float64(a.OverlapNs) / float64(a.SpanNs) }},
		{"idle_span_fraction", func(a cudaagreement.SpanAccount) float64 { return float64(a.IdleNs) / float64(a.SpanNs) }},
		{"chain_launch_fraction", func(a cudaagreement.SpanAccount) float64 { return float64(a.ChainLaunches) / float64(a.Launches) }},
	} {
		m, lo, hi := med(p.get)
		t.Logf("median over %d reps: %s=%.4f range=[%.4f, %.4f]", len(as), p.name, m, lo, hi)
	}
}
