package main

import (
	"cmp"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudasample"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/safetensors"
)

// TestHFParity compares llama-infer's full-vocabulary logits with a
// Transformers reference suite written by validation/hf_reference.py. See
// validation/README.md for the command that runs both.
//
// For each case the engine prefills the turn's new tokens in chunks,
// continuing the KV cache the previous turn left, then decodes the
// reference's generated tokens (teacher forcing). Every chunk's last row
// and every decode step's row is compared with the reference row at the
// same position. Teacher forcing keeps every row comparable after a
// disagreement; top-1 agreement on the rows that predict generated tokens
// is what greedy equality with the reference would require.
//
// Environment:
//
//	LLAMA_HF_PARITY         reference suite directory (required)
//	LLAMA_HF_PARITY_MODEL   checkpoint directory (default: the suite's)
//	LLAMA_HF_PARITY_ENGINE  egraph (default), bf16 or f32
//	LLAMA_HF_PARITY_CHUNKS  egraph prefill chunks, comma-separated: rows,
//	                        "all" for one chunk, or 0 for token by token
//	                        (default 0,1,4,8,40,all)
//	LLAMA_HF_PARITY_GRAPH   0 decodes eagerly instead of in the CUDA graph
//	LLAMA_HF_PARITY_ATOL    absolute tolerance (default: per engine)
//	LLAMA_HF_PARITY_RTOL    relative tolerance (default: per engine)
//	LLAMA_HF_PARITY_REPORT  write the JSON report to this file
func TestHFParity(t *testing.T) {
	dir := os.Getenv("LLAMA_HF_PARITY")
	if dir == "" {
		t.Skip("LLAMA_HF_PARITY must name a reference suite from validation/hf_reference.py")
	}
	suite, err := readHFSuite(dir)
	if err != nil {
		t.Fatal(err)
	}
	model := cmp.Or(os.Getenv("LLAMA_HF_PARITY_MODEL"), suite.Checkpoint)
	kind := cmp.Or(os.Getenv("LLAMA_HF_PARITY_ENGINE"), "egraph")
	tol, err := defaultTolerance(kind, suite)
	if err != nil {
		t.Fatal(err)
	}
	if tol, err = tol.fromEnv(); err != nil {
		t.Fatal(err)
	}
	chunks := []string{"0"}
	if kind == "egraph" {
		chunks = strings.Split(cmp.Or(os.Getenv("LLAMA_HF_PARITY_CHUNKS"), "0,1,4,8,40,all"), ",")
	}
	graph := os.Getenv("LLAMA_HF_PARITY_GRAPH") != "0"

	dev := cudatest.Require(t)
	td := &teardown{}
	t.Cleanup(func() {
		if td.held() {
			return
		}
		if err := dev.Close(); err != nil {
			t.Errorf("close device: %v", err)
		}
	})
	report := parityReport{
		Suite:  dir,
		Model:  model,
		Engine: kind,
		Graph:  graph && kind == "egraph",
		ATol:   tol.ATol,
		RTol:   tol.RTol,
		Ref: parityRef{
			Revision: suite.Revision, Files: suite.Files, FileDigestSources: suite.FileDigestSources, Versions: suite.Versions,
			DType: suite.DType, Device: suite.Device,
		},
	}
	if suite.HFBF16 != nil {
		report.Ref.HFBF16MaxAbs = suite.HFBF16.MaxAbs
	}
	err = dev.WithContext(func() error {
		e, err := newParityEngine(t, td, dev, model, kind, graph)
		if err != nil {
			return err
		}
		for _, c := range chunks {
			if err := e.setChunk(c); err != nil {
				return err
			}
			for _, tc := range suite.Cases {
				r, err := runParityCase(e, dir, tc, suite.bf16Rows(tc.Name), tol)
				if err != nil {
					return fmt.Errorf("chunk %s case %s: %w", c, tc.Name, err)
				}
				r.Chunk = c
				report.Results = append(report.Results, r)
				t.Logf("chunk=%-3s %-15s rows=%3d max_abs=%.3g row_rel=%.3g tol_ratio=%.3g bf16_ratio=%.3g top1=%d/%d greedy=%d/%d",
					c, tc.Name, r.Rows, r.MaxAbs, r.MaxRowRel, r.MaxTolRatio, r.MaxBF16Ratio, r.Top1, r.Rows, r.Greedy, r.GreedyRows)
				for _, d := range r.Disagree {
					t.Logf("    pos=%d ours=%d ref=%d ref_gap=%.4g ours_gap=%.4g", d.Pos, d.Ours, d.Ref, d.RefGap, d.OursGap)
				}
				if r.MaxTolRatio > 1 {
					t.Errorf("chunk %s case %s: max tolerance ratio %.3g > 1 (max abs %.3g at position %d)", c, tc.Name, r.MaxTolRatio, r.MaxAbs, r.MaxAbsPos)
				}
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	report.summarize()
	t.Logf("all: rows=%d max_abs=%.3g tol_ratio=%.3g bf16_ratio=%.3g top1=%d/%d greedy=%d/%d cases_passed=%d/%d",
		report.Rows, report.MaxAbs, report.MaxTolRatio, report.MaxBF16, report.Top1, report.Rows, report.Greedy, report.GreedyRows, report.Passed, len(report.Results))
	if out := os.Getenv("LLAMA_HF_PARITY_REPORT"); out != "" {
		b, err := json.MarshalIndent(report, "", " ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(out, append(b, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// parityTolerance is an element-wise tolerance: every logit must satisfy
// |ours-ref| <= ATol + RTol·|ref|.
type parityTolerance struct{ ATol, RTol float64 }

// defaultTolerance returns the tolerance engine kind is held to against
// suite. The f32 engine gets Luminal's float32 gate. The BF16 engines get
// an absolute tolerance measured on the suite: the largest error
// Transformers' own bfloat16 forward shows against the float32 oracle on
// the same rows. Passing it means the engine is no further from float32
// anywhere than Transformers in bfloat16 is somewhere.
func defaultTolerance(kind string, suite *hfSuite) (parityTolerance, error) {
	switch kind {
	case "f32":
		return parityTolerance{5e-3, 5e-4}, nil
	case "bf16", "egraph":
		if suite.HFBF16 == nil {
			if os.Getenv("LLAMA_HF_PARITY_ATOL") != "" {
				return parityTolerance{}, nil
			}
			return parityTolerance{}, fmt.Errorf("engine %s: the suite has no bfloat16 measurement; regenerate it without --no-bf16 or set LLAMA_HF_PARITY_ATOL", kind)
		}
		return parityTolerance{suite.HFBF16.MaxAbs, 0}, nil
	}
	return parityTolerance{}, fmt.Errorf("LLAMA_HF_PARITY_ENGINE=%q: want egraph, bf16 or f32", kind)
}

func (tol parityTolerance) fromEnv() (parityTolerance, error) {
	for _, v := range []struct {
		env string
		p   *float64
	}{{"LLAMA_HF_PARITY_ATOL", &tol.ATol}, {"LLAMA_HF_PARITY_RTOL", &tol.RTol}} {
		s := os.Getenv(v.env)
		if s == "" {
			continue
		}
		f, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return tol, fmt.Errorf("%s: %w", v.env, err)
		}
		*v.p = f
	}
	if math.IsNaN(tol.ATol) || math.IsInf(tol.ATol, 0) || tol.ATol < 0 ||
		math.IsNaN(tol.RTol) || math.IsInf(tol.RTol, 0) || tol.RTol < 0 {
		return tol, fmt.Errorf("parity tolerances must be finite and nonnegative")
	}
	return tol, nil
}

// hfSuite is suite.json from validation/hf_reference.py.
type hfSuite struct {
	Checkpoint        string            `json:"checkpoint"`
	Revision          string            `json:"revision"`
	Files             map[string]string `json:"files"`
	FileDigestSources map[string]string `json:"file_digest_sources,omitempty"`
	Versions          map[string]string `json:"versions"`
	DType             string            `json:"dtype"`
	Device            string            `json:"device"`
	Cases             []hfCase          `json:"cases"`
	HFBF16            *struct {
		MaxAbs float64 `json:"max_abs"`
		Cases  []struct {
			Name      string    `json:"name"`
			RowMaxAbs []float64 `json:"row_max_abs"`
		} `json:"cases"`
	} `json:"hf_bf16"`
}

// bf16Rows returns the per-row maximum error of Transformers' bfloat16
// forward on case name, or nil.
func (s *hfSuite) bf16Rows(name string) []float64 {
	if s.HFBF16 == nil {
		return nil
	}
	for _, c := range s.HFBF16.Cases {
		if c.Name == name {
			return c.RowMaxAbs
		}
	}
	return nil
}

// An hfCase is one chat turn. Its logits file holds the reference rows of
// positions Start through len(Prompt)+len(Generated)-1.
type hfCase struct {
	Name       string  `json:"name"`
	Start      int     `json:"start"`
	Prompt     []int32 `json:"prompt_tokens"`
	Generated  []int32 `json:"generated_tokens"`
	Text       string  `json:"text"`
	LogitsFile string  `json:"logits_file"`
}

func readHFSuite(dir string) (*hfSuite, error) {
	b, err := os.ReadFile(filepath.Join(dir, "suite.json"))
	if err != nil {
		return nil, err
	}
	var s hfSuite
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, fmt.Errorf("parse suite: %w", err)
	}
	if len(s.Cases) == 0 {
		return nil, fmt.Errorf("suite %s has no cases", dir)
	}
	return &s, nil
}

// readHFLogits returns the reference rows of tc, each vocab logits long.
func readHFLogits(dir string, tc hfCase, vocab int) (_ [][]float32, rerr error) {
	f, err := safetensors.Open(filepath.Join(dir, tc.LogitsFile))
	if err != nil {
		return nil, err
	}
	defer closeInto(&rerr, f, "close reference logits")
	tn, err := f.Tensor("logits")
	if err != nil {
		return nil, err
	}
	want := []int{len(tc.Prompt) + len(tc.Generated) - tc.Start, vocab}
	if !slices.Equal(tn.Shape, want) {
		return nil, fmt.Errorf("%s: logits shape %v, want %v", tc.LogitsFile, tn.Shape, want)
	}
	flat, err := safetensors.DecodeFloat32(tn.Dtype, tn.Data)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", tc.LogitsFile, err)
	}
	rows := make([][]float32, want[0])
	for i := range rows {
		rows[i] = flat[i*vocab : (i+1)*vocab]
	}
	return rows, nil
}

// A parityEngine is llama-infer set up as the CLI sets it up for one
// engine, with the KV cache carried from case to case as a chat runner
// carries it from turn to turn.
type parityEngine struct {
	dev      *cuda.Device
	cfg      *ModelConfig
	gpuW     *GPUModelWeights
	kv       *cudakv.Cache
	sp       *GPUScratchpad
	gm       *GPUModules
	attnFn   driver.Function
	sampleFn driver.Function
	dg       *DecodeGraph // nil: decode eagerly

	rows    int // the e-graph prefill's rows as built
	chunk   int // prefill chunk rows; 0 token by token
	logits  []float32
	cached  []int32 // the tokens in the KV cache
	inToken []int32
}

// newParityEngine loads the checkpoint in dir and builds kind's engine.
// The resources it allocates are released when t's test ends.
func newParityEngine(t *testing.T, td *teardown, dev *cuda.Device, dir, kind string, graph bool) (*parityEngine, error) {
	w, cfg, err := LoadModel(dir)
	if err != nil {
		return nil, err
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return nil, err
	}
	e := &parityEngine{dev: dev, cfg: cfg, logits: make([]float32, cfg.VocabSize), inToken: make([]int32, 1)}
	gm, err := CompileGPUModules(dev)
	if gm != nil {
		t.Cleanup(func() {
			if td.held() {
				return
			}
			if err := gm.Close(); err != nil {
				td.add(err)
				t.Errorf("close gpu modules: %v", err)
			}
		})
	}
	if err != nil {
		return nil, err
	}
	e.gm = gm
	attnC, err := cudaattn.Compile(major, minor)
	if err != nil {
		return nil, releaseErrorOwner(err)
	}
	attnMod, err := attnC.Load(dev)
	if err != nil {
		return nil, releaseErrorOwner(err)
	}
	t.Cleanup(func() {
		if td.held() {
			return
		}
		if err := attnMod.Close(); err != nil {
			td.add(err)
			t.Errorf("close attention module: %v", err)
		}
	})
	if e.attnFn, err = attnMod.Function("CausalFlashAttentionDecode"); err != nil {
		return nil, err
	}
	sampleC, err := cudasample.Compile(major, minor)
	if err != nil {
		return nil, releaseErrorOwner(err)
	}
	sampleMod, err := sampleC.Load(dev)
	if err != nil {
		return nil, releaseErrorOwner(err)
	}
	t.Cleanup(func() {
		if td.held() {
			return
		}
		if err := sampleMod.Close(); err != nil {
			td.add(err)
			t.Errorf("close sampler module: %v", err)
		}
	})
	if e.sampleFn, err = sampleMod.Function("ArgmaxLogits"); err != nil {
		return nil, err
	}

	dtype := weightBF16
	if kind == "f32" {
		dtype = weightF32
	}
	gpuW, err := uploadModel(dev, w, cfg, dtype)
	if gpuW != nil {
		t.Cleanup(func() {
			if td.held() {
				return
			}
			if err := gpuW.Free(); err != nil {
				td.add(err)
				t.Errorf("free gpu weights: %v", err)
			}
		})
	}
	if err != nil {
		return nil, err
	}
	e.gpuW = gpuW
	kv, err := cudakv.New(dev, cfg.NumHiddenLayers, cfg.NumKeyValueHeads, cfg.HeadDim(), cfg.MaxSeqLen)
	if kv != nil {
		t.Cleanup(func() {
			if td.held() {
				return
			}
			if err := kv.Close(); err != nil {
				td.add(err)
				t.Errorf("close kv cache: %v", err)
			}
		})
	}
	if err != nil {
		return nil, err
	}
	e.kv = kv
	sp, err := NewGPUScratchpad(dev, cfg)
	if sp != nil {
		t.Cleanup(func() {
			if td.held() {
				return
			}
			if err := sp.Free(); err != nil {
				td.add(err)
				t.Errorf("free scratchpad: %v", err)
			}
		})
	}
	if err != nil {
		return nil, err
	}
	e.sp = sp
	if kind == "egraph" {
		// As main does by default: generated attention, the prefill
		// sized to the KV cache, and decode attention split to cover the
		// SMs once.
		sms, err := dev.Attribute(driver.DevAttrMultiprocessorCount)
		if err != nil {
			return nil, err
		}
		eg, err := newEgraphDecoder(dev, w, cfg, e.gpuW, e.sp, egraphOptions{
			Attn:           true,
			PrefillRows:    -1,
			PrefillMinRows: 1,
			AttnSplits:     max(1, sms/cfg.NumKeyValueHeads),
		})
		if eg != nil {
			t.Cleanup(func() {
				if td.held() {
					return
				}
				if err := eg.Close(); err != nil {
					td.add(err)
					t.Errorf("close e-graph decoder: %v", err)
				}
			})
		}
		if err != nil {
			return nil, err
		}
		e.gm.eg = eg
		e.rows = eg.pf.rows
		if graph {
			dg, err := NewDecodeGraph(dev, e.gpuW, cfg, e.kv, e.sp, gm, e.attnFn, e.sampleFn)
			if dg != nil {
				t.Cleanup(func() {
					if td.held() {
						return
					}
					if err := dg.Close(); err != nil {
						td.add(err)
						t.Errorf("close decode graph: %v", err)
					}
				})
			}
			if err != nil {
				return nil, err
			}
			e.dg = dg
		}
	}
	// As main does: return the float32 host weights, which a GPU that
	// shares system memory needs.
	w = nil
	debug.FreeOSMemory()
	return e, nil
}

// setChunk sets the prefill chunk: a row count, "all" for the whole new
// suffix in one chunk, or 0 for token by token through the decode step.
//
// The CLI's -egraph-prefill-rows rounds up to a multiple of the tile
// height; a smaller chunk runs the kernels a short last chunk runs.
func (e *parityEngine) setChunk(s string) error {
	if s == "all" {
		e.chunk = e.rows
	} else {
		n, err := strconv.Atoi(s)
		if err != nil || n < 0 {
			return fmt.Errorf("prefill chunk %q: want rows, all or 0", s)
		}
		e.chunk = n
	}
	if e.chunk > 0 && e.gm.eg == nil {
		return fmt.Errorf("prefill chunk %s needs the egraph engine", s)
	}
	if e.chunk > e.rows && e.gm.eg != nil {
		return fmt.Errorf("prefill chunk %d exceeds the %d rows the prefill was built for", e.chunk, e.rows)
	}
	if e.gm.eg != nil {
		e.gm.eg.pf.rows = max(e.chunk, 1)
	}
	return nil
}

// prefill runs seq[start:] at positions start onward and calls row with
// each chunk's last position once e.logits holds that row.
func (e *parityEngine) prefill(seq []int32, start int, row func(pos int) error) error {
	read := func(pos int) error {
		if err := e.sp.Logits.CopyToHost(e.logits); err != nil {
			return err
		}
		return row(pos)
	}
	if e.chunk > 0 {
		_, err := e.gm.eg.prefill(e.gpuW, e.cfg, e.sp, e.sampleFn, seq, start, read)
		return err
	}
	for pos := start; pos < len(seq); pos++ {
		if err := e.forward(seq[pos], pos); err != nil {
			return err
		}
		if err := read(pos); err != nil {
			return err
		}
	}
	return nil
}

// decode runs tok at pos as a decode step and leaves its logits in
// e.logits.
func (e *parityEngine) decode(tok int32, pos int) error {
	if e.dg != nil {
		// The graph reads its input token from TokenOut, where the
		// previous step's argmax left it; force the reference's.
		e.inToken[0] = tok
		if err := e.sp.TokenOut.CopyFromHost(e.inToken); err != nil {
			return err
		}
		if _, err := e.dg.Step(pos, nil); err != nil {
			return err
		}
	} else if err := e.forward(tok, pos); err != nil {
		return err
	}
	return e.sp.Logits.CopyToHost(e.logits)
}

func (e *parityEngine) forward(tok int32, pos int) error {
	return e.dev.WithContext(func() error {
		_, err := GPUForwardPass(e.gpuW, e.cfg, e.kv, e.sp, e.gm, e.attnFn, e.sampleFn, tok, pos)
		return err
	})
}

// runParityCase runs tc on e, continuing the KV cache the previous case
// left when tc.Start > 0, and compares every row with the reference.
//
// If bf16 is non-nil it holds, per reference row, Transformers' bfloat16
// error, which MaxBF16Ratio measures our error against.
func runParityCase(e *parityEngine, dir string, tc hfCase, bf16 []float64, tol parityTolerance) (parityResult, error) {
	r := parityResult{Case: tc.Name, Start: tc.Start, PromptTokens: len(tc.Prompt), Generated: len(tc.Generated)}
	if tc.Start > len(e.cached) || !slices.Equal(e.cached[:tc.Start], tc.Prompt[:tc.Start]) {
		return r, fmt.Errorf("start %d: the KV cache does not hold the turn's first %d tokens", tc.Start, tc.Start)
	}
	ref, err := readHFLogits(dir, tc, e.cfg.VocabSize)
	if err != nil {
		return r, err
	}
	seq := slices.Concat(tc.Prompt, tc.Generated)
	e.cached = nil
	check := func(pos int) error {
		hf := 0.0
		if bf16 != nil {
			hf = bf16[pos-tc.Start]
		}
		r.add(pos, e.logits, ref[pos-tc.Start], hf, tol, pos >= len(tc.Prompt)-1 && pos < len(seq)-1)
		return nil
	}
	if err := e.prefill(seq[:len(tc.Prompt)], tc.Start, check); err != nil {
		return r, fmt.Errorf("prefill: %w", err)
	}
	// Decode every generated token, including a final EOS: its row is
	// compared too, and the next turn's cache holds it.
	for i, tok := range tc.Generated {
		pos := len(tc.Prompt) + i
		if err := e.decode(tok, pos); err != nil {
			return r, fmt.Errorf("decode position %d: %w", pos, err)
		}
		if err := check(pos); err != nil {
			return r, err
		}
	}
	e.cached = seq
	return r, nil
}

// A parityResult is one case's comparison. GreedyRows counts the rows that
// predict a generated token (the prompt's last row and every decode row
// but the last, whose prediction follows the turn's end) and Greedy those
// whose argmax matches the reference's.
type parityResult struct {
	Chunk        string               `json:"chunk"`
	Case         string               `json:"case"`
	Start        int                  `json:"start"`
	PromptTokens int                  `json:"prompt_tokens"`
	Generated    int                  `json:"generated_tokens"`
	Rows         int                  `json:"rows"`
	MaxAbs       float64              `json:"max_abs"`
	MaxAbsPos    int                  `json:"max_abs_pos"`
	MaxRowRel    float64              `json:"max_row_rel"` // MaxFloat64: nonzero error against an all-zero row
	MaxTolRatio  float64              `json:"max_tol_ratio"`
	MaxBF16Ratio float64              `json:"max_bf16_ratio"` // max over rows of max|d| / Transformers' bfloat16 max|d|
	Top1         int                  `json:"top1_agree"`
	Greedy       int                  `json:"greedy_agree"`
	GreedyRows   int                  `json:"greedy_rows"`
	Passed       bool                 `json:"passed"`
	Disagree     []parityDisagreement `json:"disagreements,omitempty"`
}

// A parityDisagreement is a row whose argmax differs from the
// reference's. RefGap is the reference's margin between its own top two
// logits; OursGap is our margin of our argmax over the reference's.
type parityDisagreement struct {
	Pos     int     `json:"pos"`
	Ours    int     `json:"ours"`
	Ref     int     `json:"ref"`
	RefGap  float64 `json:"ref_gap"`
	OursGap float64 `json:"ours_gap"`
}

// add compares the row at pos. hf is Transformers' bfloat16 error on
// it, or 0.
func (r *parityResult) add(pos int, ours, ref []float32, hf float64, tol parityTolerance, greedy bool) {
	var maxAbs, refMax, ratio float64
	for i, o := range ours {
		f := float64(ref[i])
		d := math.Abs(float64(o) - f)
		if math.IsNaN(d) {
			d = math.Inf(1)
		}
		maxAbs = max(maxAbs, d)
		refMax = max(refMax, math.Abs(f))
		allowed := tol.ATol + tol.RTol*math.Abs(f)
		if math.IsInf(d, 0) {
			ratio = math.Inf(1)
		} else if allowed > 0 {
			ratio = max(ratio, d/allowed)
		} else if d > 0 {
			ratio = math.Inf(1)
		}
	}
	r.Rows++
	if maxAbs > r.MaxAbs || r.Rows == 1 {
		r.MaxAbs, r.MaxAbsPos = maxAbs, pos
	}
	if refMax > 0 {
		r.MaxRowRel = max(r.MaxRowRel, maxAbs/refMax)
	} else if maxAbs > 0 {
		// Keep the report serializable when relative error is unbounded.
		// Numeric acceptance still comes from MaxTolRatio below.
		r.MaxRowRel = math.MaxFloat64
	}
	r.MaxTolRatio = max(r.MaxTolRatio, ratio)
	if hf > 0 {
		r.MaxBF16Ratio = max(r.MaxBF16Ratio, maxAbs/hf)
	}
	a, b := argmax(ours), argmax(ref)
	if greedy {
		r.GreedyRows++
	}
	if a == b {
		r.Top1++
		if greedy {
			r.Greedy++
		}
	} else {
		r.Disagree = append(r.Disagree, parityDisagreement{
			Pos: pos, Ours: a, Ref: b,
			RefGap:  float64(ref[b]) - secondMax(ref, b),
			OursGap: float64(ours[a] - ours[b]),
		})
	}
	r.Passed = r.MaxTolRatio <= 1
}

func argmax(x []float32) int {
	best := 0
	for i, v := range x {
		if v > x[best] {
			best = i
		}
	}
	return best
}

// secondMax returns the largest element of x other than x[skip].
func secondMax(x []float32, skip int) float64 {
	m := math.Inf(-1)
	for i, v := range x {
		if i != skip {
			m = max(m, float64(v))
		}
	}
	return m
}

type parityRef struct {
	Revision          string            `json:"revision"`
	Files             map[string]string `json:"files"`
	FileDigestSources map[string]string `json:"file_digest_sources,omitempty"`
	Versions          map[string]string `json:"versions"`
	DType             string            `json:"dtype"`
	Device            string            `json:"device"`
	HFBF16MaxAbs      float64           `json:"hf_bf16_max_abs,omitempty"`
}

// A parityReport is the JSON report LLAMA_HF_PARITY_REPORT names.
type parityReport struct {
	Suite       string         `json:"suite"`
	Model       string         `json:"model"`
	Engine      string         `json:"engine"`
	Graph       bool           `json:"cuda_graph_decode"`
	ATol        float64        `json:"atol"`
	RTol        float64        `json:"rtol"`
	Ref         parityRef      `json:"reference"`
	Rows        int            `json:"rows"`
	MaxAbs      float64        `json:"max_abs"`
	MaxTolRatio float64        `json:"max_tol_ratio"`
	MaxBF16     float64        `json:"max_bf16_ratio"`
	Top1        int            `json:"top1_agree"`
	Greedy      int            `json:"greedy_agree"`
	GreedyRows  int            `json:"greedy_rows"`
	Passed      int            `json:"cases_passed"`
	Results     []parityResult `json:"results"`
}

func (p *parityReport) summarize() {
	for _, r := range p.Results {
		p.Rows += r.Rows
		p.MaxAbs = max(p.MaxAbs, r.MaxAbs)
		p.MaxTolRatio = max(p.MaxTolRatio, r.MaxTolRatio)
		p.MaxBF16 = max(p.MaxBF16, r.MaxBF16Ratio)
		p.Top1 += r.Top1
		p.Greedy += r.Greedy
		p.GreedyRows += r.GreedyRows
		if r.Passed {
			p.Passed++
		}
	}
}
