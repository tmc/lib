package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// A green run of this package is not evidence of parity. Almost every
// oracle comparison is gated on an environment variable naming a model or
// a captured ds4 fixture, and an unset gate skips rather than fails —
// correctly, since neither is in the repository. What was missing is a way
// to see which gates a run actually exercised, so that "all tests pass" is
// legible rather than falsely reassuring.
type oracleGate struct {
	name    string
	enables string
}

// oracleGates is the gate of record. TestOracleGatesDocumented fails when
// the package reads a DS4_ variable that is not listed here, so the table
// cannot silently fall behind the tests.
var oracleGates = []oracleGate{
	{"DS4_CUDA_V41_OPS_GATE", "locked synthetic V4.1 CUDA activation/RoPE values and mutants plus SHA-pinned C frequency constructor; not model parity"},
	{"DS4_CUDA_V41_ENGRAM_GATE", "locked model-free V4.1 Engram F16 cuBLAS projection, BF16 gate, conversion and ownership controls; not full model parity"},
	{"DS4_CUDA_V41_ENGRAM_MUTANT", "actual Engram device source mutant: gate, value, norm or half; a named mutant must fail the native value gate"},
	{"DS4_MODEL", "every CUDA kernel, runtime and model comparison"},
	{"DS4_GGUF_PROFILE_MODEL", "the pinned GGUF profile and Metal model tests"},
	{"DS4_GGUF_MANIFEST_MODEL", "the tensor manifest digest"},
	{"DS4_GGUF_MANIFEST_OUT", "writing the manifest rather than comparing it"},
	{"DS4_GGUF_TOKENIZER_MODEL", "tokenizer round-trip against the model's vocabulary"},
	{"DS4_PRO_HEADER", "the seeded Pro tensor directory against the retained real Pro header"},
	{"DS4_PRO_SEED_GENERATOR", "the seeded Pro storage against the sealed independent C generator"},
	{"DS4_PRO_SEED_GENERATOR_SHA256", "the seal checked on DS4_PRO_SEED_GENERATOR before and after use"},

	{"DS4_LAYER0_ORACLE_DIR", "layer 0 against a captured ds4 layer"},
	{"DS4_LAYER0_POS1_ORACLE_DIR", "layer 0 at position 1"},
	{"DS4_POS_ORACLE_DIR", "one decode position against ds4"},
	{"DS4_ORACLE_STATE_DIR", "a teacher-forced logit trace against a ds4 observer capture"},
	{"DS4_ORACLE_STATE_OUT", "writing this port's own logit states in the observer's format"},
	{"DS4_ORACLE_STATE_COMPARE", "comparing captured logit states pairwise, with no model"},
	{"DS4_POS_ORACLE_POS", "which position DS4_POS_ORACLE_DIR holds"},
	{"DS4_ALL_LAYERS_ORACLE_DIR", "all 43 layers against ds4"},
	{"DS4_ALL_LAYERS_ORACLE_POS", "which position DS4_ALL_LAYERS_ORACLE_DIR holds"},
	{"DS4_TRANSITION_ORACLE_ROOT", "the prefill-to-decode transition"},
	{"DS4_COMPRESSOR_ORACLE_DIR", "the KV compressor against ds4"},
	{"DS4_COMPRESSOR_ORACLE_LAST_POS", "the last position in the compressor capture"},
	{"DS4_LAYER2_COMPRESSOR_ORACLE_DIRS", "layer 2 attention against four compressor captures"},

	{"DS4_METAL_PROFILE", "writing a Metal graph profile"},
	{"DS4_METAL_REGION_PROFILE", "ranking Metal command-buffer GPU-time regions"},
	{"DS4_METAL_REGION_PROFILE_OUT", "writing the Metal region profile receipt"},
	{"DS4_METAL_LAYER13_FIXTURE_DIR", "the layer 13 Metal regression"},
	{"DS4_METAL_HC_LAYER13_FIXTURE_DIR", "the layer 13 HC Metal regression"},
	{"DS4_METAL_HC_LAYER13_OUTPUT_DIR", "writing layer 13 HC fixtures rather than comparing them"},
	{"DS4_METAL_COUNTER_CALIBRATION", "the model-free Metal GPU counter calibration"},
	{"DS4_VISION_DEVICE", "locked synthetic Flash vision CUDA matmul device controls when set to 1"},
	{"DS4_VISION_NVRTC_LIBRARY", "the NVRTC library loaded for the Flash vision device controls"},
	{"DS4_VISION_MATMUL_ORACLE", "the pinned C kernel source for Flash vision matmul controls"},
	{"DS4_VISION_MATMUL_MUTANT", "same-site accumulator or BF16 conversion mutants in Flash vision controls"},
	{"DS4_VISION_CUDA_INCLUDE", "the CUDA include directory used by the Flash vision NVRTC compile"},
	{"DS4_VISION_NVRTC_BUILTINS", "the expected loaded NVRTC builtins file for tool provenance"},
	{"DS4_VISION_LOCK_OWNER_FILE", "the shared DGX GPU lease owner file checked by Flash vision controls"},
	{"DS4_VISION_LOCK_OWNER", "the exact sealed GPU lease owner text checked by Flash vision controls"},
	{"DS4_VISION_LOCK_EXEC", "the expected GPU lease holder executable checked by Flash vision controls"},

	{"DS4_OBSERVE_DIR", "where the observation harness writes"},
	{"DS4_OBSERVE_PROMPT_IDS", "the prompt the observation harness replays"},
	{"DS4_OBSERVE_MAX_ACCEPTED", "how far the observation harness decodes"},
	{"DS4_OBSERVE_ORACLE_DIR", "comparing an observation run against ds4"},
	{"DS4_OBSERVE_RENDER_DIR", "rendering an observation run"},

	{"DS4_PREFILL_GATE", "capturing full prefill and continuation logits"},
	{"DS4_PREFILL_BACKEND", "which backend the prefill capture exercises"},
	{"DS4_PREFILL_TOKENS", "the retained JSON token sequence for the prefill capture"},
	{"DS4_PREFILL_CASES", "which of 2, 64 and 2048 tokens the capture exercises"},
	{"DS4_PREFILL_CONTINUATION", "the fixed token evaluated after each prefill"},
	{"DS4_PREFILL_DIR", "the new directory for prefill capture artifacts"},

	{"DS4_ORACLE_WORKLOAD_GATE", "first-frontier 2048 prefill and 128 non-EOS decode measurement"},
	{"DS4_ORACLE_WORKLOAD_BACKEND", "backend for workload measurement"},
	{"DS4_ORACLE_WORKLOAD_PROMPT", "raw prompt matched to token prefix"},
	{"DS4_ORACLE_WORKLOAD_TOKENS", "retained oracle token corpus"},
	{"DS4_ORACLE_WORKLOAD_TOKENIZER_RECEIPT", "retained tokenizer discrepancy bound to workload identity"},
	{"DS4_ORACLE_WORKLOAD_DIR", "new workload artifact directory"},

	{"DS4_CUDA_PRODUCTION_HASH_GATE", "production Eval token-zero hash before and after Reset"},
	{"DS4_CUDA_COLLECTOR_GATE", "locked CUDA collector mid-value/mutant and off/on endpoint controls; each selected test qualifies only its own fixture"},
	{"DS4_CUDA_PHASE_PROFILE", "CUDA phase timing with stream events"},
	{"DS4_CUDA_PHASE_PROFILE_OUT", "new CUDA phase timing result file"},
	{"DS4_CUDA_PREFILL_PROGRESS", "CUDA EvalBatch progress diagnostic"},
	{"DS4_CUDA_PREFILL_PROGRESS_OUT", "new CUDA prefill progress JSONL file"},
	{"DS4_CUDA_PREFILL_GATHER_GATE", "gather versus per-token prefill and continuation comparison under the GPU lock"},
	{"DS4_CUDA_PREFILL_TOKENS", "retained 2048-token JSON corpus for the gather comparison"},
	{"DS4_CUDA_PREFILL_GATHER_RECEIPT", "optional output path for successful gather comparison snapshots"},
	{"DS4_CUDA_GEMM_STABILITY_GATE", "repeated prefill digests with the cuBLAS handle pinned, and across the library and hand-written arms, under the GPU lock"},
	{"DS4_CUDA_GEMM_STABILITY_ROUNDS", "how many alternating rounds those comparisons draw"},
	{"DS4_CUDA_GEMM_PIN_ARMS", "which cuBLAS handle states the pinning comparison draws"},
	{"DS4_CUDA_GEMM_WORKSPACE", "a test-owned cuBLAS workspace in place of the library-managed pool, alternated across processes"},
	{"DS4_CUDA_DENSE_D2R_ENVELOPE", "which dense call sites honour the reference's shape envelope, a measured and refuted lever kept inert by default"},
	{"DS4_CUDA_DENSE_ENVELOPE_GATE", "the dense shape-envelope A/B, which needs the GPU lock"},
	{"DS4_CUDA_DENSE_ENVELOPE_BLOCKS", "how many position-balanced blocks of six draws that A/B takes"},
	{"DS4_CUDA_DENSE_ENVELOPE_ALLOW_BAD", "a budget of non-OK draws the envelope A/B will tolerate before failing, for a deliberately hostile run; zero when unset"},
	{"DS4_CUDA_HC_MATH_DIAGNOSTIC", "the CUDA HC mathematical diagnostic"},
	{"DS4_CUDA_HC_BATCH", "batched HC pre and post operations across tokens"},
	{"DS4_CUDA_MOE_D2R", "fused Direct-to-Register (D2R) MoE expert execution"},
	{"DS4_CUDA_DISABLE_ATTN_HMMA", "attention tokentile HMMA path, off for width comparisons"},
	{"DS4_CUDA_DISABLE_LIBRARY_GEMM", "cuBLAS compressor, router and HC projections, off for the hand-written and per-token arms"},
	{"DS4_CUDA_DISABLE_COMPRESSOR_BATCH", "group-at-once compressor pooling, off for the token-at-a-time arm"},
	{"DS4_CUDA_DISABLE_ATTN_QKV_D2R", "tensor-core GEMM for attnQA/attnQB/attnKV, off for the matvec arm"},
	{"DS4_CUDA_DISABLE_RAW_ATTENTION_BATCH", "group-at-once raw attention for layers 0 and 1, off for the token-at-a-time arm"},
	{"DS4_CUDA_DISABLE_CHUNKED_ARENA", "block-suballocated batch row arena, off for one driver allocation per buffer"},
	{"DS4_CUDA_MOE_ARTIFACT_LAYERS", "how many layers repack their expert weights once at load instead of once per prefill group"},
	{"DS4_CUDA_LOAD_TIME_ARTIFACTS", "C-compatible mode: build all derived CUDA weight layouts at load and release the source mapping"},
	{"DS4_CUDA_MOE_ARTIFACT_VERIFY", "a strided digest of every MoE artifact, for checking the build is reproducible"},
	{"DS4_CUDA_DENSE_ARTIFACT_LAYERS", "how many layers repack their dense Q8_0 weights once at load instead of once per prefill group"},
	{"DS4_CUDA_DENSE_ARTIFACT_VERIFY", "a strided digest of the dense artifacts, taken after the build and again at Close"},
	{"DS4_CUDA_DENSE_ARTIFACT_MATVEC", "0 puts decode Q8_0 matvecs back on the model mapping instead of the dense artifact in device memory"},
	{"DS4_CUDA_BALLAST_GIB", "device memory allocated and never read, the control for artifact residency"},
	{"DS4_CUDA_BALLAST_CHUNKS", "how many separate allocations the ballast is split across, the shape control"},
	{"DS4_CUDA_PREFILL_REPEAT", "a second prefill in the same process, to separate a first-touch fault from one that persists"},
	{"DS4_CUDA_BATCH_ARENA_FILL", "a known value written into the batch buffers before the first prefill reads them"},
	{"DS4_CAPTURE_LIBC_EXIT", "leave through libc exit so a CUPTI agent finalizes its journal"},
	{"DS4_CUDA_PREFILL_D2R_GATE", "fused Direct-to-Register (D2R) MoE prefill regression under the GPU lock"},
	{"DS4_CUDA_ATTENTION_HMMA", "fused token-tile HMMA attention execution"},
	{"DS4_CUDA_PREFILL_HMMA_GATE", "fused token-tile HMMA attention prefill regression under the GPU lock"},
	{"DS4_CUDA_DENSE_D2R", "fused Direct-to-Register (D2R) dense Q8_0 Tensor Core execution"},
	{"DS4_CUDA_PREFILL_DENSE_D2R_GATE", "fused Direct-to-Register (D2R) dense Q8_0 prefill regression under the GPU lock"},
	{"DS4_DISABLE_CUDA_GRAPH", "disabling CUDA graph capture and replay in decode"},
	{"DS4_DUMP_LAYERS", "writing opt-in CUDA layer and attention diagnostics"},
	{"DS4_CUDA_DISABLE_MOE_D2R", "fused Direct-to-Register MoE prefill, off for the per-group repack arm"},
	{"DS4_CUDA_DISABLE_DENSE_D2R", "fused Direct-to-Register dense Q8_0 prefill, off for the matvec arm"},
	{"DS4_CUDA_DISABLE_MOE_IDS_HELPER", "fused compressed-row expert addressing helper, off for the baseline path"},
	{"DS4_CUDA_RAW_ROPE_GATE", "raw attention rotary position embedding regression under the GPU lock"},
	{"DS4_CUDA_DIRECT_STAGING", "artifact staging reads bypassing the page cache through an O_DIRECT descriptor"},
	{"DS4_CUDA_RELEASE_MAPPED_ARTIFACTS", "dropping a tensor's pages from the model mapping once its artifact is on the device (1 madvise, 2 also the page cache)"},
	{"DS4_CUDA_MAPPED_MODEL", "control arm: kernels read the file mapping instead of the resident spans and artifacts"},
	{"DS4_CUDA_WARM_MODEL", "faulting the whole model mapping in before the first kernel"},
	{"DS4_CUDA_PC_PROFILE_PERIOD", "the PC sampling period the profile test asks CUPTI for"},
	{"DS4_STATE_SNAPSHOT_GATE", "CUDA request State snapshot/restore and speculative rollback"},
	{"DS4_STATE_SAVELOAD_GATE", "CUDA request State serialized save/load and replay"},
	{"DS4_STATE_TRANSACTION_GATE", "CUDA native State identity, partial-write rollback and graph replay"},
	{"DS4_STATE_MULTI_REQUEST_GATE", "CUDA request State interleaving against serial references"},
	{"DS4_STATE_CONCURRENT_GATE", "CUDA concurrent request graph isolation against serial references"},
	{"DS4_CUDA_SPECULATIVE_GROUPED", "opt-in grouped speculative transformer readout, unqualified until its device gate passes"},
	{"DS4_DSPARK_SUPPORT_TEST", "host hash, size and tensor admission of the published DSpark support file, not forward or device execution"},
	{"DS4_SPECULATIVE_GROUPED_GATE", "grouped speculative n=2 and n=8 row predictions and accepted-prefix replay"},
	{"DS4_UPSTREAM_RESTORE_MANIFEST", "hash-pinned upstream CUDA state restore and continuation comparison"},
	{"DS4_UPSTREAM_RESTORE_LOGITS", "diagnostic raw continuation logits without changing gate tolerances"},
	{"DS4_METAL_UPSTREAM_RESTORE_MANIFEST", "hash-pinned upstream Metal shared-state restore and continuation comparison"},
	{"DS4_UPSTREAM_FLOAT_PLAN_RECORD", "host-only retained upstream payload staging into Metal float-buffer geometry"},
	{"DS4_METAL_Q4_EXPERT_GATE", "locked selected Q4_K Metal synthetic float64 oracle and same-site scale mutant"},
	{"DS4_METAL_MIXED_DIAGNOSTIC_DIR", "fresh exclusive model-free Metal mixed stage extraction artifacts, not numerical qualification"},
	{"DS4_METAL_UPSTREAM_ROPE_GATE", "locked compatibility of the Metal inverse RoPE with pinned upstream kernel_dsv4_rope_tail_f32 on device, not exact arithmetic"},
	{"DS4_METAL_UPSTREAM_ROPE_ORACLE_MUTANT", "named mutation of the extracted upstream RoPE shader that the upstream compatibility gate must reject"},
	{"DS4_CUDA_MXFP4_SELECTED_GATE", "locked selected MXFP4 CUDA Q8_1 synthetic float64 oracle and same-site scale/nibble mutants"},
	{"DS4_METAL_MXFP4_SELECTED_GATE", "locked selected MXFP4 Metal F32 synthetic float64 oracle and same-site scale/nibble mutants"},
	{"DS4_METAL_STATE_GATE", "locked registered-model independent Metal state, snapshot and native restore continuation gate"},
	{"DS4_METAL_SPECULATIVE_GATE", "locked registered-model scalar Metal verifier accepted-prefix and continuation self-oracle gate"},
	{"DS4_METAL_SPECULATIVE_GROUPED", "opt-in grouped Metal device-forward and all-row batched readout, physical equivalence unqualified"},
	{"DS4_METAL_SPECULATIVE_GROUPED_GATE", "locked registered-model grouped Metal n=2/8 all-row logits and all accepted-prefix state gate"},
	{"DS4_METAL_TRANSACTION_GATE", "locked tiny shared-buffer native Metal Load compensation and skipped-rollback mutant gate"},
	{"DS4_DUMP_POSITION", "position selector for opt-in CUDA diagnostics"},
	{"DS4_HEAD_OBSERVER_PREFIX", "exclusive opt-in live output-head diagnostic files; synchronizes and is not a performance mode"},
	{"DS4_HEAD_OBSERVER_POSITION", "required evaluated position selector for the opt-in output-head observer"},
	{"DS4_HEAD_OBSERVER_MODEL_SHA256", "externally verified model checksum assertion, explicitly labeled as supplied provenance"},
	{"DS4_HEAD_OBSERVER_SOURCE_MANIFEST", "source manifest file hashed by the opt-in output-head observer; external build binding remains required"},
	{"DS4_STATE_GRAPH_GATE", "CUDA request State graph capture and replay"},
	{"DS4_STATE_GRAPH_SNAPSHOT_GATE", "CUDA request State graph snapshot and restore"},
	{"DS4_STATE_CONCURRENT_GRAPH_GATE", "CUDA request State concurrent request graph replay"},

	{"DS4_DSPARK_SCHEDULER", "0 disables the DSpark draft skip scheduler"},
	{"DS4_DSPARK_SCHEDULER_WINDOW", "DSpark scheduler acceptance window, 4 when unset or zero"},
	{"DS4_DSPARK_SCHEDULER_SKIP", "DSpark scheduler base and long skip lengths"},
	{"DS4_DSPARK_SCHEDULER_SLOW_SKIP", "DSpark scheduler skip after a slow window"},
	{"DS4_DSPARK_SCHEDULER_NO_DRAFT_SKIP", "DSpark scheduler skip after a window with no draft"},
	{"DS4_DSPARK_SCHEDULER_SHORT_ACCEPT_NO_DRAFT_SKIP", "DSpark scheduler skip after a short accept with no draft"},
	{"DS4_DSPARK_SCHEDULER_COLD_LOW_CONFIDENCE_SKIP", "DSpark scheduler skip after a cold low-confidence window"},
	{"DS4_DSPARK_SCHEDULER_COLD_LOW_CONFIDENCE_MILLI", "DSpark scheduler cold confidence threshold in thousandths"},
	{"DS4_DSPARK_SCHEDULER_MIN_AVG_MILLI", "DSpark scheduler minimum average accepted length in thousandths"},
	{"DS4_DSPARK_SCHEDULER_MAX_MS_PER_ACCEPT_MILLI", "DSpark scheduler time-per-accept ceiling, off when zero"},
	{"DS4_DSPARK_SCHEDULER_MAX_EXTRA_SAVED_RATIO_MILLI", "DSpark scheduler extra-to-saved ratio ceiling, off when zero"},
	{"DS4_DSPARK_SCHEDULER_BREAK_EVEN_WINDOW", "DSpark scheduler break-even window, off when zero"},
	{"DS4_DSPARK_SCHEDULER_TAIL_MIN_TOKENS", "DSpark scheduler minimum remaining tokens before drafting"},
	{"DS4_METAL_DSPARK_COMPILE_GATE", "locked opt-in Metal DSpark library compile packet, zero dispatches"},
	{"DS4_METAL_DSPARK_PACKET_DIR", "sealed library directory for the Metal DSpark compile packet"},
	{"DS4_METAL_DSPARK_DENSE_GATE", "locked opt-in Metal DSpark dense primitive gate, not the support transformer"},
	{"DS4_METAL_DSPARK_OPERATOR_GATE", "locked opt-in model-free Metal DSpark operator gates"},
	{"DS4_METAL_DSPARK_FIXTURE_DIR", "sealed numerical fixture directory for the Metal DSpark operator and request-head gates"},
	{"DS4_METAL_DSPARK_REQUEST_HEAD_GATE", "locked physical Metal DSpark request-head gate"},
	{"DS4_METAL_DSPARK_REQUEST_LIBRARY_GATE", "locked Metal DSpark request-head library and shader mutant compile, no command buffers"},

	{"DS4_METAL_PREFILL_CHUNK", "Metal streaming context reserve prefill chunk"},
	{"DS4_METAL_GRAPH_RAW_CAP", "Metal streaming context reserve raw KV cap"},
	{"DS4_TEST_METAL_CONTEXT_ORACLE", "pinned C oracle for the Metal streaming context reserve"},
	{"DS4_METAL_DISABLE_STREAMING_EXPERT_HOTLIST", "skipping the Metal streaming expert preload when set at all"},
	{"DS4_METAL_STREAMING_EXPERT_HOTLIST", "hot expert list file the Metal streaming preload reads"},
	{"DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP", "cap on experts the Metal streaming preload admits"},
	{"DS4_METAL_SSD_ADDRESS_TEST", "locked Metal SSD selected-expert address numerical packet"},
	{"DS4_METAL_SSD_ADDRESS_OUTPUT", "fresh artifact directory for the Metal SSD address packet"},
	{"DS4_SSD_ADDRESS_FIXTURE_OUTPUT", "writing the SSD address fixture results rather than only comparing them"},
	{"DS4_CUDA_ADDRESS_NATIVE", "native CUDA expert address oracle, which allocates a device"},

	{"DS4_Q4_ORACLE_DIR", "host-compiled CUDA Q4 batch fragments against the C oracle sources"},
	{"DS4_Q4_BATCH_DEVICE", "synthetic full-width CUDA Q4 batch orchestration on a device"},
	{"DS4_Q4_BATCH_MUTATION", "same-site mutant for the CUDA Q4 batch device test, never production dispatch"},
	{"DS4_Q4_BATCH_OUT", "writing the CUDA Q4 batch device vectors for bit-for-bit repeat comparison"},
	{"DS4_PROBE_DEVICE_CUBIN", "sm121 CUDA probe CUBIN qualification with an empty NVRTC cache"},
	{"DS4_PROBE_CACHE", "fresh empty user cache directory the CUBIN probe requires"},
	{"DS4_PROBE_MAPS", "where the CUBIN probe writes /proc/self/maps"},
	{"DS4_PROBE_SOURCE", "alternate runtime source for the CUDA probe cleanup wiring check"},
	{"DS4_POWER_ORACLE_SOURCE", "the external power oracle source compared by the poweroracle gate"},

	{"DS4_GROUP_PHYSICAL", "locked physical CUDA shape grouping gate"},
	{"DS4_GROUP_RECEIPT", "receipt directory for the physical shape grouping gate"},
	{"DS4_GROUP_SOURCE", "alternate shape grouping kernel source for the contract and physical gates"},
	{"DS4_SHAPE_ORACLE", "upstream ds4 checkout pinned for the shape grouping contract"},
	{"DS4_SHAPE_NATIVE", "native CUDA shape attention and router value gates"},
	{"DS4_SHAPE_ADMISSION", "reviewed outer admission required by the native shape value gates"},
	{"DS4_SHAPE_RECEIPT", "sealed output directory for the native shape value gates"},
	{"DS4_SHAPE_ATTENTION_NATIVE_SOURCE", "alternate attention kernel source for the native shape value gate"},
	{"DS4_SHAPE_ROUTER_NATIVE_SOURCE", "alternate router kernel source for the native shape value gate"},
	{"DS4_SHAPE_ATTENTION_TEST_SOURCE", "alternate source for the shape attention contract"},
	{"DS4_SHAPE_DECODE_SOURCE", "alternate source for the full shape decode schedule"},
	{"DS4_SHAPE_DECODE_TEST_SOURCE", "alternate source for the shape decode step schedule"},
	{"DS4_SHAPE_FFN_TEST_SOURCE", "alternate source for the shape FFN schedule"},
	{"DS4_SHAPE_HC_TEST_SOURCE", "alternate source for the shape HC contract"},
	{"DS4_SHAPE_HC_MIX_TEST_SOURCE", "alternate source for the shape HC mix contract"},
	{"DS4_SHAPE_INDEXER_TEST_SOURCE", "alternate source for the shape indexer contract"},
	{"DS4_SHAPE_OUTPUT_TEST_SOURCE", "alternate source for the shape output contract"},
	{"DS4_SHAPE_ROPE_TEST_SOURCE", "alternate source for the shape inverse RoPE argument check"},
	{"DS4_SHAPE_ROUTER_TEST_SOURCE", "alternate source for the shape router contract"},
	{"DS4_PRO_STATE_NATIVE", "native CUDA production State owner value gate"},
	{"DS4_PRO_STATE_ADMISSION", "reviewed outer admission required by the native State gate"},
	{"DS4_PRO_STATE_RECEIPT", "receipt directory for the native State gate"},
	{"DS4_PRO_STATE_LIBRARIES", "sealed library manifest the native State gate pins"},
	{"DS4_SUPPORT_CODEC_NATIVE", "CUDA and Metal support snapshot round trips through the native adapters"},
	{"DS4_SUPPORT_METHOD_RECEIPTS", "absolute directory for the caller fixture control receipts"},

	{"DS4_BENCH_CONTEXT", "how much context the Metal workload benchmarks allocate"},
	{"DS4_BENCH_CHUNKS", "which prefill chunk sizes the workload benchmark sweeps"},
	{"DS4_BENCH_DEPTHS", "which decode depths the workload benchmark sweeps"},

	{"DS4_REQUIRE_GATES", "failing this test when any gate above is unset"},
}

// TestOracleGates reports which oracle gates this run exercises. Run it
// with -v to read the summary; set DS4_REQUIRE_GATES to make an unset gate
// a failure, which is what a manual parity gate wants.
func TestOracleGates(t *testing.T) {
	var set, unset []string
	for _, g := range oracleGates {
		if g.name == "DS4_REQUIRE_GATES" {
			continue
		}
		if os.Getenv(g.name) != "" {
			set = append(set, g.name)
		} else {
			unset = append(unset, g.name+" — "+g.enables)
		}
	}
	t.Logf("%d of %d oracle gates set", len(set), len(set)+len(unset))
	for _, name := range set {
		t.Logf("  set:   %s", name)
	}
	for _, line := range unset {
		t.Logf("  unset: %s", line)
	}
	if os.Getenv("DS4_REQUIRE_GATES") == "" {
		if len(set) == 0 {
			t.Log("no oracle comparison ran; a pass here is not evidence of parity")
		}
		return
	}
	if len(unset) > 0 {
		t.Errorf("DS4_REQUIRE_GATES is set but %d gates are not:\n  %s",
			len(unset), strings.Join(unset, "\n  "))
	}
}

// ds4EnvName rejects a trailing underscore: "DS4_DSPARK_" is a prefix the
// tests strip from a child environment, not a variable anything reads.
var ds4EnvName = regexp.MustCompile(`^DS4_[A-Z0-9_]*[A-Z0-9]$`)

// TestOracleGatesDocumented keeps oracleGates honest. A gate the tests read
// but the table omits is invisible in the summary, which is exactly the
// failure the summary exists to prevent.
func TestOracleGatesDocumented(t *testing.T) {
	listed := make(map[string]bool, len(oracleGates))
	for _, g := range oracleGates {
		if listed[g.name] {
			t.Errorf("%s is listed twice", g.name)
		}
		listed[g.name] = true
	}
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	used := make(map[string]bool)
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".go" {
			continue
		}
		src, err := os.ReadFile(e.Name())
		if err != nil {
			t.Fatal(err)
		}
		names, err := oracleGateNames(e.Name(), src)
		if err != nil {
			t.Fatal(err)
		}
		for _, name := range names {
			used[name] = true
		}
	}
	var missing []string
	for name := range used {
		if !listed[name] {
			missing = append(missing, name)
		}
	}
	sort.Strings(missing)
	if len(missing) > 0 {
		t.Errorf("read but not listed in oracleGates: %s", strings.Join(missing, " "))
	}
	for _, g := range oracleGates {
		if !used[g.name] {
			t.Errorf("%s is listed but no test reads it", g.name)
		}
	}
}

// oracleGateNames excludes comments, embedded kernel source, the CUDA
// macro names used by constant-mirroring tests and the oracleGates table
// itself, whose names would otherwise count as their own readers. Gate names
// are complete Go string literals, including names passed through helpers.
func oracleGateNames(filename string, src []byte) ([]string, error) {
	f, err := parser.ParseFile(token.NewFileSet(), filename, src, 0)
	if err != nil {
		return nil, err
	}
	var names []string
	ast.Inspect(f, func(n ast.Node) bool {
		if spec, ok := n.(*ast.ValueSpec); ok && len(spec.Names) == 1 && spec.Names[0].Name == "oracleGates" {
			return false
		}
		lit, ok := n.(*ast.BasicLit)
		if !ok || lit.Kind != token.STRING {
			return true
		}
		value, err := strconv.Unquote(lit.Value)
		// These name compile-time CUDA and Metal macros, not environment controls.
		switch value {
		case "DS4_MOE_GATHER_TILE", "DS4_MOE_IQ2_GATHER_TILE", "DS4_MOE_Q2_GATHER_TILE", "DS4_MOE_MAX_EXPERTS",
			"DS4_BENCH_MOE_TILES", "DS4_BENCH_TOKENS", "DS4_CUDA_PC_PROFILE_PHASE",
			"DS4_METAL_HC_STABLE", "DS4_METAL_NORM_RSQRT_DISABLE":
			return true
		}
		if err == nil && ds4EnvName.MatchString(value) {
			names = append(names, value)
		}
		return true
	})
	return names, nil
}

func TestOracleGateNamesMoEConstants(t *testing.T) {
	source := `package p
var names = []string{"DS4_MOE_GATHER_TILE", "DS4_MOE_MAX_EXPERTS", "DS4_DSPARK_", "DS4_CUDA_PREFILL_GATHER_GATE"}
var oracleGates = []oracleGate{{"DS4_LISTED_ONLY", "a table entry is not a reader"}}`
	names, err := oracleGateNames("fixture.go", []byte(source))
	if err != nil {
		t.Fatal(err)
	}
	if len(names) != 1 || names[0] != "DS4_CUDA_PREFILL_GATHER_GATE" {
		t.Fatalf("names = %q, want only the gather gate", names)
	}
}
