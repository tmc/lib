package main

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
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
	{"DS4_MODEL", "every CUDA kernel, runtime and model comparison"},
	{"DS4_GGUF_PROFILE_MODEL", "the pinned GGUF profile and Metal model tests"},
	{"DS4_GGUF_MANIFEST_MODEL", "the tensor manifest digest"},
	{"DS4_GGUF_MANIFEST_OUT", "writing the manifest rather than comparing it"},
	{"DS4_GGUF_TOKENIZER_MODEL", "tokenizer round-trip against the model's vocabulary"},

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
	{"DS4_METAL_GRAPH_DUMP_TRACE", "dumping the Metal graph trace"},
	{"DS4_METAL_LAYER13_FIXTURE_DIR", "the layer 13 Metal regression"},
	{"DS4_METAL_HC_LAYER13_FIXTURE_DIR", "the layer 13 HC Metal regression"},
	{"DS4_METAL_HC_LAYER13_OUTPUT_DIR", "writing layer 13 HC fixtures rather than comparing them"},
	{"DS4_METAL_COUNTER_CALIBRATION", "the model-free Metal GPU counter calibration"},

	{"DS4_OBSERVE", "the observation harness"},
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
	{"DS4_CUDA_PHASE_PROFILE", "CUDA phase timing with stream events"},
	{"DS4_CUDA_PHASE_PROFILE_OUT", "new CUDA phase timing result file"},
	{"DS4_CUDA_PREFILL_PROGRESS", "CUDA EvalBatch progress diagnostic"},
	{"DS4_CUDA_PREFILL_PROGRESS_OUT", "new CUDA prefill progress JSONL file"},
	{"DS4_CUDA_HC_MATH_DIAGNOSTIC", "the CUDA HC mathematical diagnostic"},

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

var ds4EnvName = regexp.MustCompile(`DS4_[A-Z0-9_]+`)

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
		for _, name := range ds4EnvName.FindAllString(string(src), -1) {
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
