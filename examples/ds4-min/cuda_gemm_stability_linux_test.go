//go:build linux

package main

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// The campaign's corruption oracle is a SHA-256 over the prefill logits, which
// is an exact-identity oracle: it is only a valid oracle for a computation that
// is bit-exact by construction. Two prefill projections go through cuBLAS and
// so are not obviously that. evalBatchFFN routes the MoE router through
// encodeF16MatvecGEMM and cudaHCBatch.pre projects the whole group in one
// GemmEx, both with cublas.GemmDefault, on the one handle newCUDAHC puts in
// TF32TensorOpMath (cuda_hc_linux.go:61). The router feeds top-k expert
// selection four lines later, so a one-ULP move at a routing boundary flips
// which expert runs and the logits then differ by far more than a ULP.
//
// These two tests are not the same experiment and should not be read as one.
//
// TestCUDAGEMMHandlePinning is the primary. It holds the kernel fixed and
// varies only the handle state that selects and bounds it, so a difference it
// finds is attributable to one knob. Algorithm selection under GemmDefault is a
// function of shape, dtype, arch, library version and available workspace, not
// a per-process draw, so "cuBLAS re-picks each run" was never the right form of
// the hypothesis. What stays live is nondeterminism with the algorithm held
// fixed -- a split-k reduction whose split count comes from occupancy at launch
// -- and selection drifting because one of its inputs, the workspace, moves
// with the host's memory state.
//
// TestCUDAGEMMStability is the coarse arm swap, kept because it is the only one
// that reaches a bit-exact reference. It changes precision policy, kernel and
// summation order together, so a difference it finds is not attributable to any
// one of them.
//
// Both alternate their arms rather than running one arm through. A fixed order
// confounds arm with position and with whatever the preceding draws did to the
// host's memory state, which is the failure mode ensureBatch's own comment
// records: at 512 per-slot cuBLAS handles the logits changed, and changed
// differently from run to run, two digests from one build. Positions are
// reported beside the digests so a period-2 or first-slot effect stays visible
// rather than being absorbed into an arm.
//
// # The workspace is the decisive factor, and it is per process
//
// Of GemmDefault's selection inputs -- shape, dtype, arch, library version and
// available workspace -- every one but the last is fixed by the build. Workspace
// availability is not: it moves with the host's memory state. A deterministic
// function of a drifting input drifts, so selection can change run to run with
// no split-k, no per-launch randomness and no per-process dice. That is the one
// mechanism this tree has already observed, in ensureBatch's comment, and it
// fits six distinct divergent digests with no repeats better than a small fixed
// outcome space does, because workspace availability is near-continuous.
//
// norpr cannot separate that from a nondeterministic kernel; both would show the
// same thing. Pinning a fixed, test-owned buffer can: with the drifting input
// removed, divergence that survives indicts the kernel and divergence that
// vanishes indicts selection, and the remedy for the second is a pinned
// workspace in the shipping path rather than any math-mode change.
//
// DS4_CUDA_GEMM_WORKSPACE picks the condition, and it is a property of the whole
// process rather than an arm in the sweep. That is forced, not preferred:
// cublasSetWorkspace hands the handle a user-owned buffer for all subsequent
// calls and cuBLAS exposes no call that gives the library-managed pool back, so
// a handle can cross this boundary once and never return. Alternating it inside
// one process would mean either a one-way phase change -- the workspace contrast
// then confounded with position, which is the thing this file exists to avoid --
// or destroying and rebuilding the batch between draws, which resets the very
// allocation state whose drift is under suspicion.
//
// So alternate it the way this repo's other ABBA campaigns alternate: across
// processes, default and pinned interleaved by the operator under one lock
// window. Everything else stays within a process and stays clean.
//
// This is also the only arm here whose application cannot be read back. There is
// no cublasGetWorkspace, so it rests on the setter's error alone, where math and
// atomics mode are both compared against what was asked for. The allocation and
// the set are therefore fatal rather than skippable: a workspace arm that fell
// through to the library default would be a silent duplicate of tf32.
//
// # The workspace contrast is between-process, and time is a covariate
//
// The within-process arms share a process, a page cache and a neighbour set, so
// the only thing that differs between two of their draws is the handle. The
// workspace contrast cannot have that: default and pinned are by construction
// two processes at two different times, and anything that drifts with the clock
// -- neighbours starting, page cache growing, a previous arm's memory not yet
// released -- drifts the two baselines apart for reasons that are not the
// workspace. memAvailableGiB is exactly such a covariate, and measuring it is
// not enough on its own, because a covariate that is partly a proxy for *when* a
// draw ran makes the draws non-exchangeable: a runs test on this tree's own
// headroom sweep read 33 runs about the median against 61.0 expected, z = -5.13,
// and block-permuting the labels within contiguous stretches of run order rather
// than globally moved a wrong-digest deficit from p=0.0002 to p=0.0935 -- an
// order of magnitude, from the same draws. The series it was measured on ranged
// 32.5 GiB across the sweep, which is the scale of drift a between-process
// contrast here is exposed to.
//
// Two consequences for how this is run and read.
//
// Interleave the conditions finely. Sixty draws of default followed by sixty of
// pinned puts an hour of clock inside the contrast; alternating single processes
// ABBA-style is the floor, and finer is better. The lock window is one window
// either way, so the interleave costs nothing but operator attention.
//
// Do not permute globally afterwards. Every draw logs its wall-clock time and
// its host headroom for that reason -- the line is key=value so it can be
// parsed straight out of the test log -- and the contrast should be assessed by
// permuting condition labels within contiguous blocks of run order. A global
// permutation assumes the exchangeability the runs test above rejects.
//
// # A partial poisoning is a different fault, and no arm here varies layout
//
// Every draw records nonFiniteExtent rather than a first index, because the two
// corruption shapes want different fixes and an instrument that reports only
// where the poisoning starts reports the same thing for both. A wholly poisoned
// vector is the memory-pressure corruption cudaDenseArtifactLayers describes. A
// single bad element, or a short run at the front, is a layout or offset fault:
// a mapping boundary, a first tile, an offset off by one.
//
// That distinction is not decoration here, it is a stop condition. Nothing in
// this campaign varies layout -- the handle arms change precision policy and the
// workspace factor changes where a scratch buffer comes from, and neither moves
// an offset. So a partial extent is a fault this harness cannot have caused and
// cannot bound, and the run should be read as having found something other than
// what it set out to measure. cudaGEMMTabulate says so in as many words rather
// than folding it into the non-finite count, which is the same merge this file
// spends its first page arguing against, one level down.
//
// # The block size is ten draws, fixed here and not after the fact
//
// Thirty-three runs over 120 draws is a mean run length of about 3.6, so ten
// draws is roughly three correlation lengths: long enough that a shuffle inside
// a block is not a shuffle within one excursion of the series, short enough that
// the permutation still respects the drift. That is the derivation; the number
// is ten.
//
// It is written here, before the run, because block size is a knob, and it is
// precisely the knob anyone would reach for if the answer came out wrong. The
// three candidate blockings of the headroom analysis gave 0.0130, 0.0286 and
// 0.0935 -- a three-fold spread, all three defensible in isolation, and a later
// reader free to choose among them after seeing them is not reading a result.
// Other block sizes may be printed as a sensitivity. Only the block-10 figure
// is the result.
//
// # A confound check can return a false clear
//
// The natural check -- are the divergent draws clustered in run order -- clears
// data that is badly confounded. In the headroom sweep the divergent draws sat
// at positions 23, 47, 50, 55, 75 and 103 of 120: thoroughly scattered, and that
// check would have certified p=0.0002. The assumption that fails is about the
// covariate, not the labels. A smooth autocorrelated series hands even perfectly
// scattered labels an extreme-looking contrast, because what makes the two
// groups differ is where the series was, not which draws were picked.
//
// So the exchangeability assumption is measured in the run rather than left to
// the reader: cudaGEMMTabulate prints a Wald-Wolfowitz runs-test z on the
// per-draw headroom about its own median, beside the table it is needed for. A
// large negative z says the draws of this run are not exchangeable and the block
// permutation is mandatory rather than cautious.
//
// # A null here does not exonerate cuBLAS
//
// GemmDefault selection is a function of shape, dtype, arch, library version
// and available workspace, and whether a GEMM splits its k at all is decided by
// the shape. A shape that does not split gives a clean zero for a reason that
// has nothing to do with whether the path is bit-defined. The oracle verdict
// rests on the written contract -- cuda_f16_default_linux_test.go:19 logs
// "cuBLAS order not certified" and compares on atol/rtol rather than a hash --
// not on any rate these draws produce. What is under test here is the
// mechanism of the corruption, not the validity of the oracle.
//
// # Why this runs at two artifact layers, and why not at eight
//
// The harness fixes the configuration at cudaGEMMMoEArtifactLayers rather than
// at the shipping default, and that is a power decision. But the rate to size it
// against is not the one cudaMoEArtifactDefaultLayers' comment records, because
// that comment's 1/30, 5/30 and 9/30 are a *merged* bucket. checkLogits returns
// on the first non-finite value before it forms the digest -- logits_check.go:31
// -- so the outcome space is three-way, clean against wrong digest against
// non-finite, and the recorded rates collapse the last two. Decomposed over the
// same 120 draws:
//
//	layers  non-finite  wrong digest  merged
//	     0        0/30          1/30    1/30
//	     2        1/30          4/30    5/30
//	     8        7/30          1/30    9/30
//
// The merged column reproduces the comment exactly, so this refines it rather
// than disputing it.
//
// The class under test here is the wrong digest. The mechanism on trial is a
// near-tie at a top-k routing boundary flipping an expert, which produces a
// wrong digest, not a non-finite. So the eight-layer cell -- which anyone
// reading the merged column will propose, because 9/30 is the largest number on
// the page -- is the *quietest* cell available for this class at 1/30, tied with
// zero layers, and it would cost 19.12 GiB and a tenth of its draws collapsing
// to a fifth of the throughput to move there. Its 9/30 is almost entirely the
// other class. Stay at two.
//
// That is worth stating on its own account: the merged bucket is monotone in
// bytes held because *one* of its two classes is monotone, and reading the
// monotone total as the rate of the phenomenon you care about inverts the
// ranking of the cells. A rate is only usable once its outcome space is
// partitioned, which is also why cudaGEMMTabulate below keeps non-finite draws
// out of the digest tabulation rather than letting them enter it as a second
// distinct digest. That exclusion is load-bearing beyond tidiness: a SHA-256
// over NaN bit patterns is itself deterministic, so a corruption that reproduces
// its bit pattern produces a *repeating* digest, and a few such draws in an arm
// do not merely add a distinct digest -- they can outnumber the clean one and
// become the arm's modal digest, at which point the majority reading inverts and
// the clean draws are the ones reported as drift.
//
// So the base rate is 4/30, and an ablation that drove it to a clean zero would
// read p=0.056 one-sided by Fisher's exact test: C(56,30)/C(60,30) = (30·29·28·
// 27)/(60·59·58·57). Against the shipping default's 1/30 the same clean zero is
// p=0.5, a whole lock window spent to learn nothing -- and worse, that null
// would read as evidence against the amplifier story when it is only evidence
// that the cell was too quiet. Power lives where the base rate is, so do not
// "simplify" this harness by running it at the shipping default either.
//
// The same arithmetic run backwards is the floor check cudaGEMMFisherFloor
// applies to every arm before any rate is read off it. An arm of n draws that
// came back a perfect zero can produce no p smaller than C(30,4)/C(30+n,4), so
// at n=30 the best attainable value is 0.056 and at n=1 it is 0.87. A p equal to
// that floor is not a result about the arm; it is the arm's size, restated. Two
// arms in the headroom analysis this file's workspace premise came from returned
// exactly their floors and were written up as corroboration.

// cudaGEMMDraw is one prefill: which arm produced it, where in the sequence it
// fell, and what came back. nonFinite separates the two failure classes, which
// is the whole reason this type carries it rather than folding a poisoned run
// into its digest: see the decomposition at the top of this file.
type cudaGEMMDraw struct {
	arm       string
	position  int
	digest    string
	token     int
	nonFinite bool
	extent    logitExtent
	elapsed   time.Duration
	// at and availGiB are the covariates the workspace contrast needs and the
	// within-process arms do not: that contrast spans two processes at two
	// times, so a draw has to carry when it ran and what the host's memory
	// looked like when it did. They are recorded, never asserted on.
	at       time.Time
	availGiB float64
}

// cudaGEMMLog is the one line a draw emits, key=value so the sweep can be
// parsed out of the test log and block-permuted afterwards. Every draw prints
// the same fields in the same order, clean or poisoned, because a record whose
// shape depends on the outcome cannot be read as a table.
func (d cudaGEMMDraw) cudaGEMMLog(workspace string, tokens int) string {
	outcome, digest := "clean", d.digest
	if len(digest) > 16 {
		digest = digest[:16]
	}
	if d.nonFinite {
		outcome, digest = "nonfinite", "-"
	}
	return fmt.Sprintf("draw=%d arm=%s workspace=%s outcome=%s digest=%s extent=%v token=%d elapsed=%v tokPerSec=%.2f availGiB=%.1f at=%s",
		d.position, d.arm, workspace, outcome, digest, d.extent, d.token, d.elapsed,
		float64(tokens)/d.elapsed.Seconds(), d.availGiB, d.at.UTC().Format(time.RFC3339Nano))
}

// cudaGEMMFixture is the state every draw in either test starts from: a loaded
// runtime whose batch resources, and so whose cuBLAS handle, already exist.
type cudaGEMMFixture struct {
	r         *cudaRuntime
	tokens    []int
	rounds    int
	workspace string
}

// cudaGEMMWorkspaceBytes is the size to hand cublasSetWorkspace, chosen to match
// the library's own default pool for the architecture so that the pinned arm
// differs from the default arm in where the buffer comes from and not in how
// much of it there is. cuBLAS documents the default pool as 4 MiB below compute
// 9.0 and 32 MiB at and above it.
func cudaGEMMWorkspaceBytes(major int) uint {
	if major >= 9 {
		return 32 << 20
	}
	return 4 << 20
}

// cudaGEMMMoEArtifactLayers is the MoE artifact depth every draw runs at. It is
// two, not cudaMoEArtifactDefaultLayers, which is zero: see the power argument
// at the top of this file. Nothing may override it, because an ablation and a
// change of cell landing in the same run is two variables, not one.
const cudaGEMMMoEArtifactLayers = 2

// cudaGEMMFixedEnv is the configuration every draw is pinned to, written by the
// harness rather than inherited. The three kernel flags are the ones that ship
// on and that cudaMoEArtifactDefaultLayers' safety measurement was taken with;
// holding them fixed is what keeps the handle the only thing that varies.
var cudaGEMMFixedEnv = map[string]string{
	"DS4_CUDA_MOE_D2R":             "1",
	"DS4_CUDA_DENSE_D2R":           "1",
	"DS4_CUDA_ATTENTION_HMMA":      "1",
	"DS4_CUDA_MOE_ARTIFACT_LAYERS": strconv.Itoa(cudaGEMMMoEArtifactLayers),
}

// cudaGEMMRefusedEnv would move the configuration out from under the arms. Each
// is refused rather than overridden: a harness that quietly won an argument with
// its operator's environment produces a number nobody can place afterwards. The
// three DISABLE_ doors are here because they are read before the enable flags
// and so would silently undo cudaGEMMFixedEnv.
var cudaGEMMRefusedEnv = []string{
	"DS4_CUDA_DISABLE_MOE_D2R",
	"DS4_CUDA_DISABLE_DENSE_D2R",
	"DS4_CUDA_DISABLE_ATTN_HMMA",
	"DS4_CUDA_MOE_ARTIFACT_LAYERS",
	"DS4_CUDA_DENSE_ARTIFACT_LAYERS",
	"DS4_CUDA_MOE_D2R",
	"DS4_CUDA_DENSE_D2R",
	"DS4_CUDA_ATTENTION_HMMA",
	"DS4_CUDA_DISABLE_LIBRARY_GEMM",
	"DS4_CUDA_DISABLE_ATTN_QKV_D2R",
	"DS4_CUDA_DISABLE_RAW_ATTENTION_BATCH",
	"DS4_CUDA_DISABLE_COMPRESSOR_BATCH",
	"DS4_CUDA_DISABLE_CHUNKED_ARENA",
	"DS4_CUDA_HC_BATCH",
	"DS4_CUDA_BALLAST_GIB",
	"DS4_DISABLE_CUDA_GRAPH",
}

// cudaGEMMSetup pins the configuration, loads the model, runs one unmeasured
// prefill and hands back the runtime. The warm-up is not politeness: ensureBatch
// creates the cuBLAS handle, so there is nothing to pin before it, and
// cudaDenseArtifactLayers records that an unfaulted page gives the first
// evaluation wrong bytes and leaves every later one correct. A measured draw 0
// would carry that on one arm only.
func cudaGEMMSetup(t *testing.T) *cudaGEMMFixture {
	t.Helper()
	gate := os.Getenv("DS4_CUDA_GEMM_STABILITY_GATE")
	if gate == "" {
		t.Skip("DS4_CUDA_GEMM_STABILITY_GATE is not set")
	}
	if gate != "1" || os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("DS4_CUDA_GEMM_STABILITY_GATE=1 and a GPU lock are required")
	}
	model, input := os.Getenv("DS4_MODEL"), os.Getenv("DS4_CUDA_PREFILL_TOKENS")
	if model == "" || input == "" {
		t.Fatal("DS4_MODEL and DS4_CUDA_PREFILL_TOKENS are required")
	}
	// One round draws every arm twice, so n draws per arm is rounds = n/2: the
	// n=60 draw this exists for is DS4_CUDA_GEMM_STABILITY_ROUNDS=30. Two is a
	// smoke run and nothing more; no rate can be read off it.
	rounds := 2
	if s := os.Getenv("DS4_CUDA_GEMM_STABILITY_ROUNDS"); s != "" {
		n, err := strconv.Atoi(s)
		if err != nil || n < 1 {
			t.Fatalf("DS4_CUDA_GEMM_STABILITY_ROUNDS = %q, want a positive integer", s)
		}
		rounds = n
	}
	// A property of the process, not an arm: cuBLAS gives no way back to the
	// library-managed pool once a buffer is handed over. Alternate it by running
	// alternating processes under one lock window.
	workspace := "default"
	if s := os.Getenv("DS4_CUDA_GEMM_WORKSPACE"); s != "" {
		if s != "default" && s != "pinned" {
			t.Fatalf("DS4_CUDA_GEMM_WORKSPACE = %q, want \"default\" or \"pinned\"", s)
		}
		workspace = s
	}
	// Refused, not overridden. t.Setenv below would win over most of these and
	// lose to the DISABLE_ doors, and either way the operator would be reading a
	// configuration they did not get.
	for _, name := range cudaGEMMRefusedEnv {
		if v, ok := os.LookupEnv(name); ok {
			t.Fatalf("%s=%q is set; this harness fixes the kernel configuration itself so that the "+
				"cuBLAS handle is the only thing that varies across its arms. Unset it and re-run.", name, v)
		}
	}
	for name, value := range cudaGEMMFixedEnv {
		t.Setenv(name, value)
	}
	data, err := os.ReadFile(input)
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := parsePrefillTokens(data)
	if err != nil {
		t.Fatal(err)
	}
	// Exactly one prefill group, so every draw captures and replays the same
	// graphs and nothing varies with the group count.
	if len(tokens) < cudaPrefillGroupWidth {
		t.Fatalf("input contains %d tokens, want at least %d", len(tokens), cudaPrefillGroupWidth)
	}
	tokens = tokens[:cudaPrefillGroupWidth]

	r, err := newCUDARuntime(model, 4096)
	if err != nil {
		t.Fatal(err)
	}
	// One cleanup for both, because the order matters. What the buffer needs is
	// that no work is still in flight against it, which is a synchronize and not
	// a close; closing first tears down the device the free then needs, and the
	// free fails with "nil or closed device". Every pinned process failed this
	// way while reporting a stable digest, so the arm looked clean in the table
	// and failed in the exit status.
	var pinned cudamem.Slice[byte]
	t.Cleanup(func() {
		if pinned.Len() > 0 {
			if err := r.device.WithContext(driver.CtxSynchronize); err != nil {
				t.Error(err)
			}
			if err := pinned.Free(); err != nil {
				t.Error(err)
			}
		}
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})
	// Setting an environment variable is not the same as the runtime reading it.
	// The configuration is read back off the runtime and the run refuses to
	// produce draws under one it did not ask for -- a rate is only interpretable
	// against the cell it was measured in.
	t.Logf("configuration: moeD2R=%v denseD2R=%v attnHMMA=%v attnQKVD2R=%v denseArtifactLayers=%d moeArtifactLayers=%d groupWidth=%d rounds=%d drawsPerArm=%d",
		r.d2rEnabled(), r.denseD2REnabled(), r.attentionHMMAEnabled(), r.attnQKVD2REnabled(),
		cudaDenseArtifactLayers(), cudaMoEArtifactLayers(), cudaPrefillGroupWidth, rounds, 2*rounds)
	if !r.d2rEnabled() || !r.denseD2REnabled() || !r.attentionHMMAEnabled() {
		t.Fatalf("kernel flags did not take: moeD2R=%v denseD2R=%v attnHMMA=%v, want all true",
			r.d2rEnabled(), r.denseD2REnabled(), r.attentionHMMAEnabled())
	}
	if got := cudaMoEArtifactLayers(); got != cudaGEMMMoEArtifactLayers {
		t.Fatalf("MoE artifact layers = %d, want %d: this is the cell whose base rate the draw count was sized for",
			got, cudaGEMMMoEArtifactLayers)
	}

	if err := r.Reset(); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	if err := r.EvalBatch(tokens, 0); err != nil {
		t.Fatalf("warm-up prefill: %v", err)
	}
	digest, nonFinite := prefillLogitDigest(r.Logits())
	t.Logf("warm-up (not counted): digest=%s nonFinite=%v elapsed=%v", digest[:16], nonFinite, time.Since(start))

	// The handle exists only after ensureBatch, so this is the first moment the
	// workspace can be pinned, and it is pinned once for the life of the
	// process. Both the allocation and the set are fatal: falling through to the
	// library pool would leave a run labelled "pinned" that is a duplicate of
	// the default condition, and nothing downstream could tell.
	if workspace == "pinned" {
		b := r.activeHCBatch()
		if b == nil {
			t.Fatal("no batch HC after the warm-up prefill: there is no handle to pin a workspace on")
		}
		size := cudaGEMMWorkspaceBytes(r.major)
		pinned, err = cudamem.Alloc[byte](r.device, int(size))
		if err != nil {
			t.Fatalf("allocate %d-byte cuBLAS workspace: %v", size, err)
		}
		if err := r.device.WithContext(func() error {
			return b.h.blas.SetWorkspace(driver.DevicePtr(pinned.Ptr()), size)
		}); err != nil {
			t.Fatalf("SetWorkspace(%d bytes): %v", size, err)
		}
		t.Logf("workspace: pinned to a test-owned %d-byte buffer at sm_%d%d; cuBLAS exposes no getter, "+
			"so this rests on the setter's error and not on a readback", size, r.major, r.minor)
	} else {
		t.Logf("workspace: library-managed pool (the shipping condition); alternate single processes " +
			"with DS4_CUDA_GEMM_WORKSPACE=pinned rather than running the two conditions in blocks, " +
			"and block-permute the labels within contiguous stretches of run order when reading the result")
	}
	return &cudaGEMMFixture{r: r, tokens: tokens, rounds: rounds, workspace: workspace}
}

// draw resets, checks that the reset can be seen, applies arm, and prefills.
//
// Reset destroys the batch graphs, which is what makes an in-process arm change
// mean anything: both cuBLAS calls sit inside the evalBatchFFN closure that is
// cudagraph.Captured, and cuBLAS consults the handle when it selects a kernel,
// which is at capture. Without a fresh capture every later draw would replay the
// first draw's kernels and report one arm twice under several names. The check
// below is that dependency made to fail loudly rather than silently.
func (f *cudaGEMMFixture) draw(t *testing.T, arm string, position int, apply func(*testing.T) bool) (cudaGEMMDraw, bool) {
	t.Helper()
	if err := f.r.Reset(); err != nil {
		t.Fatalf("draw %d: reset: %v", position, err)
	}
	if f.r.batch != nil && f.r.batch.graphs[0].ffn != nil {
		t.Fatalf("draw %d: Reset left layer 0's ffn graph instantiated; "+
			"a change made between draws cannot reach a replayed capture", position)
	}
	if !apply(t) {
		return cudaGEMMDraw{}, false
	}
	// Headroom is sampled before the prefill rather than after, because what
	// bears on cuBLAS's workspace is the state the allocation is made under.
	avail := memAvailableGiB()
	start := time.Now()
	if err := f.r.EvalBatch(f.tokens, 0); err != nil {
		t.Fatalf("draw %d arm %s: prefill: %v", position, arm, err)
	}
	elapsed := time.Since(start)
	logits := f.r.Logits()
	if len(logits) != 129280 {
		t.Fatalf("draw %d arm %s: logits length %d, want 129280", position, arm, len(logits))
	}
	digest, nonFinite := prefillLogitDigest(logits)
	d := cudaGEMMDraw{arm, position, digest, greedyToken(logits), nonFinite.bad(), nonFinite, elapsed, start, avail}
	// A non-finite draw is an outcome of the cell, not a fault in the harness,
	// and at this artifact depth it is an outcome with a measured rate. Marking
	// it failed here would abort the sweep that exists to measure it, so it is
	// counted and carried; the accounting happens once, at the end.
	if d.nonFinite {
		t.Logf("%s (excluded from the digest tabulation)", d.cudaGEMMLog(f.workspace, len(f.tokens)))
		return d, true
	}
	t.Logf("%s", d.cudaGEMMLog(f.workspace, len(f.tokens)))
	return d, true
}

// cudaGEMMFisherFloor is the smallest one-sided Fisher p an arm of n draws can
// produce against the cell's base rate even if every one of those draws comes
// back clean: the probability that all cudaGEMMBaseEvents events land in the
// reference's cudaGEMMBaseDraws draws rather than in the arm's n.
//
// It is printed beside every rate because a p equal to this number carries one
// bit -- that the arm was a perfect zero -- and none about how large the effect
// is. At n=30 the floor is 0.056; at n=1 it is 0.87, which no amount of cleanness
// can improve on.
const (
	cudaGEMMBaseDraws  = 30
	cudaGEMMBaseEvents = 4
)

func cudaGEMMFisherFloor(n int) float64 {
	p := 1.0
	for i := range cudaGEMMBaseEvents {
		p *= float64(cudaGEMMBaseDraws-i) / float64(cudaGEMMBaseDraws+n-i)
	}
	return p
}

// cudaGEMMTabulate reports, per arm, the three-way outcome -- clean draws by
// distinct digest, and non-finite draws counted apart -- and returns the arms
// that produced more than one digest and the arms that produced a non-finite.
//
// The two classes are kept apart because they have different mechanisms and, in
// the artifact-depth data, opposite trends: see the decomposition at the top of
// this file, where a merged rate inverts the ranking of the cells. A non-finite
// run still hashes to something, so folding it in would enter it as a second
// distinct digest and report a poisoned run as a near-tie flip.
//
// A count alone cannot separate an unstable arm from one whose draws happened to
// sit in a contended window; the positions can, so they are printed too.
func cudaGEMMTabulate(t *testing.T, draws []cudaGEMMDraw, arms []string) (unstable, poisoned map[string]bool) {
	t.Helper()
	seen := make(map[string]map[string][]int)
	nonFinite := make(map[string][]int)
	extents := make(map[string][]logitExtent)
	total := make(map[string]int)
	for _, d := range draws {
		total[d.arm]++
		if d.nonFinite {
			nonFinite[d.arm] = append(nonFinite[d.arm], d.position)
			extents[d.arm] = append(extents[d.arm], d.extent)
			continue
		}
		if seen[d.arm] == nil {
			seen[d.arm] = make(map[string][]int)
		}
		seen[d.arm][d.digest] = append(seen[d.arm][d.digest], d.position)
	}
	unstable, poisoned = make(map[string]bool), make(map[string]bool)
	for _, arm := range arms {
		byDigest := seen[arm]
		digests := make([]string, 0, len(byDigest))
		for digest := range byDigest {
			digests = append(digests, digest)
		}
		sort.Slice(digests, func(i, j int) bool { return len(byDigest[digests[i]]) > len(byDigest[digests[j]]) })
		wrong := 0
		for _, digest := range digests[min(1, len(digests)):] {
			wrong += len(byDigest[digest])
		}
		floor := cudaGEMMFisherFloor(total[arm])
		t.Logf("arm %-9s %d draws: %d finite over %d distinct digest(s), %d non-finite; wrongDigest=%d/%d nonFinite=%d/%d floorP=%.3f",
			arm, total[arm], total[arm]-len(nonFinite[arm]), len(digests), len(nonFinite[arm]),
			wrong, total[arm], len(nonFinite[arm]), total[arm], floor)
		if floor > 0.05 {
			t.Logf("    n=%d cannot reach p<0.05 against the %d/%d base rate even at a perfect zero, so read "+
				"this arm's rate as a description and not as a test", total[arm], cudaGEMMBaseEvents, cudaGEMMBaseDraws)
		}
		for _, digest := range digests {
			t.Logf("    %s x%d at draws %v", digest, len(byDigest[digest]), byDigest[digest])
		}
		if at := nonFinite[arm]; len(at) > 0 {
			t.Logf("    non-finite x%d at draws %v extents %v", len(at), at, extents[arm])
			poisoned[arm] = true
			// Total against partial, not merged. A vector poisoned end to end
			// is the memory-pressure corruption this cell is known to produce;
			// anything short of that is a layout or offset fault, which no arm
			// in this campaign varies and so nothing here can have caused.
			for _, e := range extents[arm] {
				if e.n == e.len {
					continue
				}
				what := "a partial poisoning"
				if e.n == 1 {
					what = "a single non-finite element"
				}
				t.Logf("    LAYOUT: arm %s has %s (%v). Nothing in this campaign varies layout, so this "+
					"is not the corruption the run was sized to measure and its rate does not belong "+
					"in the same column as a wholly poisoned draw", arm, what, e)
			}
		}
		if len(digests) > 1 {
			unstable[arm] = true
		}
	}
	// Balance is reported, not asserted. Forward-then-reverse puts arm i at
	// position i in one pass and at len-1-i in the other, so every arm's mean
	// position is the same by construction -- but an arm that went unavailable
	// mid-sweep, or a list the caller shortened, breaks that silently. A run
	// whose arms sit at different mean positions has arm correlated with
	// position, and its comparison is worth less than it looks.
	mean := make(map[string]float64)
	count := make(map[string]int)
	for _, d := range draws {
		mean[d.arm] += float64(d.position)
		count[d.arm]++
	}
	var lo, hi float64
	first := true
	for _, arm := range arms {
		if count[arm] == 0 {
			continue
		}
		mean[arm] /= float64(count[arm])
		if first || mean[arm] < lo {
			lo = mean[arm]
		}
		if first || mean[arm] > hi {
			hi = mean[arm]
		}
		first = false
	}
	spread := hi - lo
	parts := make([]string, 0, len(arms))
	for _, arm := range arms {
		parts = append(parts, fmt.Sprintf("%s=%.1f", arm, mean[arm]))
	}
	t.Logf("mean draw position per arm: %s (spread %.1f)", strings.Join(parts, " "), spread)

	// Exchangeability, measured rather than assumed. This is about the sweep as
	// a whole and not about any arm: it says whether the covariate these draws
	// were taken against drifted, which decides whether a later analysis may
	// permute labels globally or must permute within blocks of
	// cudaGEMMPermutationBlock draws.
	if runs, expected, z, ok := cudaGEMMRunsZ(draws); ok {
		t.Logf("availGiB runs test over %d draws: runs=%d expected=%.1f z=%.2f; permutation block=%d draws (pre-registered)",
			len(draws), runs, expected, z, cudaGEMMPermutationBlock)
		if z < -2 {
			t.Logf("    the headroom series is autocorrelated in this run, so the draws are not exchangeable: " +
				"permute condition labels within blocks, never globally, and do not read a clustering check " +
				"on the divergent draws as a clear")
		}
	} else {
		t.Logf("availGiB runs test: not defined for this run (too few draws off the median); "+
			"permutation block=%d draws (pre-registered)", cudaGEMMPermutationBlock)
	}
	if spread > 1 {
		t.Logf("WARNING: the arms did not sit at equal mean positions, so arm and position are " +
			"correlated in this run and a difference between arms may be a difference between slots")
	}
	return unstable, poisoned
}

// cudaGEMMPermutationBlock is the pre-registered permutation block size, in
// draws. See the derivation at the top of this file: it is fixed before the run
// because choosing it after seeing the p-values is the knob that makes a
// confounded result look clean.
const cudaGEMMPermutationBlock = 10

// cudaGEMMRunsZ is the Wald-Wolfowitz runs statistic for the per-draw host
// headroom about its own median, in draw order: the number of runs observed,
// the number expected under exchangeability, and the z between them. Draws
// sitting exactly on the median are dropped, which is what the test's null
// distribution assumes.
//
// It tests the covariate, not the outcome, which is the whole point -- the
// clustering check that tests the outcome clears data a smooth series has
// already confounded. ok is false when there are too few draws on one side for
// the statistic to be defined, which is a real answer and not a zero.
func cudaGEMMRunsZ(draws []cudaGEMMDraw) (runs int, expected, z float64, ok bool) {
	avail := make([]float64, 0, len(draws))
	for _, d := range draws {
		avail = append(avail, d.availGiB)
	}
	sorted := append([]float64(nil), avail...)
	sort.Float64s(sorted)
	if len(sorted) < 4 {
		return 0, 0, 0, false
	}
	median := sorted[len(sorted)/2]
	if len(sorted)%2 == 0 {
		median = (sorted[len(sorted)/2-1] + sorted[len(sorted)/2]) / 2
	}
	var n1, n2 int
	last := 0
	for _, v := range avail {
		side := 0
		switch {
		case v > median:
			side, n1 = 1, n1+1
		case v < median:
			side, n2 = -1, n2+1
		default:
			continue
		}
		if side != last {
			runs++
			last = side
		}
	}
	if n1 < 2 || n2 < 2 {
		return runs, 0, 0, false
	}
	a, b := float64(n1), float64(n2)
	n := a + b
	expected = 2*a*b/n + 1
	variance := 2 * a * b * (2*a*b - n) / (n * n * (n - 1))
	if variance <= 0 {
		return runs, expected, 0, false
	}
	return runs, expected, (float64(runs) - expected) / math.Sqrt(variance), true
}

// cudaGEMMDigests is the set of digests one arm produced.
func cudaGEMMDigests(draws []cudaGEMMDraw, arm string) map[string]bool {
	set := make(map[string]bool)
	for _, d := range draws {
		// A poisoned run's digest is a hash of NaNs and says nothing about
		// which kernel ran, so it must not enter an engagement comparison.
		if d.arm == arm && !d.nonFinite {
			set[d.digest] = true
		}
	}
	return set
}

// cudaGEMMPinArm is one setting of the handle. Both fields are named on every
// arm and set on every draw: an arm that left one of them alone would inherit
// whatever the previous draw set, which under an alternating order is a
// carry-over from the other arm.
type cudaGEMMPinArm struct {
	name    string
	math    cublas.Math
	atomics cublas.AtomicsMode
}

// TestCUDAGEMMHandlePinning draws repeated prefills with the cuBLAS handle held
// in several states, alternating the arms, and reports whether each state is
// bit-stable and whether it moves the digest at all.
//
// The arms, and what each one would settle:
//
//	tf32      the shipping state, TF32TensorOpMath: ten mantissa bits.
//	fp32      DefaultMath. Twenty-four bits. The proposed mechanism is a
//	          discrete amplifier at a top-k routing boundary, and near-ties are
//	          far more likely at ten bits than at twenty-four, so a divergence
//	          rate that moves here supports it and one that does not hurts it.
//	norpr     TF32 with DisallowReducedPrecisionReduction. This is the one that
//	          names split-k directly: it constrains the reduction of the
//	          partials, which is where a split-k GEMM's run-to-run freedom is.
//	pedantic  PedanticMath, which pins away tensor-op substitution generally.
//	noatomic  AtomicsNotAllowed, set explicitly. Read the measured default
//	          logged below before reading anything into this arm: cuBLAS
//	          documents AtomicsNotAllowed as the default for a fresh handle, and
//	          documents the mode as governing the alternate symv/hemv
//	          implementations rather than GEMM split-k. If the default already
//	          reads AtomicsNotAllowed then this arm changes nothing and cannot
//	          discriminate, and the test says so rather than counting it.
//
// DS4_CUDA_GEMM_PIN_ARMS selects a comma-separated subset; the default is the
// tf32/fp32 pair, which is the cheapest sharp comparison.
func TestCUDAGEMMHandlePinning(t *testing.T) {
	f := cudaGEMMSetup(t)
	r := f.r

	b := r.activeHCBatch()
	if b == nil {
		t.Fatal("no batch HC after the warm-up prefill: neither cuBLAS call site is reachable, " +
			"so pinning the handle cannot change this configuration")
	}
	h := b.h.blas

	// The defaults are measured, not assumed. Whether the shipping handle
	// already forbids atomics decides whether the noatomic arm is a control or
	// a no-op, and that is a fact about this library version on this host.
	baseMath, err := h.MathMode()
	if err != nil {
		t.Fatalf("MathMode: %v", err)
	}
	baseAtomics, err := h.AtomicsMode()
	if err != nil {
		t.Fatalf("AtomicsMode: %v", err)
	}
	t.Logf("shipping handle state as measured: math=%v atomics=%v", baseMath, baseAtomics)
	if baseAtomics == cublas.AtomicsNotAllowed {
		t.Log("the shipping handle already forbids atomics, so the noatomic arm sets nothing " +
			"and an agreement between it and tf32 is not evidence about split-k")
	}

	all := []cudaGEMMPinArm{
		{"tf32", cublas.TF32TensorOpMath, baseAtomics},
		{"fp32", cublas.DefaultMath, baseAtomics},
		{"norpr", cublas.TF32TensorOpMath | cublas.MathDisallowReducedPrecisionReduction, baseAtomics},
		{"pedantic", cublas.PedanticMath, baseAtomics},
		{"noatomic", cublas.TF32TensorOpMath, cublas.AtomicsNotAllowed},
	}
	want := []string{"tf32", "fp32"}
	if s := os.Getenv("DS4_CUDA_GEMM_PIN_ARMS"); s != "" {
		want = strings.Split(s, ",")
	}
	var arms []cudaGEMMPinArm
	for _, name := range want {
		name = strings.TrimSpace(name)
		found := false
		for _, a := range all {
			if a.name == name {
				arms, found = append(arms, a), true
				break
			}
		}
		if !found {
			t.Fatalf("DS4_CUDA_GEMM_PIN_ARMS names %q; the arms are tf32 fp32 norpr pedantic noatomic", name)
		}
	}
	if len(arms) == 0 {
		t.Fatal("DS4_CUDA_GEMM_PIN_ARMS selects no arms")
	}
	// One arm is a legitimate and in fact the cheapest way to run the workspace
	// experiment, which is a between-process contrast: tf32 alone under
	// DS4_CUDA_GEMM_WORKSPACE=default against tf32 alone under pinned is the
	// decisive pair, and it costs half what carrying a second handle arm costs.
	// What it gives up is the within-process engagement check, so the run says
	// so rather than reporting an unchecked green.
	if len(arms) == 1 {
		t.Logf("one arm (%s): this run makes no within-process comparison and cannot check that the "+
			"handle state reached the kernel. Its result is only interpretable against a second "+
			"process run with the other DS4_CUDA_GEMM_WORKSPACE condition.", arms[0].name)
	}

	// Both setters are error-checked and both values are read back. Neither
	// check subsumes the other: a rejected setter leaves the prior value in
	// place, so a readback that is not compared against what was asked for
	// agrees with itself and reports success.
	apply := func(a cudaGEMMPinArm) func(*testing.T) bool {
		return func(t *testing.T) bool {
			t.Helper()
			if err := r.device.WithContext(func() error {
				if err := h.SetMathMode(a.math); err != nil {
					return fmt.Errorf("SetMathMode(%v): %w", a.math, err)
				}
				if err := h.SetAtomicsMode(a.atomics); err != nil {
					return fmt.Errorf("SetAtomicsMode(%v): %w", a.atomics, err)
				}
				got, err := h.MathMode()
				if err != nil {
					return err
				}
				if got != a.math {
					return fmt.Errorf("math mode reads %v after setting %v", got, a.math)
				}
				gotAtomics, err := h.AtomicsMode()
				if err != nil {
					return err
				}
				if gotAtomics != a.atomics {
					return fmt.Errorf("atomics mode reads %v after setting %v", gotAtomics, a.atomics)
				}
				return nil
			}); err != nil {
				t.Errorf("arm %s unavailable on this library: %v", a.name, err)
				return false
			}
			return true
		}
	}

	// Forward then reverse over the arm list each round, which is the n-arm
	// form of ABBA: every arm sits in every half of a round equally often, so
	// arm and position are orthogonal by construction rather than by luck.
	var draws []cudaGEMMDraw
	names := make([]string, len(arms))
	for i, a := range arms {
		names[i] = a.name
	}
	unavailable := make(map[string]bool)
	for round := range f.rounds {
		for pass := range 2 {
			for i := range arms {
				a := arms[i]
				if pass == 1 {
					a = arms[len(arms)-1-i]
				}
				if unavailable[a.name] {
					continue
				}
				d, ok := f.draw(t, a.name, len(draws), apply(a))
				if !ok {
					unavailable[a.name] = true
					continue
				}
				draws = append(draws, d)
			}
		}
		t.Logf("round %d complete: %d draws", round, len(draws))
	}

	var broken []string
	for _, name := range names {
		if unavailable[name] {
			broken = append(broken, fmt.Sprintf("arm %s produced no draws, so the run's denominator is short an arm", name))
		}
	}
	unstable, poisoned := cudaGEMMTabulate(t, draws, names)
	for _, name := range names {
		if unstable[name] {
			broken = append(broken, fmt.Sprintf("arm %s is not bit-stable across its finite draws", name))
		}
		// Reported as its own class, never merged into the digest count. A run
		// carrying both has produced two findings, not one.
		if poisoned[name] {
			broken = append(broken, fmt.Sprintf("arm %s produced non-finite logits, which is the other "+
				"corruption class and not a near-tie flip", name))
		}
	}

	// Engagement. An arm whose digests are all also tf32's changed nothing the
	// prefill read. For norpr and noatomic that is a legitimate result -- cuBLAS
	// may not have been using a reduced-precision reduction or an atomic path
	// for these shapes in the first place -- so it is reported, not raised. What
	// is raised is every arm agreeing with tf32, because then no arm was
	// engaged and a uniformly stable table is one arm measured several times.
	base := cudaGEMMDigests(draws, "tf32")
	engaged := 0
	for _, name := range names {
		if name == "tf32" {
			continue
		}
		moved := false
		for digest := range cudaGEMMDigests(draws, name) {
			if !base[digest] {
				moved = true
			}
		}
		if moved {
			engaged++
		}
		t.Logf("arm %-9s %s tf32", name, map[bool]string{true: "moves the digest away from", false: "agrees with"}[moved])
	}
	if len(names) > 1 && len(base) > 0 && engaged == 0 && !unavailable["tf32"] {
		broken = append(broken, "no arm moved the digest away from tf32: the handle state never reached the "+
			"kernel, so no arm was measured")
	}

	stable := len(unstable) == 0
	t.Logf("verdict over %d rounds, %d draws per arm, arms alternated forward and reverse, workspace=%s: %s",
		f.rounds, len(draws)/max(len(names), 1), f.workspace,
		map[bool]string{
			true:  "every arm bit-stable; on this host, in this memory state, the handle's precision policy is not a source of run-to-run digest drift",
			false: "at least one arm is not bit-stable; the arms it holds fixed bound where the drift comes from",
		}[stable])
	if len(broken) > 0 {
		t.Fatalf("handle pinning:\n\t%s", strings.Join(broken, "\n\t"))
	}
}

// TestCUDAGEMMStability is the coarse arm swap: LIB is the shipping path and
// HAND is DS4_CUDA_DISABLE_LIBRARY_GEMM=1, which takes the per-token and
// hand-written arms. It is the secondary experiment because the two arms differ
// in precision policy, kernel and summation order at once, so a difference
// between them is attributable to none of the three.
//
// It is kept because it is the only comparison that reaches an arm with a
// bit-exactness promise: TestCUDAHCBatchExact records that the per-token arm
// reproduces the scalar reference bit for bit, where the group-at-once arm
// promises a bound. HAND is not, however, a cuBLAS-free arm -- cuda_hc_linux.go
// still issues one GemmEx per row under it -- so a stable HAND beside an
// unstable LIB indicts wide-N algorithm selection, not the library.
func TestCUDAGEMMStability(t *testing.T) {
	f := cudaGEMMSetup(t)
	r := f.r

	arms := []struct {
		name    string
		disable bool
	}{{"LIB", false}, {"HAND", true}}
	names := []string{"LIB", "HAND"}
	order := []int{0, 1, 1, 0}
	var draws []cudaGEMMDraw
	for round := range f.rounds {
		for _, a := range order {
			arm := arms[a]
			d, _ := f.draw(t, arm.name, len(draws), func(*testing.T) bool {
				r.disableLibraryGEMM = arm.disable
				return true
			})
			draws = append(draws, d)
		}
		t.Logf("round %d complete: %d draws", round, len(draws))
	}

	var broken []string
	unstable, poisoned := cudaGEMMTabulate(t, draws, names)
	for _, name := range names {
		if unstable[name] {
			broken = append(broken, fmt.Sprintf("arm %s is not bit-stable across its finite draws", name))
		}
		// Reported as its own class, never merged into the digest count. A run
		// carrying both has produced two findings, not one.
		if poisoned[name] {
			broken = append(broken, fmt.Sprintf("arm %s produced non-finite logits, which is the other "+
				"corruption class and not a near-tie flip", name))
		}
	}

	// The control. The prefill digest ledger in TestCUDAPrefillCUDAGraph records
	// that moving the router onto cuBLAS moved the digest, so these two arms are
	// known to disagree. If they agree on every draw the ablation did not apply,
	// and two stable-looking arms are one arm measured twice.
	shared := 0
	lib := cudaGEMMDigests(draws, "LIB")
	for digest := range cudaGEMMDigests(draws, "HAND") {
		if lib[digest] {
			shared++
		}
	}
	if shared > 0 {
		broken = append(broken, fmt.Sprintf("%d digest(s) common to both arms: the disableLibraryGEMM "+
			"ablation did not change the computation, so neither arm was measured", shared))
	}

	verdict := map[[2]bool]string{
		{false, false}: "both arms bit-stable: any digest divergence seen elsewhere has another source",
		{true, false}:  "LIB unstable, HAND stable: the drift is in the group-at-once library path",
		{false, true}:  "LIB stable, HAND unstable: the drift is not wide-N algorithm selection",
		{true, true}:   "both arms unstable: the cause is below both projections",
	}[[2]bool{unstable["LIB"], unstable["HAND"]}]
	t.Logf("verdict over %d rounds, %d draws per arm, ABBA, workspace=%s: %s", f.rounds, f.rounds*2, f.workspace, verdict)

	if len(broken) > 0 {
		t.Fatalf("GEMM stability:\n\t%s", strings.Join(broken, "\n\t"))
	}
}
