package cudacache

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"sort"

	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

const (
	compilerKeyDomain   = "cudacache-compiler-key-v3"
	searchPolicyDomain  = "cudacache-search-policy-v1"
	searchReceiptDomain = "cudacache-search-receipt-v1"
)

// CompilerIdentity records the complete compilation toolchain, driver environment,
// target architecture, flags, and requested output format necessary to guarantee
// binary cache validity.
//
// Provenance names the exact tool bytes, driver version and recipe. A zero
// Provenance hashes as a recipe with no tools, so two toolchains with the
// same version strings share keys; see ProbeToolchain.
type CompilerIdentity struct {
	NVRTCVersion string                // e.g. "12.8" or "13.0"
	DriverSONAME string                // e.g. "libcuda.so.1" or loaded driver path
	TargetArch   string                // e.g. "sm_121a", "sm_100a"
	Flags        []string              // Exact compilation options
	OutputKind   nvrtcx.OutputKind     // Output kind magic
	Provenance   ToolProvenance        // exact tool bytes, driver version and recipe
	JITOptions   cudakernel.JITOptions // effective driver options, including explicit zero
}

// CompilerCacheKey computes a deterministic SHA-256 digest string hashing inputKey
// and the full compiler/environment identity: nvrtc.Version, driver SONAME,
// sm_121a/sm_100a target architecture, exact compilation flags, and output kind magic bytes.
// Driver JIT option presence and values are also bound; explicit level zero is
// distinct from unset, and positive levels have one identity regardless of
// OptLevelSet. Any shift in driver, toolchain, flags, target, output kind or
// effective JIT options produces a cache miss.
//
// The key also hashes id.Provenance.Fingerprint, so a tool library whose
// bytes change under an unchanged version string yields a different key.
// Keys computed by this version differ from those of earlier versions, which
// did not hash provenance: entries stored under an earlier key are misses.
func CompilerCacheKey(inputKey string, id CompilerIdentity) string {
	h := sha256.New()
	writeField := func(field string) {
		var size [8]byte
		binary.LittleEndian.PutUint64(size[:], uint64(len(field)))
		h.Write(size[:])
		h.Write([]byte(field))
	}

	writeField(compilerKeyDomain)
	writeField("input_key")
	writeField(inputKey)
	writeField("nvrtc_version")
	writeField(id.NVRTCVersion)
	writeField("driver_soname")
	writeField(id.DriverSONAME)
	writeField("target_arch")
	writeField(id.TargetArch)

	// Sort a copy of flags to ensure canonical flag ordering if caller did not normalize,
	// or preserve exact flags? The specification states: "exact compilation flags".
	// We hash length and each flag in order.
	writeField("flags_count")
	writeField(fmt.Sprintf("%d", len(id.Flags)))
	for _, f := range id.Flags {
		writeField("flag")
		writeField(f)
	}

	// Output kind magic bytes
	outputMagic := fmt.Sprintf("output_kind:%d", uint8(id.OutputKind))
	writeField("output_kind")
	writeField(outputMagic)

	writeField("provenance")
	writeField(id.Provenance.Fingerprint())
	opts, vals := id.JITOptions.DriverOptions()
	writeField("jit_options")
	writeField(fmt.Sprint(len(opts)))
	for i, opt := range opts {
		writeField(fmt.Sprint(int32(opt)))
		writeField(fmt.Sprint(vals[i]))
	}

	return hex.EncodeToString(h.Sum(nil))
}

// SearchPolicyID identifies a search strategy, algorithm parameters, exploration options,
// random seeds, rewrite rules, and scoring model epoch.
//
// Crucially, SearchPolicyID is strictly isolated from binary byte compatibility identities
// (such as CompilerCacheKey or cudakernel.Artifact.Digest). Multiple search policies
// may discover and reuse the exact same underlying compiled binary artifact, but their
// search receipts remain strictly bound to their respective SearchPolicyIDs.
type SearchPolicyID struct {
	Strategy     string            `json:"strategy"`                // e.g. "beam", "mcts", "saturation"
	Options      map[string]string `json:"options,omitempty"`       // e.g. {"beam_width": "8", "max_depth": "4"}
	Seed         int64             `json:"seed"`                    // Seed for deterministic exploration
	RuleSet      string            `json:"rule_set,omitempty"`      // Canonical name/hash of rewrite rules
	ScoringEpoch string            `json:"scoring_epoch,omitempty"` // Scoring model version/epoch
}

// String returns a compact, deterministic string encoding of the SearchPolicyID.
func (sp SearchPolicyID) String() string {
	h := sha256.New()
	writeField := func(field string) {
		var size [8]byte
		binary.LittleEndian.PutUint64(size[:], uint64(len(field)))
		h.Write(size[:])
		h.Write([]byte(field))
	}

	writeField(searchPolicyDomain)
	writeField("strategy")
	writeField(sp.Strategy)
	writeField("seed")
	writeField(fmt.Sprintf("%d", sp.Seed))
	writeField("rule_set")
	writeField(sp.RuleSet)
	writeField("scoring_epoch")
	writeField(sp.ScoringEpoch)

	keys := make([]string, 0, len(sp.Options))
	for k := range sp.Options {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	writeField("options_count")
	writeField(fmt.Sprintf("%d", len(keys)))
	for _, k := range keys {
		writeField(k)
		writeField(sp.Options[k])
	}

	return hex.EncodeToString(h.Sum(nil))
}

// SearchReceiptKey computes the unique key for indexing an evaluation receipt under a search policy.
func SearchReceiptKey(policyID SearchPolicyID, workloadKey string) string {
	h := sha256.New()
	writeField := func(field string) {
		var size [8]byte
		binary.LittleEndian.PutUint64(size[:], uint64(len(field)))
		h.Write(size[:])
		h.Write([]byte(field))
	}
	writeField(searchReceiptDomain)
	writeField("policy")
	writeField(policyID.String())
	writeField("workload")
	writeField(workloadKey)
	return hex.EncodeToString(h.Sum(nil))
}

// WorkloadIdentity describes a program/workload being compiled and evaluated.
// It includes shape dimensions, numerical execution mode, pre-pack identity,
// and optional canonical bindings identity (dtype, layout, objective, and
// input/weight binding identity omitted from legacy fields).
type WorkloadIdentity struct {
	ProgramID     string `json:"program_id"`
	ShapeDims     []int  `json:"shape_dims"`
	NumericalMode string `json:"numerical_mode"` // e.g. "fp32", "fp16", "bf16", "fp8_e4m3", "mxfp4"
	PrepackID     string `json:"prepack_id,omitempty"`
	BindingsKey   string `json:"bindings_key,omitempty"` // Canonical producing workload bindings key (dtype/layout/objective/input+weight identity)
}

// Key computes a deterministic hash for a workload identity.
func (w WorkloadIdentity) Key() string {
	h := sha256.New()
	writeField := func(field string) {
		var size [8]byte
		binary.LittleEndian.PutUint64(size[:], uint64(len(field)))
		h.Write(size[:])
		h.Write([]byte(field))
	}
	writeField("workload-identity-v1")
	writeField(w.ProgramID)
	writeField(w.NumericalMode)
	writeField(w.PrepackID)
	writeField(fmt.Sprintf("%v", w.ShapeDims))
	writeField("bindings_key")
	writeField(w.BindingsKey)
	return hex.EncodeToString(h.Sum(nil))
}
