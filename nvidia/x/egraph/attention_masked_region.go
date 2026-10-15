package egraph

import (
	"errors"
	"slices"

	"github.com/tmc/lib/nvidia/x/cudair"
)

// AttentionRowMask is a validated predicate for a row of at most 64 keys.
// Its zero value is invalid. Constructors own their inputs; accessors return
// copies. A predicate describes which source scores must already be -Inf;
// it does not modify scores or assert that a graph contains a mask operator.
type AttentionRowMask struct {
	kind     string
	query    int
	prefix   int
	parents  []int
	excluded []bool
}

// NewCausalAttentionRowMask includes keys 0 through query, inclusive.
// keys must be in [1,64] and query in [0,keys).
func NewCausalAttentionRowMask(query, keys int) (AttentionRowMask, error) {
	if keys < 1 || keys > 64 || query < 0 || query >= keys {
		return AttentionRowMask{}, ErrInvalidMaskIndex
	}
	mask := AttentionRowMask{kind: "causal", query: query, excluded: make([]bool, keys)}
	for key := range mask.excluded {
		mask.excluded[key] = key > query
	}
	return mask, nil
}

// NewPrefixAttentionRowMask includes the first prefix keys.
// keys must be in [1,64] and prefix in [0,keys]. A zero prefix excludes all keys.
func NewPrefixAttentionRowMask(prefix, keys int) (AttentionRowMask, error) {
	if keys < 1 || keys > 64 || prefix < 0 || prefix > keys {
		return AttentionRowMask{}, ErrInvalidMaskIndex
	}
	mask := AttentionRowMask{kind: "prefix", prefix: prefix, excluded: make([]bool, keys)}
	for key := range mask.excluded {
		mask.excluded[key] = key >= prefix
	}
	return mask, nil
}

// NewTreeAttentionRowMask includes a shared prefix and the query's ancestors,
// including query itself. Keys are the prefix followed by len(parents) tree
// nodes. query and parents index tree nodes, without the prefix offset.
// Each parent must be -1 (a root) or an earlier tree node; this validates the
// entire forest, including branches outside the selected ancestor chain.
// prefix must be nonnegative and the total key count in [1,64].
func NewTreeAttentionRowMask(parents []int, query, prefix int) (AttentionRowMask, error) {
	if prefix < 0 || prefix > 64 || len(parents) == 0 || len(parents) > 64-prefix || query < 0 || query >= len(parents) {
		return AttentionRowMask{}, ErrInvalidMaskIndex
	}
	for node, parent := range parents {
		if parent < -1 || parent >= node {
			return AttentionRowMask{}, errors.New("egraph: tree attention parent must be -1 or an earlier node")
		}
	}
	mask := AttentionRowMask{kind: "tree-prefix", query: query, prefix: prefix,
		parents: slices.Clone(parents), excluded: make([]bool, prefix+len(parents))}
	for node := range parents {
		mask.excluded[prefix+node] = true
	}
	for node := query; node != -1; node = parents[node] {
		mask.excluded[prefix+node] = false
	}
	return mask, nil
}

// Dense returns the dense predicate: true means that a key is excluded.
func (m AttentionRowMask) Dense() []bool { return slices.Clone(m.excluded) }

// Kind returns "causal", "prefix", "tree-prefix", or "" for the zero value.
func (m AttentionRowMask) Kind() string { return m.kind }

// Definition returns the query, prefix length and owned parent list used by
// the constructor. Query is meaningful for causal and tree masks; prefix is
// meaningful for prefix and tree masks; parents is meaningful for tree masks.
func (m AttentionRowMask) Definition() (query, prefix int, parents []int) {
	return m.query, m.prefix, slices.Clone(m.parents)
}

func (m AttentionRowMask) clone() AttentionRowMask {
	m.parents = slices.Clone(m.parents)
	m.excluded = slices.Clone(m.excluded)
	return m
}

// NewCheckedMaskedOnlineAttentionCandidate derives a CPU alternative from the
// exact nine-node graph admitted by NewCheckedOnlineAttentionCandidate.
// The supplied score input must already contain -Inf at every excluded key
// and finite scores at every included key; all values must be finite.
// Neither the immutable source snapshot nor any input binding is changed.
// A fully masked row emits canonical FP32 NaNs (bits 0x7fc00000), independently
// of the reference. Their bits must exactly match the unchanged source output,
// even with nonzero tolerance; otherwise the candidate is refused. This is a
// host-checked policy, not a promise about NaN propagation on other platforms.
// Nonempty rows are checked against cudaref.EvaluateSnapshot on the exact
// copied inputs under policy. This is input-bound numerical evidence, not
// a graph equality, search discovery, hardware schedule or TMA skipping claim.
func NewCheckedMaskedOnlineAttentionCandidate(snapshot *cudair.Snapshot, inputs map[string][]float32, mask AttentionRowMask, tileSize int, policy AttentionCandidatePolicy) (*OnlineAttentionCandidate, error) {
	return newCheckedOnlineAttentionCandidate(snapshot, inputs, tileSize, policy, &mask)
}
