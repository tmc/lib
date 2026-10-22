package kernelagent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"sort"

	"github.com/tmc/lib/nvidia/x/egraph"
)

const maxProposalBytes = 16 << 10

// ModelIdentity identifies the trusted proposal transport's configured model.
// Kind is fixed or model. A model response cannot select this identity.
type ModelIdentity struct{ Kind, Name, Version string }

// Request contains data only: no archive, compiler, executable or filesystem
// capability is given to a proposer. Prompt and diagnostic sizes are bounded.
type Request struct {
	Epoch, Operation, Prompt string
	Allowed                  []string
	Diagnostics              Summary
}

// Reply contains an untrusted JSON edit of the form {"rules":["catalog-key"]}.
type Reply struct {
	Model ModelIdentity
	Edit  json.RawMessage
}

// Proposer is implemented by a trusted transport; its returned edit is untrusted.
// Implementations must honor cancellation. OpenAI provides a Responses API client.
type Proposer interface {
	Propose(context.Context, Request) (Reply, error)
}

// FixedEnumerator returns one configured edit per epoch. Its zero value is
// exhausted. It is a proposal source, not a search algorithm.
type FixedEnumerator struct {
	Edits []json.RawMessage
	next  int
}

// Propose returns the next edit without using diagnostics.
func (f *FixedEnumerator) Propose(ctx context.Context, _ Request) (Reply, error) {
	if err := ctx.Err(); err != nil {
		return Reply{}, err
	}
	if f.next >= len(f.Edits) {
		return Reply{}, io.EOF
	}
	if len(f.Edits[f.next]) > maxProposalBytes {
		return Reply{}, fmt.Errorf("kernelagent: fixed edit exceeds proposal bound")
	}
	r := Reply{Model: ModelIdentity{"fixed", "enumerator", "1"}, Edit: bytes.Clone(f.Edits[f.next])}
	f.next++
	return r, nil
}

type ProposalEvidence struct {
	Model                ModelIdentity
	ReportedModel        *ModelIdentity `json:",omitempty"`
	Prompt, ReturnedEdit string
	ReturnedBytes        int
	Rules                []string
}

var catalogKey = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

func admitProposal(raw []byte, catalog map[string]*egraph.Rewrite) ([]*egraph.Rewrite, []string, error) {
	if len(raw) == 0 || len(raw) > maxProposalBytes {
		return nil, nil, fmt.Errorf("kernelagent: proposal size out of bounds")
	}
	// Read the single-field object explicitly, so duplicate fields and trailing
	// JSON cannot change the interpretation of a recorded edit.
	d := json.NewDecoder(bytes.NewReader(raw))
	tok, err := d.Token()
	if err != nil || tok != json.Delim('{') {
		return nil, nil, fmt.Errorf("kernelagent: proposal must be an object")
	}
	tok, err = d.Token()
	if err != nil || tok != "rules" {
		return nil, nil, fmt.Errorf("kernelagent: proposal requires only rules")
	}
	var keys []string
	if err := d.Decode(&keys); err != nil || len(keys) > 16 {
		return nil, nil, fmt.Errorf("kernelagent: invalid or oversized rule list")
	}
	tok, err = d.Token()
	if err != nil || tok != json.Delim('}') {
		return nil, nil, fmt.Errorf("kernelagent: unexpected proposal field")
	}
	if _, err := d.Token(); err != io.EOF {
		return nil, nil, fmt.Errorf("kernelagent: trailing proposal data")
	}
	seen := make(map[string]bool)
	rules := make([]*egraph.Rewrite, 0, len(keys))
	for _, key := range keys {
		r := catalog[key]
		if !catalogKey.MatchString(key) || seen[key] || r == nil || r.LHS == nil || r.RHS == nil {
			return nil, nil, fmt.Errorf("kernelagent: unknown, repeated or invalid rule %q", key)
		}
		seen[key] = true
		rules = append(rules, r)
	}
	return rules, keys, nil
}

func allowedRules(catalog map[string]*egraph.Rewrite) ([]string, error) {
	if len(catalog) > 16 {
		return nil, fmt.Errorf("kernelagent: catalog exceeds 16 rules")
	}
	var keys []string
	for key, r := range catalog {
		if !catalogKey.MatchString(key) || r == nil || r.LHS == nil || r.RHS == nil {
			return nil, fmt.Errorf("kernelagent: invalid trusted rule catalog")
		}
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys, nil
}
