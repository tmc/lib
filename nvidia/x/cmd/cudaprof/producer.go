package main

import (
	"cmp"
	"encoding/json"
	"fmt"
	"runtime/debug"
	"slices"
	"strings"
	"sync"
)

// The capability record states what the producer can do. Until this file it did
// not state which producer, and the two are not separable: a cudaprof built
// before a capability existed reports its absence in the same words a current
// one uses to report that the runtime lacks it. A reader holding only the
// artifact cannot tell those apart, and the failure is quiet -- the record is
// well formed, the field is present, the value is wrong about the only question
// it was consulted for.
//
// The case that prompted it: a verification run on a remote host would have used
// a cudaprof preinstalled on that host's PATH, built before the layout probe
// this record reports on existed. It would have printed a clean layout_check
// describing a producer nobody was asking about.

// producerIdentity is what cudaprof knows about its own build.
//
// Every field is omitted when the build does not carry it. Absence is the
// honest rendering: a field reading "unknown" invites a reader to treat it as a
// value, whereas a missing producer block says plainly that the artifact cannot
// answer who wrote it. A binary built with -buildvcs=false, or from a source
// tree with no VCS data, is exactly such a binary.
type producerIdentity struct {
	Revision string `json:"revision,omitempty"` // vcs.revision, the commit built from
	Modified bool   `json:"modified,omitempty"` // vcs.modified: the tree had uncommitted changes
	Time     string `json:"time,omitempty"`     // vcs.time, the commit's timestamp
	Version  string `json:"version,omitempty"`  // module version, when built as a dependency
}

var buildProducer = sync.OnceValue(func() producerIdentity {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return producerIdentity{}
	}
	var p producerIdentity
	if v := info.Main.Version; v != "" && v != "(devel)" {
		p.Version = v
	}
	for _, s := range info.Settings {
		switch s.Key {
		case "vcs.revision":
			p.Revision = s.Value
		case "vcs.time":
			p.Time = s.Value
		case "vcs.modified":
			p.Modified = s.Value == "true"
		}
	}
	return p
})

// empty reports whether the build carried no identity at all. A producer block
// with nothing in it is worse than none: it looks like an answer.
func (p producerIdentity) empty() bool { return p == producerIdentity{} }

// stampProducer adds cudaprof's own build identity to a capability record.
//
// The agent writes the record and names itself in agent_revision; this names
// the binary that read it, which is a separate artifact that goes stale
// separately -- the agent is a .so resolved next to the executable, and either
// can be older than the other. Both identities are needed to say what a
// capability record describes.
//
// A record this cannot parse is returned unchanged. Losing a capability record
// to a stamping failure would trade a record that does not say who wrote it for
// no record at all, which is the worse of the two.
//
// The identity is a parameter rather than read from the build here so the
// stamping can be tested: a test binary carries no vcs.revision of its own, so
// a stampProducer that consulted the build directly would be exercised only in
// its no-op arm and would pass with the body deleted.
func stampProducer(capability []byte, p producerIdentity) []byte {
	if len(capability) == 0 || p.empty() {
		return capability
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(capability, &m); err != nil {
		return capability
	}
	stamp, err := json.Marshal(p)
	if err != nil {
		return capability
	}
	m["producer"] = stamp
	out, err := json.Marshal(m)
	if err != nil {
		return capability
	}
	return out
}

// agentRevision is the source revision the agent stamped into a capability
// record, or "" when the agent's build did not supply one.
//
// A narrow struct rather than the full record, and "" for both an absent field
// and an empty one, because for this caller they are the same fact: the agent
// supplied no identity. Callers that must separate a field that is absent from
// one that decoded to its zero value cannot use this -- they have to read the
// bytes, which is what control_test.go's capabilityJSON exists for.
func agentRevision(capability []byte) string {
	if len(capability) == 0 {
		return ""
	}
	var rec struct {
		AgentRevision string `json:"agent_revision"`
	}
	if err := json.Unmarshal(capability, &rec); err != nil {
		return ""
	}
	return rec.AgentRevision
}

// contributingProducer is one process whose records a capture merged, and the
// agent that wrote them.
type contributingProducer struct {
	process  uint64
	revision string // "" when that agent's build did not name itself
	// records is what that process contributed to the merged timeline: its
	// count of activity records carrying device work, the same quantity the
	// published activity_records field carries. Markers and capture-range
	// edges travel in the same record stream and are not counted, so this is
	// smaller than the journal's total record count.
	records uint64
}

// producerScope states which of a capture's producers the capability record
// describes. It is written only when that is not all of them.
type producerScope struct {
	// Described is the revision whose capabilities the record states. Omitted
	// when that agent named no revision: the scope is still worth stating --
	// the producers disagree -- and naming one side as "" would read as a
	// revision rather than as silence.
	Described string          `json:"described,omitempty"`
	Others    []otherProducer `json:"others"`
	Note      string          `json:"note"`
}

// otherProducer is one contributing process the capability record does not
// describe. Records counts what that process contributed to the merged set,
// in the same quantity -- and under the same name -- as the activity_records
// field published for the capture and for each worker.
type otherProducer struct {
	Process  uint64 `json:"process"`
	Revision string `json:"revision,omitempty"`
	Records  uint64 `json:"activity_records"`
}

// stampProducerScope states, in the capability record, when the record does
// not describe every producer whose records the capture merged.
//
// A capture can span processes: each loads its own agent, writes its own
// journal, and the session merges their activity records into one profile. The
// capability record does not merge -- it is taken from the target's journal
// alone -- so it describes one agent and accompanies the work of several. When
// they are the same build that is exactly right and nothing needs saying. When
// they are not, the record makes a claim about a collection it only partly
// covers, and it makes it in the same words either way: this is the defect
// agent_revision was added to fix, one level up from the single journal.
//
// The omission is stated rather than repaired. Reporting several agents'
// capabilities would be a schema change to a published record; saying that the
// record covers some of the capture is the honest form of what is known now.
//
// Nothing is written when no disagreement is visible, including when no agent
// named itself: an empty scope block on every capture is noise, and noise in
// the field that exists to be believed is worse than silence.
func stampProducerScope(capability []byte, described string, others []contributingProducer) []byte {
	var undescribed []otherProducer
	for _, o := range others {
		if o.revision == described {
			continue
		}
		undescribed = append(undescribed, otherProducer{Process: o.process, Revision: o.revision, Records: o.records})
	}
	if len(capability) == 0 || len(undescribed) == 0 {
		return capability
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(capability, &m); err != nil {
		return capability
	}
	slices.SortFunc(undescribed, func(a, b otherProducer) int { return cmp.Compare(a.Process, b.Process) })
	scope, err := json.Marshal(producerScope{
		Described: described,
		Others:    undescribed,
		Note:      producerScopeNote(described, undescribed),
	})
	if err != nil {
		return capability
	}
	m["producer_scope"] = scope
	out, err := json.Marshal(m)
	if err != nil {
		return capability
	}
	return out
}

// producerScopeNote is the sentence a reader who is not querying the fields
// will see. It names the count, the revisions, and the records, so that "this
// capture had one agent" is distinguishable from "we did not look".
func producerScopeNote(described string, others []otherProducer) string {
	var revisions []string
	var unnamed int
	var records uint64
	for _, o := range others {
		records += o.Records
		if o.Revision == "" {
			unnamed++
			continue
		}
		if !slices.Contains(revisions, o.Revision) {
			revisions = append(revisions, o.Revision)
		}
	}
	slices.Sort(revisions)
	who := "an agent that names no revision"
	if described != "" {
		who = "agent revision " + described
	}
	var b strings.Builder
	fmt.Fprintf(&b, "this record describes %s; %d other process(es) contributing %d record(s) ran a different agent", who, len(others), records)
	if len(revisions) > 0 {
		fmt.Fprintf(&b, " (%s)", strings.Join(revisions, ", "))
	}
	if unnamed > 0 {
		fmt.Fprintf(&b, ", %d of them an agent that names no revision", unnamed)
	}
	b.WriteString("; their capabilities are not stated here")
	return b.String()
}
