package main

import (
	"encoding/json"
	"runtime/debug"
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
