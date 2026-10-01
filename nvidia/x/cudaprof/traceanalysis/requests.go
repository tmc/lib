package traceanalysis

import (
	"fmt"
	"sort"
)

// Request retains explicitly scoped lifecycle observations. Executions are
// application-reported successful executions, not inferred GPU launches.
// Missing counters or boundaries remain unavailable. This is not a coverage
// or correctness receipt.
type Request struct {
	Journal         string     `json:"journal"`
	Process         uint32     `json:"process"`
	ID              string     `json:"request_id"`
	Role            string     `json:"role"`
	Endpoints       []Endpoint `json:"endpoints"`
	Spacing         []int64    `json:"delivery_spacing_ns"`
	ExecutionDeltas []int      `json:"successful_execution_deltas"`
	Unavailable     []string   `json:"unavailable"`
	Diagnostics     []string   `json:"diagnostics"`
}

func requests(endpoints []Endpoint) []Request {
	type key struct {
		journal  string
		process  uint32
		id, role string
	}
	groups := map[key][]Endpoint{}
	for _, e := range endpoints {
		if e.Metadata.Version != 2 || e.Source.Journal == "" || e.Process == 0 {
			continue
		}
		k := key{e.Source.Journal, e.Process, e.Metadata.RequestID, e.Metadata.Role}
		groups[k] = append(groups[k], e)
	}
	out := []Request{}
	for k, es := range groups {
		sort.SliceStable(es, func(i, j int) bool { return es[i].Time < es[j].Time })
		r := Request{Journal: k.journal, Process: k.process, ID: k.id, Role: k.role, Endpoints: es, Spacing: []int64{}, ExecutionDeltas: []int{}, Unavailable: []string{}}
		starts, ends := 0, 0
		var previous *Endpoint
		var lastCount *int
		seen := map[int]bool{}
		for i := range es {
			e := &es[i]
			m := e.Metadata
			if m.Event == "request_start" {
				starts++
				if m.Boundary != "submission" {
					r.Unavailable = append(r.Unavailable, "request start lacks submission boundary")
				}
			}
			if m.Event == "request_end" {
				ends++
				if m.Boundary != "completion" {
					r.Unavailable = append(r.Unavailable, "request end lacks completion boundary")
				}
			}
			if m.ExecutionUnit != "logical_forward" {
				r.Unavailable = append(r.Unavailable, "successful execution unit missing or unsupported")
			}
			if m.Executions == nil {
				r.Unavailable = append(r.Unavailable, "successful execution counter missing")
			} else {
				if lastCount != nil && *m.Executions < *lastCount {
					r.Unavailable = append(r.Unavailable, "successful execution counter decreased")
				}
				lastCount = m.Executions
			}
			if m.Boundary == "" {
				r.Unavailable = append(r.Unavailable, "endpoint boundary missing")
			}
			if m.Event != "delivered" {
				continue
			}
			if m.Token == nil {
				r.Unavailable = append(r.Unavailable, "delivered ordinal missing")
				continue
			}
			if m.Boundary != "consumer_delivery" {
				r.Unavailable = append(r.Unavailable, "delivered endpoint is not consumer_delivery")
				continue
			}
			if seen[*m.Token] {
				r.Unavailable = append(r.Unavailable, fmt.Sprintf("duplicate delivered token %d", *m.Token))
			}
			seen[*m.Token] = true
			if previous != nil {
				if *m.Token != *previous.Metadata.Token+1 {
					r.Unavailable = append(r.Unavailable, "delivered ordinals are not consecutive and increasing")
				}
				if e.Time <= previous.Time {
					r.Unavailable = append(r.Unavailable, "delivery timestamps are not strictly increasing")
				}
				r.Spacing = append(r.Spacing, e.Time-previous.Time)
				if m.Executions != nil && previous.Metadata.Executions != nil {
					delta := *m.Executions - *previous.Metadata.Executions
					if delta >= 0 {
						r.ExecutionDeltas = append(r.ExecutionDeltas, delta)
					}
					if delta != *m.Token-*previous.Metadata.Token {
						r.Diagnostics = append(r.Diagnostics, "delivered-token delta differs from successful execution delta; delivery windows contain different work")
					}
				}
			}
			previous = e
		}
		if len(seen) < 2 {
			r.Unavailable = append(r.Unavailable, "delivery spacing requires at least two delivered endpoints")
		}
		if starts != 1 || ends != 1 {
			r.Unavailable = append(r.Unavailable, "request requires one start and one end endpoint")
		}
		if starts == 1 && ends == 1 && (es[0].Metadata.Event != "request_start" || es[len(es)-1].Metadata.Event != "request_end") {
			r.Unavailable = append(r.Unavailable, "request endpoints do not enclose retained lifecycle observations")
		}
		if len(r.Unavailable) != 0 {
			r.Spacing = nil
			r.ExecutionDeltas = nil
		}
		r.Unavailable = uniqueStrings(r.Unavailable)
		r.Diagnostics = uniqueStrings(r.Diagnostics)
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.Journal != b.Journal {
			return a.Journal < b.Journal
		}
		if a.Process != b.Process {
			return a.Process < b.Process
		}
		if a.ID != b.ID {
			return a.ID < b.ID
		}
		return a.Role < b.Role
	})
	return out
}

func uniqueStrings(values []string) []string {
	out := []string{}
	seen := map[string]bool{}
	for _, value := range values {
		if !seen[value] {
			seen[value] = true
			out = append(out, value)
		}
	}
	return out
}
