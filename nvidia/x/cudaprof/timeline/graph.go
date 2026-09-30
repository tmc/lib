package timeline

// GraphEvent is an observed graph API lifecycle event. Handles are scoped to
// Journal, a process generation. Success must be supplied from the API result;
// absence is unknown. Graph and Executable are native handles, not CUPTI graph
// membership IDs. Child graphs require separately retained relationships.
type GraphEvent struct {
	Sequence                                        uint64 // source journal record sequence
	Journal                                         string
	Domain                                          string
	TimestampDomain                                 string // monotonic when native and CUPTI clocks are unified; unknown otherwise
	CallbackID, Context                             uint32
	Process, Thread                                 uint32
	NodeHandle, SourceGraphHandle, SourceNodeHandle uint64
	NodeID, SourceNodeID                            uint64 // CUPTI IDs queried from explicitly observed resource handles
	Kind                                            string // create, instantiate, update_graph, update, update_exec, replay, destroy_graph, destroy_exec
	Graph, Executable                               uint64
	Correlation                                     uint32
	Start, End                                      int64
	Success                                         *bool
	Child                                           bool
}

// GraphIdentity names one lifetime and update version of a native handle.
// Generation is the one-based creation occurrence in the observed journal.
type GraphIdentity struct {
	Journal             string
	Handle              uint64
	Generation, Version uint64
}

// GraphLink describes a lifecycle event without inferring missing identities.
// Event indexes the input. Status is matched, missing, ambiguous or unsupported.
// Source names the graph version observed at instantiation or update; later
// graph changes do not rewrite an already instantiated executable.
type GraphLink struct {
	Event                     int
	Status                    string
	Graph, Executable, Source GraphIdentity
	// Children preserves graph versions explicitly copied into child nodes.
	Children []GraphIdentity
}

// GraphLifecycle resolves explicitly observed successful lifecycle events in
// input order. It refuses unordered events, duplicate live creations, unknown
// API results and unavailable child-graph relationships. Replay needs an observed live
// executable and a nonzero submission correlation; time never supplies identity.
// Events must include every mutation to qualify an executable version.
func GraphLifecycle(events []GraphEvent) []GraphLink {
	type key struct {
		journal string
		handle  uint64
	}
	type state struct {
		id, source GraphIdentity
		invalid    bool
		children   []GraphIdentity
	}
	graphs, execs := map[key]state{}, map[key]state{}
	type childNode struct {
		parent GraphIdentity
		source state
	}
	childNodes := map[key]childNode{}
	generations := map[key]uint64{}
	execGenerations := map[key]uint64{}
	last := map[string]int64{}
	type submission struct {
		journal, domain string
		correlation     uint32
	}
	counts := map[submission]int{}
	for _, e := range events {
		if e.Kind == "replay" && e.Correlation != 0 {
			counts[submission{e.Journal, e.Domain, e.Correlation}]++
		}
	}
	out := make([]GraphLink, 0, len(events))
	for i, e := range events {
		link := GraphLink{Event: i, Status: "missing"}
		gk, ek := key{e.Journal, e.Graph}, key{e.Journal, e.Executable}
		if previous, ok := last[e.Journal]; ok && e.Start < previous {
			link.Status = "ambiguous"
			// Ordering loss poisons all known identities in this generation.
			for k, s := range graphs {
				if k.journal == e.Journal {
					s.invalid = true
					graphs[k] = s
				}
			}
			for k, s := range execs {
				if k.journal == e.Journal {
					s.invalid = true
					execs[k] = s
				}
			}
			out = append(out, link)
			continue
		}
		last[e.Journal] = e.Start
		if e.Journal == "" || e.End < e.Start || e.Success == nil || (e.Child && e.Kind != "child") {
			link.Status = "unsupported"
			if e.Kind == "update_exec" || e.Kind == "update" || e.Kind == "instantiate" || e.Kind == "destroy_exec" || e.Kind == "destroy_graph" || e.Kind == "update_graph" || e.Kind == "create" || e.Kind == "child" || e.Kind == "clone" {
				if s, ok := execs[ek]; ok {
					s.invalid = true
					execs[ek] = s
				}
				if e.Kind == "update_graph" || e.Kind == "destroy_graph" || e.Kind == "create" || e.Kind == "child" || e.Kind == "clone" {
					if s, ok := graphs[gk]; ok {
						s.invalid = true
						graphs[gk] = s
					}
				}
			}
			out = append(out, link)
			continue
		}
		if !*e.Success {
			link.Status = "unsupported"
			out = append(out, link)
			continue
		}
		g, gok := graphs[gk]
		x, xok := execs[ek]
		switch e.Kind {
		case "child":
			child, ok := graphs[key{e.Journal, e.SourceGraphHandle}]
			if !gok || !ok || e.NodeHandle == 0 {
				if gok {
					g.invalid = true
					graphs[gk] = g
				}
				break
			}
			if g.invalid || child.invalid || e.Graph == e.SourceGraphHandle {
				g.invalid = true
				graphs[gk] = g
				link.Status = "ambiguous"
				break
			}
			g.id.Version++
			childNodes[key{e.Journal, e.NodeHandle}] = childNode{g.id, child}
			g.children = append(append([]GraphIdentity(nil), g.children...), child.id)
			g.children = append(g.children, child.children...)
			graphs[gk] = g
			link.Graph, link.Source, link.Children, link.Status = g.id, child.id, append([]GraphIdentity(nil), g.children...), "matched"
		case "child_copy":
			child, ok := childNodes[key{e.Journal, e.NodeHandle}]
			parent, live := graphs[key{e.Journal, child.parent.Handle}]
			if !ok || !live || parent.id.Generation != child.parent.Generation || e.Graph == 0 {
				break
			}
			if parent.invalid || child.source.invalid {
				link.Status = "ambiguous"
				break
			}
			if gok {
				if g.invalid || g.source != child.source.id {
					link.Status = "ambiguous"
					break
				}
			} else {
				generations[gk]++
				g = state{id: GraphIdentity{e.Journal, e.Graph, generations[gk], 1}, source: child.source.id, children: append([]GraphIdentity(nil), child.source.children...)}
				graphs[gk] = g
			}
			link.Graph, link.Source, link.Status = g.id, child.source.id, "matched"
		case "clone":
			original, ok := graphs[key{e.Journal, e.SourceGraphHandle}]
			if !ok || e.Graph == 0 {
				break
			}
			if gok || original.invalid {
				link.Status = "ambiguous"
				if gok {
					g.invalid = true
					graphs[gk] = g
				}
				break
			}
			generations[gk]++
			g = state{id: GraphIdentity{e.Journal, e.Graph, generations[gk], 1}, source: original.id, children: append([]GraphIdentity(nil), original.children...)}
			graphs[gk] = g
			link.Graph, link.Source, link.Children, link.Status = g.id, original.id, append([]GraphIdentity(nil), g.children...), "matched"
		case "create":
			if e.Graph == 0 {
				link.Status = "unsupported"
				break
			}
			if gok {
				g.invalid = true
				graphs[gk] = g
				link.Status = "ambiguous"
				break
			}
			generations[gk]++
			g = state{id: GraphIdentity{e.Journal, e.Graph, generations[gk], 1}}
			graphs[gk] = g
			link.Graph = g.id
			link.Status = "matched"
		case "instantiate":
			if e.Executable == 0 || e.Graph == 0 {
				link.Status = "unsupported"
				break
			}
			if xok {
				x.invalid = true
				execs[ek] = x
				link.Status = "ambiguous"
				break
			}
			if !gok {
				break
			}
			if g.invalid {
				link.Status = "ambiguous"
				break
			}
			execGenerations[ek]++
			x = state{id: GraphIdentity{e.Journal, e.Executable, execGenerations[ek], 1}, source: g.id, children: append([]GraphIdentity(nil), g.children...)}
			execs[ek] = x
			link.Graph = g.id
			link.Executable = x.id
			link.Source = x.source
			link.Status = "matched"
		case "update_graph":
			if !gok {
				break
			}
			if g.invalid {
				link.Status = "ambiguous"
				break
			}
			g.id.Version++
			graphs[gk] = g
			link.Graph = g.id
			link.Status = "matched"
		case "update_exec":
			if !xok {
				break
			}
			if x.invalid {
				link.Status = "ambiguous"
				break
			}
			x.id.Version++
			x.source = GraphIdentity{}
			execs[ek] = x
			link.Executable = x.id
			link.Status = "matched"
		case "update":
			if !xok {
				break
			}
			if !gok || g.invalid {
				x.invalid = true
				execs[ek] = x
				if gok {
					link.Status = "ambiguous"
				}
				break
			}
			if x.invalid {
				link.Status = "ambiguous"
				break
			}
			x.id.Version++
			x.source = g.id
			x.children = append([]GraphIdentity(nil), g.children...)
			execs[ek] = x
			link.Graph = g.id
			link.Executable = x.id
			link.Source = x.source
			link.Status = "matched"
		case "replay":
			if counts[submission{e.Journal, e.Domain, e.Correlation}] > 1 {
				link.Status = "ambiguous"
				break
			}
			if !xok || e.Correlation == 0 {
				break
			}
			if x.invalid {
				link.Status = "ambiguous"
				break
			}
			link.Executable = x.id
			link.Source = x.source
			link.Status = "matched"
		case "destroy_graph":
			if !gok {
				break
			}
			link.Graph = g.id
			link.Status = "matched"
			if g.invalid {
				link.Status = "ambiguous"
			}
			delete(graphs, gk)
		case "destroy_exec":
			if !xok {
				break
			}
			link.Executable = x.id
			link.Status = "matched"
			if x.invalid {
				link.Status = "ambiguous"
			}
			delete(execs, ek)
		default:
			link.Status = "unsupported"
			if e.Kind == "unsupported" {
				for k, s := range graphs {
					if k.journal == e.Journal {
						s.invalid = true
						graphs[k] = s
					}
				}
				for k, s := range execs {
					if k.journal == e.Journal {
						s.invalid = true
						execs[k] = s
					}
				}
			}
		}
		if link.Status == "matched" && link.Executable.Handle != 0 {
			link.Children = append([]GraphIdentity(nil), x.children...)
		} else if link.Status == "matched" && link.Graph.Handle != 0 && link.Children == nil {
			link.Children = append([]GraphIdentity(nil), g.children...)
		}
		out = append(out, link)
	}
	return out
}
