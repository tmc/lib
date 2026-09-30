package dependency

import (
	"sort"

	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

type graphNode struct {
	dependencies []uint64
	receipt      timeline.DependencyEvent
}

type graphShape struct {
	nodes map[uint64]graphNode
	valid bool
}

type graphInventory struct {
	shapes    map[timeline.GraphIdentity]graphShape
	replays   map[submissionKey]timeline.GraphLink
	aliases   map[nodeIdentity]uint64
	ambiguous map[nodeIdentity]bool
}

type nodeIdentity struct {
	journal string
	id      uint64
}

type submissionKey struct {
	journal, domain string
	correlation     uint32
}

func observedGraphs(records timeline.Records) graphInventory {
	inventory := graphInventory{shapes: map[timeline.GraphIdentity]graphShape{}, replays: map[submissionKey]timeline.GraphLink{}, aliases: map[nodeIdentity]uint64{}, ambiguous: map[nodeIdentity]bool{}}
	receipts := map[submissionKey][]timeline.DependencyEvent{}
	for _, e := range records.Dependencies {
		if e.Site == "exit" && e.Kind == "graph_node" && e.Success != nil && *e.Success {
			k := submissionKey{e.Journal, e.Domain, e.Correlation}
			receipts[k] = append(receipts[k], e)
		}
	}
	links := timeline.GraphLifecycle(records.GraphLifecycle)
	for i, e := range records.GraphLifecycle {
		link := links[i]
		// Resource callbacks explicitly relate cloned node identities. Membership
		// IDs, names and execution intervals do not supply this mapping.
		if e.Kind == "resource" && e.NodeID != 0 && e.SourceNodeID != 0 {
			if prior, ok := inventory.aliases[nodeIdentity{e.Journal, e.NodeID}]; ok && prior != e.SourceNodeID {
				inventory.ambiguous[nodeIdentity{e.Journal, e.NodeID}] = true
			}
			inventory.aliases[nodeIdentity{e.Journal, e.NodeID}] = e.SourceNodeID
		}
		if link.Status != "matched" {
			continue
		}
		switch e.Kind {
		case "create":
			inventory.shapes[link.Graph] = graphShape{nodes: map[uint64]graphNode{}, valid: true}
		case "clone":
			inventory.shapes[link.Graph] = inventory.shapes[link.Source]
		case "update_graph":
			previous := link.Graph
			previous.Version--
			shape := inventory.shapes[previous]
			next := graphShape{nodes: map[uint64]graphNode{}, valid: shape.valid}
			for id, node := range shape.nodes {
				next.nodes[id] = node
			}
			nodes := receipts[submissionKey{e.Journal, e.Domain, e.Correlation}]
			if len(nodes) != 1 || nodes[0].Graph != e.Graph || nodes[0].Node == 0 {
				next.valid = false
			} else {
				node := nodes[0]
				if _, exists := next.nodes[node.Node]; exists {
					next.valid = false
				}
				for _, id := range node.Dependencies {
					if _, exists := next.nodes[id]; !exists {
						next.valid = false
					}
				}
				next.nodes[node.Node] = graphNode{append([]uint64(nil), node.Dependencies...), node}
			}
			inventory.shapes[link.Graph] = next
		case "child":
			// Lifecycle identity is retained, but full expansion needs distinct
			// observed cloned child-node identities for every child occurrence.
			inventory.shapes[link.Graph] = graphShape{valid: false}
		case "replay":
			inventory.replays[submissionKey{e.Journal, e.Domain, e.Correlation}] = link
		}
	}
	return inventory
}

func (g graphInventory) original(journal string, id uint64) (uint64, bool) {
	seen := map[uint64]bool{}
	for {
		if seen[id] || g.ambiguous[nodeIdentity{journal, id}] {
			return 0, false
		}
		seen[id] = true
		next, ok := g.aliases[nodeIdentity{journal, id}]
		if !ok || next == id {
			return id, true
		}
		id = next
	}
}

// graphExecution resolves each kernel through an observed replay identity and
// explicit graph-node IDs. It refuses incomplete shapes and duplicated nodes.
func (g graphInventory) graphExecution(e timeline.DependencyEvent, indices []int, activities map[int]timeline.Activity) (roots, leaves []int, edges [][2]int, receipts []timeline.DependencyEvent, ok bool) {
	link, exists := g.replays[submissionKey{e.Journal, e.Domain, e.Correlation}]
	if !exists || link.Executable.Handle != e.Executable {
		return
	}
	shape := g.shapes[link.Source]
	if !shape.valid || len(shape.nodes) == 0 || len(shape.nodes) != len(indices) {
		return
	}
	nodes := map[uint64]int{}
	for _, i := range indices {
		a := activities[i]
		id, valid := g.original(e.Journal, a.GraphNodeID)
		if !valid || id == 0 {
			return
		}
		if _, exists := shape.nodes[id]; !exists {
			return
		}
		if _, exists := nodes[id]; exists {
			return
		}
		nodes[id] = i
	}
	outgoing := map[uint64]bool{}
	ids := make([]uint64, 0, len(shape.nodes))
	for id := range shape.nodes {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	for _, id := range ids {
		node := shape.nodes[id]
		index := nodes[id]
		if len(node.dependencies) == 0 {
			roots = append(roots, index)
		}
		for _, from := range node.dependencies {
			edges = append(edges, [2]int{nodes[from], index})
			receipts = append(receipts, node.receipt)
			outgoing[from] = true
		}
	}
	for _, id := range ids {
		if !outgoing[id] {
			leaves = append(leaves, nodes[id])
		}
	}
	ok = true
	return
}
