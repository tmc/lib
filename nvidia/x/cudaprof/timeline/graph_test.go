package timeline

import (
	"fmt"
	"testing"
)

func ExampleGraphLifecycle() {
	success := true
	events := []GraphEvent{
		{Journal: "j", Kind: "create", Graph: 1, Success: &success},
		{Journal: "j", Kind: "instantiate", Graph: 1, Executable: 2, Success: &success},
		{Journal: "j", Kind: "replay", Executable: 2, Correlation: 7, Success: &success},
	}
	replay := GraphLifecycle(events)[2]
	fmt.Println(replay.Status, replay.Executable.Version, replay.Source.Handle)
	// Output: matched 1 1
}

func TestGraphLifecycle(t *testing.T) {
	success := true
	base := []GraphEvent{{Journal: "j", Kind: "create", Graph: 1, Success: &success}, {Journal: "j", Kind: "instantiate", Graph: 1, Executable: 2, Success: &success}}
	for _, tt := range []struct {
		name                string
		tail                []GraphEvent
		want                string
		version, generation uint64
	}{
		{"replay", []GraphEvent{{Kind: "replay", Executable: 2, Correlation: 7}}, "matched", 1, 1},
		{"update", []GraphEvent{{Kind: "update", Graph: 1, Executable: 2}, {Kind: "replay", Executable: 2, Correlation: 7}}, "matched", 2, 1},
		{"graph destroyed after instantiate", []GraphEvent{{Kind: "destroy_graph", Graph: 1}, {Kind: "replay", Executable: 2, Correlation: 7}}, "matched", 1, 1},
		{"executable destroyed", []GraphEvent{{Kind: "destroy_exec", Executable: 2}, {Kind: "replay", Executable: 2, Correlation: 7}}, "missing", 0, 0},
		{"executable reused", []GraphEvent{{Kind: "destroy_exec", Executable: 2}, {Kind: "instantiate", Graph: 1, Executable: 2}, {Kind: "replay", Executable: 2, Correlation: 7}}, "matched", 1, 2},
		{"duplicate live handle", []GraphEvent{{Kind: "instantiate", Graph: 1, Executable: 2}, {Kind: "replay", Executable: 2, Correlation: 7}}, "ambiguous", 0, 0},
		{"unknown update source", []GraphEvent{{Kind: "update", Graph: 3, Executable: 2}, {Kind: "replay", Executable: 2, Correlation: 7}}, "ambiguous", 0, 0},
		{"child update", []GraphEvent{{Kind: "update", Graph: 1, Executable: 2, Child: true}, {Kind: "replay", Executable: 2, Correlation: 7}}, "ambiguous", 0, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			events := append([]GraphEvent(nil), base...)
			for _, e := range tt.tail {
				e.Journal = "j"
				e.Success = &success
				events = append(events, e)
			}
			links := GraphLifecycle(events)
			got := links[len(links)-1]
			if got.Status != tt.want || got.Executable.Version != tt.version || got.Executable.Generation != tt.generation {
				t.Fatalf("got %+v", got)
			}
		})
	}
}

func TestGraphLifecycleUnknownResult(t *testing.T) {
	success := true
	events := []GraphEvent{
		{Journal: "j", Kind: "create", Graph: 1, Success: &success},
		{Journal: "j", Kind: "instantiate", Graph: 1, Executable: 2, Success: &success},
		{Journal: "j", Kind: "update", Graph: 1, Executable: 2},
		{Journal: "j", Kind: "replay", Executable: 2, Correlation: 7, Success: &success},
	}
	links := GraphLifecycle(events)
	if links[2].Status != "unsupported" || links[3].Status != "ambiguous" {
		t.Fatalf("unknown update qualified: %+v", links)
	}
}

func ExampleGraphEvent() {
	success := true
	event := GraphEvent{Journal: "process.journal", Kind: "replay", Executable: 2, Correlation: 7, Success: &success}
	fmt.Println(event.Kind, event.Executable, event.Correlation)
	// Output: replay 2 7
}

func ExampleGraphIdentity() {
	identity := GraphIdentity{Journal: "process.journal", Handle: 2, Generation: 1, Version: 3}
	fmt.Println(identity.Generation, identity.Version)
	// Output: 1 3
}

func ExampleGraphLink() {
	link := GraphLink{Event: 0, Status: "missing"}
	fmt.Println(link.Event, link.Status)
	// Output: 0 missing
}

func TestGraphReplayCorrelationDomains(t *testing.T) {
	success := true
	events := []GraphEvent{{Journal: "j", Domain: "driver", Kind: "create", Graph: 1, Success: &success}, {Journal: "j", Domain: "driver", Kind: "instantiate", Graph: 1, Executable: 2, Success: &success}, {Journal: "j", Domain: "driver", Kind: "replay", Executable: 2, Correlation: 7, Success: &success}, {Journal: "j", Domain: "runtime", Kind: "replay", Executable: 2, Correlation: 7, Success: &success}}
	links := GraphLifecycle(events)
	if links[2].Status != "matched" || links[3].Status != "matched" || links[2].Executable != links[3].Executable {
		t.Fatalf("domains collided: %+v", links)
	}
	events = append(events, events[3])
	links = GraphLifecycle(events)
	if links[2].Status != "matched" || links[3].Status != "ambiguous" || links[4].Status != "ambiguous" {
		t.Fatalf("duplicate runtime correlation qualified: %+v", links)
	}
}

func TestGraphLifecycleChildSnapshots(t *testing.T) {
	success := true
	events := []GraphEvent{
		{Journal: "j", Kind: "create", Graph: 1, Success: &success},
		{Journal: "j", Kind: "update_graph", Graph: 1, Success: &success},
		{Journal: "j", Kind: "create", Graph: 2, Success: &success},
		{Journal: "j", Kind: "child", Graph: 2, SourceGraphHandle: 1, NodeHandle: 10, Success: &success},
		{Journal: "j", Kind: "instantiate", Graph: 2, Executable: 3, Success: &success},
		{Journal: "j", Kind: "update_graph", Graph: 1, Success: &success},
		{Journal: "j", Kind: "destroy_graph", Graph: 1, Success: &success},
		{Journal: "j", Kind: "create", Graph: 1, Success: &success},
		{Journal: "j", Domain: "driver", Kind: "replay", Executable: 3, Correlation: 7, Success: &success},
	}
	links := GraphLifecycle(events)
	replay := links[len(links)-1]
	if replay.Status != "matched" || len(replay.Children) != 1 || replay.Children[0].Generation != 1 || replay.Children[0].Version != 2 {
		t.Fatalf("replay=%+v", replay)
	}
	// A graph copied at child-node creation stays at that version even after
	// its source graph is updated, destroyed and its handle reused.
	events = append(events, GraphEvent{Journal: "j", Kind: "clone", Graph: 4, SourceGraphHandle: 2, Success: &success})
	clone := GraphLifecycle(events)[len(events)-1]
	if clone.Status != "matched" || clone.Source.Handle != 2 || len(clone.Children) != 1 || clone.Children[0].Version != 2 {
		t.Fatalf("clone=%+v", clone)
	}
}

func TestGraphLifecycleMissingChild(t *testing.T) {
	success := true
	events := []GraphEvent{{Journal: "j", Kind: "create", Graph: 1, Success: &success}, {Journal: "j", Kind: "child", Graph: 1, SourceGraphHandle: 9, NodeHandle: 2, Success: &success}, {Journal: "j", Kind: "instantiate", Graph: 1, Executable: 3, Success: &success}}
	links := GraphLifecycle(events)
	if links[1].Status != "missing" || links[2].Status != "ambiguous" {
		t.Fatalf("links=%+v", links)
	}
}
