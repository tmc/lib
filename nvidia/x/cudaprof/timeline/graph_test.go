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
