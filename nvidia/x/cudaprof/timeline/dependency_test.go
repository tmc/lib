package timeline

import "fmt"

func ExampleDependencyEvent() {
	event := DependencyEvent{Journal: "process.journal", Domain: "driver", Kind: "event_wait", Site: "exit", Ordinal: 4, Stream: 1, Event: 2}
	fmt.Println(event.Kind, event.Stream, event.Event)
	// Output: event_wait 1 2
}
