package timeline

import "fmt"

func ExampleUnifiedMemory() {
	record := UnifiedMemory{CounterKind: 3, Value: 0xfeed, Start: 10, End: 10, EndValid: false}
	fmt.Printf("CPU fault instruction: %#x; interval end valid: %t\n", record.Value, record.EndValid)
	// Output: CPU fault instruction: 0xfeed; interval end valid: false
}
func ExampleMemoryPolicy() {
	event := MemoryPolicy{Event: "prefetch_request", ResultKnown: true, Result: 0}
	fmt.Println(event.Event, event.Result)
	// Output: prefetch_request 0
}
