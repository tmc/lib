package cudaevidence_test

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
)

func ExampleValue_UnmarshalJSON() {
	var value cudaevidence.Value
	err := json.Unmarshal([]byte(`{"class":"unread","number":null,"reason":"provider unavailable"}`), &value)
	fmt.Println(value.Record().Reason, err)
	// Output: provider unavailable <nil>
}

func Example() {
	var value cudaevidence.Value
	fmt.Println(value.Record().Reason)
	// Output: not read
}
func ExampleClass() {
	fmt.Println(cudaevidence.Measured, cudaevidence.Derived, cudaevidence.Inferred, cudaevidence.Modeled)
	// Output: measured derived inferred modeled
}
func ExampleIdentity_Validate() {
	id := cudaevidence.Identity{ArtifactDigest: strings.Repeat("1", 64), Entry: "kernel", Target: "sm_70", CompilerKey: strings.Repeat("2", 64), WorkloadKey: strings.Repeat("3", 64)}
	fmt.Println(id.Validate())
	// Output: <nil>
}
func ExampleNew() {
	n := 0.0
	v, err := cudaevidence.New(cudaevidence.Record{Class: cudaevidence.Derived, Number: &n, Basis: "sum of retained records", Source: cudaevidence.Source{Identity: cudaevidence.Identity{ArtifactDigest: strings.Repeat("1", 64), Entry: "kernel", Target: "sm_70", CompilerKey: strings.Repeat("2", 64), WorkloadKey: strings.Repeat("3", 64)}, Producer: "analysis", Locator: "report/1"}})
	fmt.Println(v.Record().Class, *v.Record().Number, err)
	// Output: derived 0 <nil>
}
func ExampleValue_Record() {
	var value cudaevidence.Value
	record := value.Record()
	fmt.Println(record.Number == nil, record.Reason)
	// Output: true not read
}
func ExampleValue_MarshalJSON() {
	value, _ := cudaevidence.New(cudaevidence.Record{Reason: "counter unavailable"})
	data, err := value.MarshalJSON()
	var record map[string]any
	_ = json.Unmarshal(data, &record)
	fmt.Println(record["class"], record["number"], record["reason"], err)
	// Output: unread <nil> counter unavailable <nil>
}
func ExampleDecode() {
	value, err := cudaevidence.Decode([]byte(`{"class":"unread","number":null,"reason":"not sampled"}`))
	fmt.Println(value.Record().Reason, err)
	// Output: not sampled <nil>
}
func ExampleValue_CheckClaim() {
	n := 12.0
	id := cudaevidence.Identity{ArtifactDigest: strings.Repeat("1", 64), Entry: "kernel", Target: "sm_70", CompilerKey: strings.Repeat("2", 64), WorkloadKey: strings.Repeat("3", 64)}
	value, _ := cudaevidence.New(cudaevidence.Record{Class: cudaevidence.Inferred, Number: &n, Basis: "static interpretation", Source: cudaevidence.Source{Identity: id, Producer: "analysis", Locator: "report/1"}})
	fmt.Println(value.CheckClaim(cudaevidence.Inferred, id) == nil)
	fmt.Println(value.CheckClaim(cudaevidence.Measured, id) == nil)
	// Output:
	// true
	// false
}
func ExampleLoss() {
	loss := cudaevidence.Loss{Dropped: 2, Total: 10}
	fmt.Println(loss.Dropped, loss.Total)
	// Output: 2 10
}
func ExampleSource() {
	source := cudaevidence.Source{Identity: cudaevidence.Identity{ArtifactDigest: strings.Repeat("1", 64), Entry: "kernel", Target: "sm_70", CompilerKey: strings.Repeat("2", 64), WorkloadKey: strings.Repeat("3", 64)}, Producer: "capture", Locator: "launch/7"}
	fmt.Println(source.Producer, source.Locator)
	// Output: capture launch/7
}
func ExampleRecord() {
	record := cudaevidence.Record{Reason: "producer unavailable"}
	value, err := cudaevidence.New(record)
	fmt.Println(value.Record().Reason, err)
	// Output: producer unavailable <nil>
}
func ExampleValue() {
	var value cudaevidence.Value
	fmt.Println(value.Record().Class == cudaevidence.Unread)
	// Output: true
}
func ExampleIdentity() {
	id := cudaevidence.Identity{ArtifactDigest: strings.Repeat("1", 64), Entry: "kernel", Target: "sm_70", CompilerKey: strings.Repeat("2", 64), WorkloadKey: strings.Repeat("3", 64)}
	fmt.Println(id.Entry, id.Target)
	// Output: kernel sm_70
}
