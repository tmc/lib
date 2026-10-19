package cudatrial

import "fmt"

func ExampleReceipt_Clone() {
	original := Receipt{Provenance: map[string]string{"source": "observed"}}
	copy := original.Clone()
	original.Provenance["source"] = "changed"
	fmt.Println(copy.Provenance["source"])
	// Output: observed
}

func ExampleReceipt_CloneWithMemory() {
	var reserved int64
	_, err := (Receipt{Name: "arm"}).CloneWithMemory(func(n int64) error { reserved += n; return nil })
	fmt.Println(err, reserved > 0)
	// Output: <nil> true
}
