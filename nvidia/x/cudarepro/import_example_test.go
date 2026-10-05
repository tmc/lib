package cudarepro

import (
	"bytes"
	"encoding/json"
	"fmt"
)

func ExampleReadFinding() {
	// This simulated record demonstrates import. Historical records missing
	// structured comparisons and actual confirmation runs are refused.
	original, source := findingRecord()
	data, err := json.Marshal(original)
	if err != nil {
		fmt.Println(err)
		return
	}
	finding, err := ReadFinding(bytes.NewReader(data), source)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(finding.Seed, finding.Confirmation.Findings, err)
	// Output: 17 2 <nil>
}
