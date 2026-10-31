package nvtelemetry

import (
	"bytes"
	"fmt"
	"strings"
)

func ExampleWriteOTLPLogs_suppliedEvent() {
	one := uint64(1)
	r := Reading{Name: "ecc_error_event", Unit: "code", DeviceUUID: "GPU-example", Evidence: EvidenceMeasured, Number: &Number{Uint: &one}, Observation: &Observation{Kind: "event", SourceClass: "synthetic"}}
	var b bytes.Buffer
	err := WriteOTLPLogs(&b, []Reading{r}, ExportOptions{})
	fmt.Println(err, strings.Contains(b.String(), "nvtelemetry.logs.events.v1"))
	// Output: <nil> true
}
