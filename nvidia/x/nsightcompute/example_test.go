package nsightcompute_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/nsightcompute"
)

// exampleEnvelope is a synthetic exporter control, not a vendor report.
func exampleEnvelope() nsightcompute.Envelope {
	csv := "ID,Process ID,Process Name,Host Name,Kernel Name,Context,Stream,Block Size,Grid Size,Device,CC,Section Name,Metric Name,Metric Unit,Metric Value,Rule Name,Rule Type,Rule Description,Estimated Speedup Type,Estimated Speedup\n" +
		"0,1,example,host,kernel,1,1,\"(1, 1, 1)\",\"(1, 1, 1)\",0,9.0,Timing,gpu__time_duration.sum,ns,42,\n"
	return nsightcompute.Envelope{SchemaVersion: 1, ToolVersion: nsightcompute.ToolVersion, Format: nsightcompute.Format, SourcePath: "synthetic example", ReportDigest: fmt.Sprintf("%x", sha256.Sum256([]byte("synthetic example"))), CSVDigest: fmt.Sprintf("%x", sha256.Sum256([]byte(csv))), CSV: csv}
}

func Example() {
	r, err := nsightcompute.Import(exampleEnvelope(), nil)
	fmt.Println(err, len(r.Rows), r.Rows[0].JoinReason)
	// Output: <nil> 1 external launch identity unavailable
}

func ExampleRead() {
	b, _ := json.Marshal(exampleEnvelope())
	e, err := nsightcompute.Read(bytes.NewReader(b), 1<<20)
	fmt.Println(err, e.ToolVersion, e.Format)
	// Output: <nil> 2026.3.0.0 ncu-details-csv-2026.3.0
}

func ExampleImport() {
	r, _ := nsightcompute.Import(exampleEnvelope(), nil)
	m := r.Rows[0].Metric
	fmt.Println(m.Name, m.RawValue, m.Unit, m.Value.Record().Number)
	// Output: gpu__time_duration.sum 42 ns <nil>
}

func ExampleReadRaw() {
	export := `==PROF== Connected to process 1382
"ID","Kernel Name","Block Size","Grid Size","gpu__time_duration.sum","dram__bytes_read.sum"
"","","","","ns","byte"
"0","copy_kernel","(256, 1, 1)","(132, 1, 1)","16,608","8,617,984"
`
	t, err := nsightcompute.ReadRaw(strings.NewReader(export), nsightcompute.MaxBytes)
	if err != nil {
		panic(err)
	}
	for i := range t.Launches {
		l := &t.Launches[i]
		ns, _, _ := t.Value(l, "gpu__time_duration.sum")
		bytes, _, _ := t.Value(l, "dram__bytes_read.sum")
		fmt.Printf("%s: %.0f ns, %.1f GB/s\n", l.Kernel, ns, bytes/ns)
	}
	// Output:
	// copy_kernel: 16608 ns, 518.9 GB/s
}
