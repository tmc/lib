package nsightsystems_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/nsightsystems"
)

// metadataFixture is a synthetic read-only database boundary. A real adapter
// belongs outside this package and must verify its stable database identity,
// bind process/cancellation/output limits, and project exact SQLite columns.
type metadataFixture struct{}

func (metadataFixture) ReadTable(ctx context.Context, name string, limit int) (nsightsystems.Table, error) {
	if name != "META_DATA_EXPORT" {
		return nsightsystems.Table{}, nsightsystems.ErrUnavailable
	}
	return nsightsystems.Table{DatabaseSHA256: strings.Repeat("b", 64), Columns: []nsightsystems.Column{{Name: "name", Type: "TEXT"}, {Name: "value", Type: "TEXT", Nullable: true}}, Rows: []json.RawMessage{json.RawMessage(`{"name":"EXPORT_SCHEMA_VERSION","value":"3.11.0"}`)}}, nil
}

func Example() {
	source := nsightsystems.Source{ToolVersion: "2024.2", ToolBuild: "2024.2-fixture", ReportPath: "synthetic.nsys-rep", ReportSHA256: strings.Repeat("a", 64), DatabaseSHA256: strings.Repeat("b", 64), Clock: "unestablished"}
	r, err := nsightsystems.Import(context.Background(), metadataFixture{}, source, nsightsystems.Limits{})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(r.SchemaVersion, r.Basis, len(r.Events))
	// Output: 3.11.0 external:nsys 0
}

func ExampleImport() {
	Example()
	// Output: 3.11.0 external:nsys 0
}
func ExampleReader() {
	var r nsightsystems.Reader = metadataFixture{}
	_, err := r.ReadTable(context.Background(), "OSRT_API", 1)
	fmt.Println(errors.Is(err, nsightsystems.ErrUnavailable))
	// Output: true
}
func ExampleErrUnavailable() {
	fmt.Println(nsightsystems.ErrUnavailable)
	// Output: nsys: table unavailable
}
func ExampleColumn() {
	c := nsightsystems.Column{Name: "end", Type: "INTEGER", Nullable: true}
	fmt.Println(c.Name, c.Nullable)
	// Output: end true
}
func ExampleTable() {
	t := nsightsystems.Table{Rows: []json.RawMessage{json.RawMessage(`{"start":1}`)}}
	fmt.Println(len(t.Rows))
	// Output: 1
}
func ExampleSource() {
	s := nsightsystems.Source{Clock: "unestablished", ShiftNS: 17}
	fmt.Println(s.Clock, s.ShiftNS)
	// Output: unestablished 17
}
func ExampleLimits() {
	l := nsightsystems.Limits{Bytes: 1 << 20, Rows: 100}
	fmt.Println(l.Rows)
	// Output: 100
}
func ExampleProvider() {
	p := nsightsystems.Provider{Table: "OSRT_API", Reason: "table absent from export"}
	fmt.Println(p.Available, p.Reason)
	// Output: false table absent from export
}
func ExampleEvent() {
	e := nsightsystems.Event{StartNS: 100}
	fmt.Println(e.StartNS, e.EndNS == nil)
	// Output: 100 true
}
func ExampleJoin() {
	j := nsightsystems.Join{Event: 0, API: 3}
	fmt.Println(j.Event, j.API)
	// Output: 0 3
}
func ExampleUnjoined() {
	u := nsightsystems.Unjoined{Event: 1, Reason: "ambiguous API correlation"}
	fmt.Println(u.Reason)
	// Output: ambiguous API correlation
}
func ExampleReport() {
	r := nsightsystems.Report{Basis: "external:nsys"}
	fmt.Println(r.Basis)
	// Output: external:nsys
}
