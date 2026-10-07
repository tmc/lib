package cudansysimport_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudansysimport"
)

// metadataFixture is a synthetic read-only database boundary. A real adapter
// belongs outside this package and must verify its stable database identity,
// bind process/cancellation/output limits, and project exact SQLite columns.
type metadataFixture struct{}

func (metadataFixture) ReadTable(ctx context.Context, name string, limit int) (cudansysimport.Table, error) {
	if name != "META_DATA_EXPORT" {
		return cudansysimport.Table{}, cudansysimport.ErrUnavailable
	}
	return cudansysimport.Table{DatabaseSHA256: strings.Repeat("b", 64), Columns: []cudansysimport.Column{{Name: "name", Type: "TEXT"}, {Name: "value", Type: "TEXT", Nullable: true}}, Rows: []json.RawMessage{json.RawMessage(`{"name":"EXPORT_SCHEMA_VERSION","value":"3.11.0"}`)}}, nil
}

func Example() {
	source := cudansysimport.Source{ToolVersion: "2024.2", ToolBuild: "2024.2-fixture", ReportPath: "synthetic.nsys-rep", ReportSHA256: strings.Repeat("a", 64), DatabaseSHA256: strings.Repeat("b", 64), Clock: "unestablished"}
	r, err := cudansysimport.Import(context.Background(), metadataFixture{}, source, cudansysimport.Limits{})
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
	var r cudansysimport.Reader = metadataFixture{}
	_, err := r.ReadTable(context.Background(), "OSRT_API", 1)
	fmt.Println(errors.Is(err, cudansysimport.ErrUnavailable))
	// Output: true
}
func ExampleErrUnavailable() {
	fmt.Println(cudansysimport.ErrUnavailable)
	// Output: nsys: table unavailable
}
func ExampleColumn() {
	c := cudansysimport.Column{Name: "end", Type: "INTEGER", Nullable: true}
	fmt.Println(c.Name, c.Nullable)
	// Output: end true
}
func ExampleTable() {
	t := cudansysimport.Table{Rows: []json.RawMessage{json.RawMessage(`{"start":1}`)}}
	fmt.Println(len(t.Rows))
	// Output: 1
}
func ExampleSource() {
	s := cudansysimport.Source{Clock: "unestablished", ShiftNS: 17}
	fmt.Println(s.Clock, s.ShiftNS)
	// Output: unestablished 17
}
func ExampleLimits() {
	l := cudansysimport.Limits{Bytes: 1 << 20, Rows: 100}
	fmt.Println(l.Rows)
	// Output: 100
}
func ExampleProvider() {
	p := cudansysimport.Provider{Table: "OSRT_API", Reason: "table absent from export"}
	fmt.Println(p.Available, p.Reason)
	// Output: false table absent from export
}
func ExampleEvent() {
	e := cudansysimport.Event{StartNS: 100}
	fmt.Println(e.StartNS, e.EndNS == nil)
	// Output: 100 true
}
func ExampleJoin() {
	j := cudansysimport.Join{Event: 0, API: 3}
	fmt.Println(j.Event, j.API)
	// Output: 0 3
}
func ExampleUnjoined() {
	u := cudansysimport.Unjoined{Event: 1, Reason: "ambiguous API correlation"}
	fmt.Println(u.Reason)
	// Output: ambiguous API correlation
}
func ExampleReport() {
	r := cudansysimport.Report{Basis: "external:nsys"}
	fmt.Println(r.Basis)
	// Output: external:nsys
}
