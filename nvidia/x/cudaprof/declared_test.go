package cudaprof

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/google/pprof/profile"
)

func ExampleBuilder_SetDeclaredSampleTypes() {
	b := NewBuilder()
	err := b.SetDeclaredSampleTypes(SampleType{Name: "dram__bytes_read.sum", Unit: "bytes"})
	fmt.Println(err)
	// Output: <nil>
}

func ExampleBuilder_AddDeclaredSample() {
	b := NewBuilder()
	_ = b.SetDeclaredSampleTypes(SampleType{Name: "dram__bytes_read.sum", Unit: "bytes"})
	err := b.AddDeclaredSample([]int64{4096}, []Frame{{FunctionName: "decode"}}, map[string]string{"capture": "1"})
	fmt.Println(err)
	// Output: <nil>
}

func ExampleSampleType() {
	t := SampleType{Name: "dram__bytes_read.sum", Unit: "bytes"}
	fmt.Println(t.Name, t.Unit)
	// Output: dram__bytes_read.sum bytes
}

func TestDeclaredRoundTrip(t *testing.T) {
	b := NewBuilder()
	schema := []SampleType{{"dram__bytes_read.sum", "bytes"}, {"smsp__inst_executed.sum", "instructions"}}
	if err := b.SetDeclaredSampleTypes(schema...); err != nil {
		t.Fatal(err)
	}
	schema[0].Name = "mutated"
	if err := b.SetDefaultSampleType("smsp__inst_executed.sum"); err != nil {
		t.Fatal(err)
	}
	labels := map[string]string{"kernel": "actual", "benchmark": "BenchmarkDecode/full"}
	values := []int64{4096, 32}
	if err := b.AddDeclaredSample(values, []Frame{{FunctionName: "actual"}, {FunctionName: "BenchmarkDecode/full"}}, labels); err != nil {
		t.Fatal(err)
	}
	values[0] = 0
	labels["kernel"] = "mutated"
	var buf bytes.Buffer
	if err := b.WriteGzip(&buf); err != nil {
		t.Fatal(err)
	}
	p, err := profile.ParseData(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if err := p.CheckValid(); err != nil {
		t.Fatal(err)
	}
	if p.DefaultSampleType != "smsp__inst_executed.sum" || p.SampleType[0].Type != "dram__bytes_read.sum" {
		t.Fatalf("schema = %+v, default %q", p.SampleType, p.DefaultSampleType)
	}
	if len(p.Sample) != 1 || p.Sample[0].Value[0] != 4096 || p.Sample[0].Label["kernel"][0] != "actual" {
		t.Fatalf("sample = %+v", p.Sample)
	}
	if len(p.Sample[0].Location) != 2 {
		t.Fatal("lost stack")
	}
	if err := b.SetDeclaredSampleTypes(SampleType{"other", "bytes"}); err == nil {
		t.Fatal("schema changed after record")
	}
}

func TestDeclaredRejectsSchemaAndIncompleteValues(t *testing.T) {
	for _, tc := range []struct {
		name   string
		schema []SampleType
	}{
		{"empty", nil}, {"name", []SampleType{{"", "bytes"}}}, {"unit", []SampleType{{"x", ""}}},
		{"duplicate", []SampleType{{"x", "bytes"}, {"x", "bytes"}}}, {"builtin", []SampleType{{"dram_read", "bytes"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if NewBuilder().SetDeclaredSampleTypes(tc.schema...) == nil {
				t.Fatal("accepted invalid schema")
			}
		})
	}
	b := NewBuilder()
	if err := b.AddDeclaredSample([]int64{1}, []Frame{{FunctionName: "x"}}, nil); err == nil {
		t.Fatal("accepted undeclared")
	}
	if err := b.SetDeclaredSampleTypes(SampleType{"x", "bytes"}, SampleType{"y", "count"}); err != nil {
		t.Fatal(err)
	}
	for _, values := range [][]int64{nil, {1}, {1, 2, 3}, {-1, 2}} {
		if b.AddDeclaredSample(values, []Frame{{FunctionName: "x"}}, nil) == nil {
			t.Fatalf("accepted %v", values)
		}
	}
	if b.AddDeclaredSample([]int64{1, 2}, nil, nil) == nil {
		t.Fatal("accepted no frames")
	}
	if len(b.samples) != 0 {
		t.Fatal("invalid calls mutated samples")
	}
	if b.SetSampleTypes("samples") == nil {
		t.Fatal("accepted builtin schema")
	}
	if b.SetDefaultSampleType("samples") == nil {
		t.Fatal("accepted builtin default")
	}
	b.AddSample("x", "", 0, 0, 1, nil)
	if b.WriteGzip(&bytes.Buffer{}) == nil {
		t.Fatal("wrote mixed schema")
	}
}
