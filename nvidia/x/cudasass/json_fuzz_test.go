package cudasass

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatest"
)

// jsonSeeds are the nvdisasm --emit-json listings ParseJSON is fuzzed from.
var jsonSeeds = []string{"testdata/*.json"}

// jsonDecoder checks ParseJSON. The encoder is encoding/json driven by the
// package's own struct tags, writing the [metadata, functions] pair ParseJSON
// reads: the schema is the package's, and the test adds no printer of its own.
var jsonDecoder = cudatest.Decoder[*JSONDisassembly]{
	Decode: func(b []byte) (*JSONDisassembly, error) { return ParseJSON(bytes.NewReader(b)) },
	Encode: func(d *JSONDisassembly) ([]byte, error) {
		var buf bytes.Buffer
		enc := json.NewEncoder(&buf)
		enc.SetEscapeHTML(false)
		err := enc.Encode([]any{d.Metadata, d.Functions})
		return buf.Bytes(), err
	},
	Equal: jsonEqual,
}

// jsonEqual compares two parses. Metadata is retained verbatim, so its values
// are compared as compacted JSON: encoding/json re-emits a RawMessage
// compacted.
func jsonEqual(a, b *JSONDisassembly) bool {
	if !reflect.DeepEqual(a.Functions, b.Functions) || len(a.Metadata) != len(b.Metadata) {
		return false
	}
	for k, va := range a.Metadata {
		vb, ok := b.Metadata[k]
		if !ok {
			return false
		}
		var ca, cb bytes.Buffer
		if json.Compact(&ca, va) != nil || json.Compact(&cb, vb) != nil || !bytes.Equal(ca.Bytes(), cb.Bytes()) {
			return false
		}
	}
	return true
}

func FuzzSASSParseJSONRoundTrips(f *testing.F) {
	seeds := cudatest.ReadSeeds(f, jsonSeeds...)
	for _, p := range cudatest.SortedPaths(seeds) {
		f.Add(seeds[p])
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		jsonDecoder.Check(t, data)
	})
}

func TestSASSParseJSONMutantControls(t *testing.T) {
	seed, err := os.ReadFile("testdata/nvdisasm-13.0-sm121.json")
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name   string
		anchor string // text in the seed
		offset int    // byte of anchor to mutate
		b      byte
	}{
		{"metadata", `"Producer":"nvdisasm V13.0.85`, 22, '4'},
		{"function name", `"function-name":"addone"`, 17, 'b'},
		{"length", `"length":384`, 9, '4'},
		{"opcode", `"opcode":"FADD"`, 11, 'M'},
		{"operands", `"operands":"R5,R0,1"`, 18, '2'},
		{"attribute", `"control-flow":"True"`, 16, 't'},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			i := bytes.Index(seed, []byte(tt.anchor))
			if i < 0 {
				t.Fatalf("seed has no %q", tt.anchor)
			}
			jsonDecoder.CheckMutant(t, seed, i+tt.offset, tt.b)
		})
	}
}
