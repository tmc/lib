package safetensors

import (
	"bytes"
	"encoding/hex"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatest"
)

// decoded is a file with every tensor read: the value NewFile and Tensor
// together decode, and the arguments Write takes.
type decoded struct {
	Metadata map[string]string
	Tensors  []Tensor
}

// fileDecoder checks NewFile and Tensor against the package's encoder, Write.
var fileDecoder = cudatest.Decoder[*decoded]{
	Decode: func(b []byte) (*decoded, error) {
		f, err := NewFile(bytes.NewReader(b), int64(len(b)))
		if err != nil {
			return nil, err
		}
		d := &decoded{Metadata: f.Metadata}
		for _, info := range f.Tensors {
			t, err := f.Tensor(info.Name)
			if err != nil {
				return nil, err
			}
			d.Tensors = append(d.Tensors, t)
		}
		return d, nil
	},
	Encode: func(d *decoded) ([]byte, error) {
		var buf bytes.Buffer
		err := Write(&buf, d.Tensors, d.Metadata)
		return buf.Bytes(), err
	},
	// Write documents that an empty metadata map writes no entry, so an
	// empty map and nil are the same metadata.
	Equal: func(a, b *decoded) bool {
		return maps.Equal(a.Metadata, b.Metadata) && reflect.DeepEqual(a.Tensors, b.Tensors)
	},
}

func readTorchFixture(t testing.TB) []byte {
	t.Helper()
	text, err := os.ReadFile(torchFixture)
	if err != nil {
		t.Fatal(err)
	}
	data, err := hex.DecodeString(strings.Join(strings.Fields(string(text)), ""))
	if err != nil {
		t.Fatalf("decode %s: %v", torchFixture, err)
	}
	return data
}

func FuzzHeaderRoundTrips(f *testing.F) {
	f.Add(readTorchFixture(f))
	var empty bytes.Buffer
	if err := Write(&empty, nil, nil); err != nil {
		f.Fatal(err)
	}
	f.Add(empty.Bytes())
	f.Fuzz(func(t *testing.T, data []byte) {
		fileDecoder.Check(t, data)
	})
}

func TestHeaderMutantControls(t *testing.T) {
	seed := readTorchFixture(t)
	tests := []struct {
		name   string
		anchor string // text in the seed
		offset int    // byte of anchor to mutate
		b      byte
	}{
		{"metadata value", `"format":"pt"`, 11, 'u'},
		{"metadata key", `"format":"pt"`, 1, 'g'},
		{"tensor name", `"n":{"dtype"`, 1, 'm'},
		{"dtype", `"dtype":"F16"`, 11, '8'},
		{"shape", `"shape":[2,2]`, 11, '1'},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			i := bytes.Index(seed, []byte(tt.anchor))
			if i < 0 {
				t.Fatalf("seed has no %q", tt.anchor)
			}
			fileDecoder.CheckMutant(t, seed, i+tt.offset, tt.b)
		})
	}
	// A byte of tensor data, which no header anchor reaches: the last byte of
	// the file is the last byte of the last tensor.
	t.Run("data", func(t *testing.T) {
		fileDecoder.CheckMutant(t, seed, len(seed)-1, seed[len(seed)-1]^1)
	})
}

func TestSeedCorpusCoversEveryRecordedFixture(t *testing.T) {
	cudatest.CheckSeedCoverage(t, "testdata", []string{filepath.FromSlash(torchFixture)}, nil)
}
