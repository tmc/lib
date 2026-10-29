package safetensors

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

// writeDir writes the named files into a new directory and returns it.
func writeDir(t *testing.T, files map[string][]byte) string {
	t.Helper()
	dir := t.TempDir()
	for name, data := range files {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestOpenSharded(t *testing.T) {
	a := file(`{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`, 8)
	b := file(`{"b":{"dtype":"U8","shape":[3],"data_offsets":[0,3]}}`, 3)
	tests := []struct {
		name  string
		files map[string][]byte
		want  map[string]string
	}{
		{
			"index",
			map[string][]byte{
				IndexFile:           []byte(`{"metadata":{"total_size":11},"weight_map":{"a":"m-1.safetensors","b":"m-2.safetensors"}}`),
				"m-1.safetensors":   a,
				"m-2.safetensors":   b,
				"model.safetensors": b,
			},
			map[string]string{"a": "m-1.safetensors", "b": "m-2.safetensors"},
		},
		{
			"single file",
			map[string][]byte{"model.safetensors": a},
			map[string]string{"a": "model.safetensors"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := OpenSharded(writeDir(t, tt.files))
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			if !reflect.DeepEqual(s.Shard, tt.want) {
				t.Errorf("Shard = %v, want %v", s.Shard, tt.want)
			}
			for name := range tt.want {
				info, ok := s.Info(name)
				if !ok || info.Name != name {
					t.Fatalf("Info(%q) = %+v, %v", name, info, ok)
				}
				x, err := s.Tensor(name)
				if err != nil {
					t.Fatal(err)
				}
				if int64(len(x.Data)) != info.Size || x.Dtype != info.Dtype || !slices.Equal(x.Shape, info.Shape) {
					t.Errorf("Tensor(%q) = %s%v with %d bytes, want %+v", name, x.Dtype, x.Shape, len(x.Data), info)
				}
				r, err := s.TensorReader(name)
				if err != nil {
					t.Fatal(err)
				}
				if data, err := io.ReadAll(r); err != nil || !bytes.Equal(data, x.Data) {
					t.Errorf("TensorReader(%q) read %v, %v; want %v", name, data, err, x.Data)
				}
			}
			if _, ok := s.Info("missing"); ok {
				t.Error("Info reported a tensor that is not in the checkpoint")
			}
			if _, err := s.Tensor("missing"); err == nil {
				t.Error("Tensor read a tensor that is not in the checkpoint")
			}
			if _, err := s.TensorReader("missing"); err == nil {
				t.Error("TensorReader read a tensor that is not in the checkpoint")
			}
		})
	}
}

func TestOpenShardedRejects(t *testing.T) {
	a := file(`{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, 4)
	tests := []struct {
		name  string
		files map[string][]byte
		want  string
	}{
		{"empty map", map[string][]byte{IndexFile: []byte(`{"weight_map":{}}`)}, "empty weight map"},
		{"bad index", map[string][]byte{IndexFile: []byte(`{`)}, "decode"},
		{"path traversal", map[string][]byte{IndexFile: []byte(`{"weight_map":{"a":"../model.safetensors"}}`)}, "not a file in"},
		{"parent", map[string][]byte{IndexFile: []byte(`{"weight_map":{"a":".."}}`)}, "not a file in"},
		{"missing tensor", map[string][]byte{IndexFile: []byte(`{"weight_map":{"b":"model.safetensors"}}`), "model.safetensors": a}, "not in shard"},
		{"missing shard", map[string][]byte{IndexFile: []byte(`{"weight_map":{"a":"m-1.safetensors"}}`)}, "m-1.safetensors"},
		{"bad shard", map[string][]byte{IndexFile: []byte(`{"weight_map":{"a":"model.safetensors"}}`), "model.safetensors": a[:10]}, "header length"},
		{"no checkpoint", nil, "model.safetensors"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := OpenSharded(writeDir(t, tt.files))
			if err == nil {
				s.Close()
				t.Fatal("OpenSharded succeeded")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("OpenSharded error = %v, want %q", err, tt.want)
			}
		})
	}
}
