package metalx

import (
	"reflect"
	"strings"
	"testing"

	apple "github.com/tmc/apple/metal"
)

type pso = apple.MTLComputePipelineState

func TestKernelFields(t *testing.T) {
	type tagged struct {
		Embed pso `metal:"EmbeddingLookup"`
		Norm  pso `metal:"ComputeRMSNorm"`
	}
	type untagged struct {
		FastGEMV pso
	}
	type mixed struct {
		Embed   pso `metal:"EmbeddingLookup"`
		Skipped pso `metal:"-"`
		Name    string
		Plain   pso
	}

	tests := []struct {
		name string
		val  any
		want []kernelField
	}{
		{
			name: "tags name the kernels",
			val:  tagged{},
			want: []kernelField{
				{index: 0, field: "Embed", kernel: "EmbeddingLookup"},
				{index: 1, field: "Norm", kernel: "ComputeRMSNorm"},
			},
		},
		{
			name: "field name is the default",
			val:  untagged{},
			want: []kernelField{{index: 0, field: "FastGEMV", kernel: "FastGEMV"}},
		},
		{
			name: "dash skips, other types are ignored",
			val:  mixed{},
			want: []kernelField{
				{index: 0, field: "Embed", kernel: "EmbeddingLookup"},
				{index: 3, field: "Plain", kernel: "Plain"},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := kernelFields(reflect.TypeOf(tt.val))
			if err != nil {
				t.Fatalf("kernelFields: %v", err)
			}
			if len(got) != len(tt.want) {
				t.Fatalf("got %d fields, want %d: %+v", len(got), len(tt.want), got)
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("field %d = %+v, want %+v", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestKernelFieldsErrors(t *testing.T) {
	type wrongType struct {
		Embed int `metal:"EmbeddingLookup"`
	}
	type unexported struct {
		embed pso `metal:"EmbeddingLookup"`
	}
	type none struct {
		Name string
	}

	tests := []struct {
		name string
		val  any
		want string
	}{
		{"tagged field of the wrong type", wrongType{}, "want apple.MTLComputePipelineState"},
		{"unexported field", unexported{}, "is unexported"},
		{"no pipeline fields", none{}, "has no apple.MTLComputePipelineState fields"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := kernelFields(reflect.TypeOf(tt.val))
			if err == nil {
				t.Fatal("kernelFields succeeded, want error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error %q does not mention %q", err, tt.want)
			}
		})
	}
}

func TestBindBadArguments(t *testing.T) {
	type kernels struct {
		Embed pso `metal:"EmbeddingLookup"`
	}
	var k kernels
	tests := []struct {
		name string
		val  any
		want string
	}{
		{"not a pointer", kernels{}, "want non-nil pointer to struct"},
		{"nil pointer", (*kernels)(nil), "want non-nil pointer to struct"},
		{"pointer to non-struct", new(int), "want pointer to struct"},
		{"struct pointer", &k, "nil device"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := Bind(nil, nil, tt.val)
			if err == nil {
				t.Fatal("Bind succeeded, want error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error %q does not mention %q", err, tt.want)
			}
		})
	}
}
