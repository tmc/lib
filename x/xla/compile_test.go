package xla

import (
	"bytes"
	"testing"
)

func TestCompileOptions_ToProto(t *testing.T) {
	tests := []struct {
		name string
		opts *CompileOptions
		want []byte
	}{
		{
			name: "default",
			opts: DefaultCompileOptions(),
			want: []byte{
				26, 4, // Field 3 (ExecutableBuildOptionsProto), length 4
				32, 1, // Field 4 (num_replicas) = 1
				40, 1, // Field 5 (num_partitions) = 1
			},
		},
		{
			name: "custom",
			opts: &CompileOptions{
				ReplicaCount:  2,
				NumPartitions: 4,
			},
			want: []byte{
				26, 4, // Field 3 (ExecutableBuildOptionsProto), length 4
				32, 2, // Field 4 (num_replicas) = 2
				40, 4, // Field 5 (num_partitions) = 4
			},
		},
		{
			name: "empty",
			opts: &CompileOptions{
				ReplicaCount:  0,
				NumPartitions: 0,
			},
			want: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.opts.ToProto()
			if !bytes.Equal(got, tt.want) {
				t.Errorf("ToProto() = %v, want %v", got, tt.want)
			}
		})
	}
}
