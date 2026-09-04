package cudakvlease

import (
	"errors"
	"reflect"
	"testing"
)

func TestBuildPackedLayout(t *testing.T) {
	tests := []struct {
		name        string
		requests    []PackedRequest
		rows        []int
		maxTokens   int
		wantRows    []int32
		wantPos     []int32
		wantLengths []int32
		wantGen     []uint64
		wantErr     error
	}{
		{
			name: "mixed lengths and empty row",
			requests: []PackedRequest{
				{BatchRequest: BatchRequest{Generation: 7, Length: 4}, Positions: []int{0, 1, 2, 3}},
				{BatchRequest: BatchRequest{Generation: 9, Length: 2}},
				{BatchRequest: BatchRequest{Generation: 11, Length: 3}, Positions: []int{0, 2}, CausalLengths: []int{2, 3}},
			},
			rows: []int{2, 0, 1}, maxTokens: 6,
			wantRows:    []int32{2, 2, 2, 2, 1, 1},
			wantPos:     []int32{0, 1, 2, 3, 0, 2},
			wantLengths: []int32{1, 2, 3, 4, 2, 3},
			wantGen:     []uint64{7, 7, 7, 7, 11, 11},
		},
		{name: "capacity", requests: []PackedRequest{{BatchRequest: BatchRequest{Generation: 1, Length: 2}, Positions: []int{0, 1}}}, rows: []int{0}, maxTokens: 1, wantErr: ErrInvalid},
		{name: "unordered", requests: []PackedRequest{{BatchRequest: BatchRequest{Generation: 1, Length: 2}, Positions: []int{1, 0}}}, rows: []int{0}, maxTokens: 2, wantErr: ErrInvalid},
		{name: "past final length", requests: []PackedRequest{{BatchRequest: BatchRequest{Generation: 1, Length: 2}, Positions: []int{2}}}, rows: []int{0}, maxTokens: 2, wantErr: ErrInvalid},
		{name: "all empty", requests: []PackedRequest{{BatchRequest: BatchRequest{Generation: 1, Length: 2}}}, rows: []int{0}, maxTokens: 2, wantErr: ErrInvalid},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			rows, positions, lengths, generations, err := buildPackedLayout(test.requests, test.rows, test.maxTokens)
			if !errors.Is(err, test.wantErr) {
				t.Fatalf("error = %v, want %v", err, test.wantErr)
			}
			if err == nil && (!reflect.DeepEqual(rows, test.wantRows) || !reflect.DeepEqual(positions, test.wantPos) || !reflect.DeepEqual(lengths, test.wantLengths) || !reflect.DeepEqual(generations, test.wantGen)) {
				t.Fatalf("layout = (%v, %v, %v, %v), want (%v, %v, %v, %v)", rows, positions, lengths, generations, test.wantRows, test.wantPos, test.wantLengths, test.wantGen)
			}
		})
	}
}
