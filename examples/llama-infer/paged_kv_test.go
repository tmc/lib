package main

import (
	"reflect"
	"strings"
	"testing"
)

func TestDefaultPagedKVConfig(t *testing.T) {
	tests := []struct {
		name      string
		maxSeqLen int
		capacity  int
		want      pagedKVConfig
	}{
		{"exact", 32, 2, pagedKVConfig{BlockSize: 16, TotalBlocks: 4, MaxBlocksPerSeq: 2}},
		{"rounds up", 33, 1, pagedKVConfig{BlockSize: 16, TotalBlocks: 3, MaxBlocksPerSeq: 3}},
		{"zero capacity", 16, 0, pagedKVConfig{BlockSize: 16, TotalBlocks: 1, MaxBlocksPerSeq: 1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := defaultPagedKVConfig(tt.maxSeqLen, tt.capacity); got != tt.want {
				t.Errorf("defaultPagedKVConfig(%d, %d) = %+v, want %+v", tt.maxSeqLen, tt.capacity, got, tt.want)
			}
		})
	}
}

func TestPagedKVSwitches(t *testing.T) {
	t.Setenv("LLAMA_INFER_PAGED_KV", "")
	if pagedKVEnabled() {
		t.Fatal("paged KV enabled with the variable unset")
	}
	t.Setenv("LLAMA_INFER_PAGED_KV", "true")
	if pagedKVEnabled() {
		t.Fatal(`paged KV enabled by "true"; only "1" enables it`)
	}
	t.Setenv("LLAMA_INFER_PAGED_KV", "1")
	if !pagedKVEnabled() {
		t.Fatal(`paged KV not enabled by "1"`)
	}

	tests := []struct {
		value   string
		want    int
		wantErr bool
	}{
		{"", defaultPagedKVSequences, false},
		{"3", 3, false},
		{"0", 0, true},
		{"-1", 0, true},
		{"eight", 0, true},
	}
	for _, tt := range tests {
		t.Setenv("LLAMA_INFER_PAGED_KV_SEQS", tt.value)
		got, err := pagedKVSequences()
		if (err != nil) != tt.wantErr {
			t.Errorf("LLAMA_INFER_PAGED_KV_SEQS=%q: err = %v, wantErr %v", tt.value, err, tt.wantErr)
			continue
		}
		if got != tt.want {
			t.Errorf("LLAMA_INFER_PAGED_KV_SEQS=%q: got %d, want %d", tt.value, got, tt.want)
		}
	}
}

func TestBuildPagedKVTables(t *testing.T) {
	const blockSize, width = 4, 3
	tests := []struct {
		name      string
		tables    [][]int
		positions []int
		want      pagedKVTables
		wantErr   string
	}{
		{
			name:      "one row",
			tables:    [][]int{{7}},
			positions: []int{3},
			want: pagedKVTables{
				seqIndex:   []int32{0},
				positions:  []int32{3},
				seqLens:    []int32{4},
				blockTable: []int32{7, -1, -1},
			},
		},
		{
			// A prefill tile: three rows of one sequence, one block table
			// repeated, each row seeing exactly its own prefix.
			name:      "prefill tile shares a table",
			tables:    [][]int{{2, 5}, {2, 5}, {2, 5}},
			positions: []int{4, 5, 6},
			want: pagedKVTables{
				seqIndex:   []int32{0, 1, 2},
				positions:  []int32{4, 5, 6},
				seqLens:    []int32{5, 6, 7},
				blockTable: []int32{2, 5, -1, 2, 5, -1, 2, 5, -1},
			},
		},
		{
			name:      "full width",
			tables:    [][]int{{1, 2, 3}},
			positions: []int{11},
			want: pagedKVTables{
				seqIndex:   []int32{0},
				positions:  []int32{11},
				seqLens:    []int32{12},
				blockTable: []int32{1, 2, 3},
			},
		},
		{name: "no rows", wantErr: "no rows"},
		{name: "count mismatch", tables: [][]int{{1}}, positions: []int{0, 1}, wantErr: "1 block tables for 2 positions"},
		{name: "negative position", tables: [][]int{{1}}, positions: []int{-1}, wantErr: "negative position"},
		{name: "too many blocks", tables: [][]int{{1, 2, 3, 4}}, positions: []int{0}, wantErr: "table width is 3"},
		{name: "position past table", tables: [][]int{{1}}, positions: []int{4}, wantErr: "do not cover position 4"},
		{name: "empty table", tables: [][]int{{}}, positions: []int{0}, wantErr: "do not cover position 0"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := buildPagedKVTables(tt.tables, tt.positions, blockSize, width)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("err = %v, want one containing %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("got %+v\nwant %+v", got, tt.want)
			}
		})
	}
	if _, err := buildPagedKVTables([][]int{{1}}, []int{0}, 0, width); err == nil {
		t.Error("accepted block size 0")
	}
}
