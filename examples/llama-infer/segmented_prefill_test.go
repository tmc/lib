package main

import (
	"context"
	"reflect"
	"testing"
	"time"
)

func TestPrefillPackerContract(t *testing.T) {
	var packer PrefillPacker
	got, err := packer.Pack(t.Context(), []PrefillSegment{
		{RequestID: "a", Row: 2, Tokens: []int32{4, 5}, AbsoluteStart: 3, RestoredPrefix: 3},
		{RequestID: "b", Row: 0, Tokens: []int32{7}, AbsoluteStart: 127, RestoredPrefix: 127},
	}, 3, 4, 16, 128)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Offsets, []uint32{0, 2}) || !reflect.DeepEqual(got.Lengths, []uint32{2, 1}) || !reflect.DeepEqual(got.RequestIndex, []uint32{0, 0, 1}) || !reflect.DeepEqual(got.AbsolutePositions, []uint32{3, 4, 127}) {
		t.Fatalf("pack=%+v", got)
	}
	got.Tokens[0] = 15
	if packer.Packed().Tokens[0] != 4 {
		t.Fatal("caller mutated committed pack")
	}
}

func TestPrefillPackerRejectsMalformedAndRollsBack(t *testing.T) {
	var packer PrefillPacker
	want, err := packer.Pack(t.Context(), []PrefillSegment{{RequestID: "ok", Row: 0, Tokens: []int32{1}, AbsoluteStart: 0}}, 2, 2, 8, 8)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name     string
		segments []PrefillSegment
	}{
		{"overflow", []PrefillSegment{{RequestID: "x", Row: 0, Tokens: []int32{1, 2, 3}}}},
		{"capacity", []PrefillSegment{{RequestID: "x", Row: 0, Tokens: []int32{1}, AbsoluteStart: 8, RestoredPrefix: 8}}},
		{"prefix mismatch", []PrefillSegment{{RequestID: "x", Row: 0, Tokens: []int32{1}, AbsoluteStart: 2, RestoredPrefix: 1}}},
		{"duplicate row", []PrefillSegment{{RequestID: "x", Row: 0, Tokens: []int32{1}}, {RequestID: "y", Row: 0, Tokens: []int32{2}}}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := packer.Pack(t.Context(), test.segments, 2, 2, 8, 8); err == nil {
				t.Fatal("malformed pack succeeded")
			}
			if got := packer.Packed(); !reflect.DeepEqual(got, want) {
				t.Fatalf("rollback got=%+v want=%+v", got, want)
			}
		})
	}
	hole := want
	hole.Offsets[0] = 1
	if err := hole.Validate(2, 2, 8); err == nil {
		t.Fatal("KV hole succeeded")
	}
}

func TestPrefillPackerCanceledDoesNotCommit(t *testing.T) {
	var packer PrefillPacker
	want, err := packer.Pack(t.Context(), []PrefillSegment{{RequestID: "old", Row: 0, Tokens: []int32{1}}}, 2, 2, 8, 8)
	if err != nil {
		t.Fatal(err)
	}
	ctx := &lateCancelContext{}
	if _, err := packer.Pack(ctx, []PrefillSegment{{RequestID: "new", Row: 1, Tokens: []int32{2}}}, 2, 2, 8, 8); err == nil {
		t.Fatal("post-pack cancellation succeeded")
	}
	if got := packer.Packed(); !reflect.DeepEqual(got, want) {
		t.Fatalf("cancellation committed got=%+v want=%+v", got, want)
	}
}

type lateCancelContext struct{ checks int }

func (*lateCancelContext) Deadline() (time.Time, bool) { return time.Time{}, false }
func (*lateCancelContext) Done() <-chan struct{}       { return nil }
func (*lateCancelContext) Value(any) any               { return nil }
func (c *lateCancelContext) Err() error {
	c.checks++
	if c.checks > 1 {
		return context.Canceled
	}
	return nil
}
