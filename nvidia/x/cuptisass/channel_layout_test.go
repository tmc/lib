package cuptisass_test

import (
	"context"
	"encoding/binary"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuptisass"
)

func TestChannelReceiveLayoutBounds(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	tests := []struct {
		name   string
		layout cuptisass.RecordLayout
	}{
		{"empty record", cuptisass.RecordLayout{Size: 0}},
		{"short record", cuptisass.RecordLayout{Size: 3}},
		{"oversized record", cuptisass.RecordLayout{Size: 17}},
		{"negative offset", cuptisass.RecordLayout{Size: 16, CTAXOffset: -1}},
		{"field beyond record", cuptisass.RecordLayout{Size: 16, CTAXOffset: 13}},
		{"maximum offset", cuptisass.RecordLayout{Size: 16, CTAXOffset: maxInt}},
		{"overflowing field end", cuptisass.RecordLayout{Size: 16, CTAXOffset: maxInt - 2}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			doorbell := int32(16)
			buf := make([]byte, 16)
			c, err := cuptisass.NewChannel(&doorbell, buf)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if p := recover(); p != nil {
					t.Errorf("invalid layout panicked: %v", p)
				}
			}()
			calls := 0
			st, err := c.Receive(context.Background(), tt.layout, func([]byte) error {
				calls++
				return nil
			})
			if err == nil || st != (cuptisass.ChannelStats{}) || calls != 0 || doorbell != 16 {
				t.Fatalf("invalid layout consumed input: stats=%+v err=%v calls=%d doorbell=%d", st, err, calls, doorbell)
			}
		})
	}
	for _, size := range []int{4, 16} {
		buf := make([]byte, size)
		binary.LittleEndian.PutUint32(buf[size-4:], ^uint32(0))
		doorbell := int32(size)
		c, err := cuptisass.NewChannel(&doorbell, buf)
		if err != nil {
			t.Fatal(err)
		}
		st, err := c.Receive(context.Background(), cuptisass.RecordLayout{Size: size, CTAXOffset: size - 4}, func([]byte) error {
			t.Fatal("sentinel delivered as a record")
			return nil
		})
		if err != nil || !st.Terminated || st.Bytes != int64(size) || st.Batches != 1 || st.Records != 0 || doorbell != 0 {
			t.Fatalf("valid field at end of %d-byte record: stats=%+v err=%v doorbell=%d", size, st, err, doorbell)
		}
	}
}
