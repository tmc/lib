package cudaschedule

import (
	"strings"
	"sync"
	"testing"
)

func TestRecordedOutput(t *testing.T) {
	for _, limit := range []int{0, -1, (64 << 20) + 1} {
		if _, err := newRecordedOutput(limit); err == nil {
			t.Fatalf("accepted invalid limit %d", limit)
		}
	}
	b, err := newRecordedOutput(4)
	if err != nil {
		t.Fatal(err)
	}
	if n, err := b.Write([]byte("abcd")); n != 4 || err != nil {
		t.Fatalf("write=%d %v", n, err)
	}
	if data, overflow := b.snapshot(); data != "abcd" || overflow {
		t.Fatalf("exact limit: %q %v", data, overflow)
	}
	if n, err := b.Write([]byte("efgh")); n != 4 || err != nil {
		t.Fatalf("drain=%d %v", n, err)
	}
	if data, overflow := b.snapshot(); data != "abcd" || !overflow {
		t.Fatalf("overflow: %q %v", data, overflow)
	}
}

func TestRecordedOutputConcurrent(t *testing.T) {
	b, err := newRecordedOutput(1024)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for range 4 {
		wg.Go(func() {
			for range 100 {
				b.Write([]byte(strings.Repeat("x", 100)))
				b.snapshot()
			}
		})
	}
	wg.Wait()
	if data, overflow := b.snapshot(); len(data) != 1024 || !overflow {
		t.Fatalf("concurrent bound=%d %v", len(data), overflow)
	}
}
