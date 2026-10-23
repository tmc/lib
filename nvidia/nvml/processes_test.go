package nvml

import (
	"errors"
	"math"
	"strings"
	"testing"
)

func TestRunningProcessesInto(t *testing.T) {
	big := &CallError{Func: "nvmlDeviceGetComputeRunningProcesses_v3", Result: ErrInsufficientSize}
	for _, tt := range []struct {
		name    string
		size    int
		counts  []uint32 // count reported by successive calls
		err     error
		want    int
		wantErr error  // matched with errors.Is
		wantMsg string // matched as a substring
	}{
		{name: "fewer", size: 4, counts: []uint32{2, 9}, want: 2},
		{name: "exact", size: 4, counts: []uint32{4}, want: 4},
		{name: "none", size: 1, counts: []uint32{0}, want: 0},
		{name: "insufficient size", size: 2, counts: []uint32{3, 2}, err: big, wantErr: ErrInsufficientSize, wantMsg: "3 entries do not fit buffer 2"},
		{name: "insufficient size unchanged count", size: 2, counts: []uint32{2}, err: ErrInsufficientSize, wantErr: ErrInsufficientSize},
		{name: "count beyond buffer", size: 2, counts: []uint32{3, 2}, wantMsg: "count 3 exceeds buffer 2"},
		{name: "count max uint32", size: 1, counts: []uint32{math.MaxUint32}, wantMsg: "count 4294967295 exceeds buffer 1"},
		{name: "permission", size: 2, counts: []uint32{0}, err: &CallError{Result: ErrNoPermission}, wantErr: ErrNoPermission},
		{name: "unsupported", size: 2, counts: []uint32{0}, err: ErrNotSupported, wantErr: ErrNotSupported},
		{name: "uninitialized", size: 2, counts: []uint32{0}, err: ErrUninitialized, wantErr: ErrUninitialized},
	} {
		t.Run(tt.name, func(t *testing.T) {
			buf := make([]ProcessInfo, tt.size)
			calls := 0
			n, err := runningProcessesInto("compute", buf, func(count *uint32, p *ProcessInfo) error {
				if *count != uint32(len(buf)) || p != &buf[0] {
					t.Errorf("call %d: count %d pointer %p, want %d %p", calls, *count, p, len(buf), &buf[0])
				}
				// A driver whose count changes between calls must not be
				// asked again.
				*count = tt.counts[min(calls, len(tt.counts)-1)]
				calls++
				return tt.err
			})
			if calls != 1 {
				t.Fatalf("%d calls, want exactly 1", calls)
			}
			if tt.wantErr == nil && tt.wantMsg == "" {
				if err != nil || n != tt.want {
					t.Fatalf("got %d, %v, want %d, nil", n, err, tt.want)
				}
				return
			}
			if err == nil || n != 0 {
				t.Fatalf("got %d, %v, want 0 and an error", n, err)
			}
			if tt.wantErr != nil && !errors.Is(err, tt.wantErr) {
				t.Fatalf("err = %v, want errors.Is %v", err, tt.wantErr)
			}
			if !strings.HasPrefix(err.Error(), "query compute processes: ") || !strings.Contains(err.Error(), tt.wantMsg) {
				t.Fatalf("err = %q, want context and %q", err, tt.wantMsg)
			}
		})
	}
}

func TestRunningProcessesIntoRefusesBeforeCall(t *testing.T) {
	for _, buf := range [][]ProcessInfo{nil, {}, make([]ProcessInfo, 0, 8)} {
		n, err := runningProcessesInto("graphics", buf, func(*uint32, *ProcessInfo) error {
			t.Fatal("called with an empty buffer")
			return nil
		})
		if err == nil || n != 0 || !strings.Contains(err.Error(), "query graphics processes: empty buffer") {
			t.Fatalf("got %d, %v, want empty buffer refusal", n, err)
		}
	}
	// The exported methods refuse an empty buffer without loading the library.
	d := Device(0)
	for name, query := range map[string]func([]ProcessInfo) (int, error){
		"compute":     d.ComputeRunningProcessesInto,
		"graphics":    d.GraphicsRunningProcessesInto,
		"mps compute": d.MPSComputeRunningProcessesInto,
	} {
		n, err := query(nil)
		if err == nil || n != 0 || err.Error() != "query "+name+" processes: empty buffer" {
			t.Fatalf("%s: got %d, %v, want empty buffer refusal", name, n, err)
		}
	}
}

func TestProcessCapacity(t *testing.T) {
	for _, tt := range []struct {
		n    int
		want uint32
		ok   bool
	}{
		{-1, 0, false},
		{0, 0, false},
		{1, 1, true},
		{65536, 65536, true},
	} {
		got, err := processCapacity(tt.n)
		if got != tt.want || (err == nil) != tt.ok {
			t.Fatalf("processCapacity(%d) = %d, %v, want %d ok=%v", tt.n, got, err, tt.want, tt.ok)
		}
	}
	if math.MaxInt <= math.MaxUint32 {
		t.Skip("int cannot exceed uint32")
	}
	limit := uint64(math.MaxUint32)
	if got, err := processCapacity(int(limit)); err != nil || got != math.MaxUint32 {
		t.Fatalf("processCapacity(MaxUint32) = %d, %v", got, err)
	}
	if got, err := processCapacity(int(limit + 1)); err == nil || got != 0 {
		t.Fatalf("processCapacity(MaxUint32+1) = %d, %v, want refusal", got, err)
	}
}
