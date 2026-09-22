package cudatrial

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// meminfoDGX is a verbatim copy of /proc/meminfo from a Linux aarch64 host,
// captured 2026-09-22. Its MemFree and MemAvailable differ by roughly 22 GiB
// of page cache, which is the whole reason MemAvailable is the recorded term.
const meminfoDGXPath = "testdata/meminfo-dgx.txt"

const kB = 1024

func TestParseMeminfo(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    map[ConditionKind]int64
		unread  []ConditionKind
		wantErr bool
	}{
		{
			name: "all fields",
			input: "MemTotal:       127600528 kB\n" +
				"MemFree:        94850384 kB\n" +
				"MemAvailable:   118546824 kB\n" +
				"Buffers:         2372364 kB\n" +
				"Cached:         19725932 kB\n",
			want: map[ConditionKind]int64{
				ConditionHostTotalBytes:     127600528 * kB,
				ConditionHostAvailableBytes: 118546824 * kB,
				ConditionHostPageCacheBytes: (19725932 + 2372364) * kB,
			},
		},
		{
			name:   "no MemAvailable leaves it unread, not zero",
			input:  "MemTotal:       127600528 kB\nMemFree:        94850384 kB\n",
			want:   map[ConditionKind]int64{ConditionHostTotalBytes: 127600528 * kB},
			unread: []ConditionKind{ConditionHostAvailableBytes, ConditionHostPageCacheBytes},
		},
		{
			name:  "measured zero is recorded as zero",
			input: "MemAvailable:          0 kB\nCached:                0 kB\nBuffers:               0 kB\n",
			want: map[ConditionKind]int64{
				ConditionHostAvailableBytes: 0,
				ConditionHostPageCacheBytes: 0,
			},
			unread: []ConditionKind{ConditionHostTotalBytes},
		},
		{
			name:   "Cached without Buffers leaves page cache unread",
			input:  "Cached:         19725932 kB\n",
			unread: []ConditionKind{ConditionHostPageCacheBytes},
		},
		{
			name:   "unparseable value is unread",
			input:  "MemAvailable:   not-a-number kB\nMemTotal:       127600528 kB\n",
			want:   map[ConditionKind]int64{ConditionHostTotalBytes: 127600528 * kB},
			unread: []ConditionKind{ConditionHostAvailableBytes},
		},
		{
			name:   "unknown unit is unread",
			input:  "MemTotal:       124 GB\n",
			unread: []ConditionKind{ConditionHostTotalBytes},
		},
		{
			name:  "unitless value is bytes",
			input: "MemTotal:       4096\n",
			want:  map[ConditionKind]int64{ConditionHostTotalBytes: 4096},
		},
		{
			name:   "empty input reads nothing",
			input:  "",
			unread: []ConditionKind{ConditionHostTotalBytes, ConditionHostAvailableBytes, ConditionHostPageCacheBytes},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var c Conditions
			if err := parseMeminfo(strings.NewReader(tt.input), &c); (err != nil) != tt.wantErr {
				t.Fatalf("parseMeminfo error = %v, wantErr %v", err, tt.wantErr)
			}
			for kind, want := range tt.want {
				if !c.Has(kind) {
					t.Errorf("%s unread, want %d", kind, want)
					continue
				}
				if got := c.Bytes(kind); got != want {
					t.Errorf("%s = %d, want %d", kind, got, want)
				}
			}
			for _, kind := range tt.unread {
				if c.Has(kind) {
					t.Errorf("%s = %d, want unread", kind, c.Bytes(kind))
				}
				if got := c.Bytes(kind); got != Unmeasured {
					t.Errorf("%s = %d, want Unmeasured (%d)", kind, got, Unmeasured)
				}
			}
			if _, ok := c.Values["MemFree"]; ok {
				t.Error("MemFree recorded; only MemAvailable may be recorded")
			}
		})
	}
}

// TestObserveMeminfoFile exercises the present path of the Linux host reader on
// any GOOS by pointing it at a captured /proc/meminfo rather than at /proc.
func TestObserveMeminfoFile(t *testing.T) {
	var c Conditions
	if err := observeMeminfoFile(meminfoDGXPath, &c); err != nil {
		t.Fatalf("observeMeminfoFile: %v", err)
	}
	want := map[ConditionKind]int64{
		ConditionHostTotalBytes:     127600528 * kB,
		ConditionHostAvailableBytes: 118545140 * kB,
		ConditionHostPageCacheBytes: (19725944 + 2372368) * kB,
	}
	for kind, w := range want {
		if got := c.Bytes(kind); got != w {
			t.Errorf("%s = %d, want %d", kind, got, w)
		}
	}
	// The fixture's MemFree is 94848604 kB. Recording it instead of
	// MemAvailable would understate available memory by ~22 GiB.
	if got := c.Bytes(ConditionHostAvailableBytes); got == 94848604*kB {
		t.Error("host_available_bytes equals the fixture's MemFree; MemAvailable is required")
	}

	if err := observeMeminfoFile(filepath.Join(t.TempDir(), "absent"), &c); err == nil {
		t.Error("observeMeminfoFile on a missing file = nil error, want error")
	}
}

// TestObserveHostGOOS pins what the build-tagged dispatch does on this host.
func TestObserveHostGOOS(t *testing.T) {
	c := ObserveHost(context.Background())
	if c.Observed.IsZero() {
		t.Error("ObserveHost left Observed zero")
	}
	if c.Device != "" {
		t.Errorf("ObserveHost recorded device %q; it must never read a device", c.Device)
	}
	if _, err := os.Stat("/proc/meminfo"); err != nil {
		if c.Any() {
			t.Errorf("no /proc/meminfo but ObserveHost recorded %v", c.Values)
		}
		t.Logf("host has no /proc/meminfo: %d host kinds read, all Unmeasured", len(c.Values))
		return
	}
	for _, kind := range []ConditionKind{ConditionHostTotalBytes, ConditionHostAvailableBytes} {
		if !c.Has(kind) {
			t.Errorf("%s unread on a host with /proc/meminfo", kind)
		}
	}
}

func TestConditionsRecord(t *testing.T) {
	tests := []struct {
		name    string
		kind    ConditionKind
		bytes   int64
		wantErr bool
	}{
		{name: "measured zero", kind: ConditionRetainedBytes, bytes: 0},
		{name: "positive", kind: ConditionRetainedBytes, bytes: 77918175232},
		{name: "empty kind", kind: "", bytes: 1, wantErr: true},
		{name: "negative", kind: ConditionRetainedBytes, bytes: -1, wantErr: true},
		{name: "Unmeasured is not storable", kind: ConditionRetainedBytes, bytes: Unmeasured, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var c Conditions
			err := c.Record(tt.kind, tt.bytes)
			if (err != nil) != tt.wantErr {
				t.Fatalf("Record(%q, %d) error = %v, wantErr %v", tt.kind, tt.bytes, err, tt.wantErr)
			}
			if tt.wantErr {
				if c.Any() {
					t.Errorf("rejected Record stored %v", c.Values)
				}
				return
			}
			if !c.Has(tt.kind) || c.Bytes(tt.kind) != tt.bytes {
				t.Errorf("Bytes(%q) = %d, want %d", tt.kind, c.Bytes(tt.kind), tt.bytes)
			}
		})
	}
}

// TestConditionsZeroIsNotMeasured is the explicit measured-0 vs Unmeasured test.
func TestConditionsZeroIsNotMeasured(t *testing.T) {
	var unread Conditions
	if unread.Any() {
		t.Error("zero Conditions reports observations")
	}
	if got := unread.Bytes(ConditionDeviceFreeBytes); got != Unmeasured {
		t.Errorf("unread device_free_bytes = %d, want Unmeasured (%d)", got, Unmeasured)
	}
	if unread.Has(ConditionDeviceFreeBytes) {
		t.Error("unread device_free_bytes reports Has")
	}

	var measured Conditions
	if err := measured.Record(ConditionDeviceFreeBytes, 0); err != nil {
		t.Fatalf("Record: %v", err)
	}
	if !measured.Has(ConditionDeviceFreeBytes) {
		t.Error("measured zero reports Has = false")
	}
	if got := measured.Bytes(ConditionDeviceFreeBytes); got != 0 {
		t.Errorf("measured zero = %d, want 0", got)
	}
	if measured.Bytes(ConditionDeviceFreeBytes) == unread.Bytes(ConditionDeviceFreeBytes) {
		t.Error("a measured zero is indistinguishable from an unread value")
	}
}

func TestRecordDevicePool(t *testing.T) {
	var c Conditions
	if err := c.RecordDevicePool("", 1, 2); err == nil {
		t.Error("RecordDevicePool with no device name = nil error, want error")
	}
	if err := c.RecordDevicePool("cuda:0", 1<<34, 1<<37); err != nil {
		t.Fatalf("RecordDevicePool: %v", err)
	}
	if c.Device != "cuda:0" {
		t.Errorf("Device = %q, want %q", c.Device, "cuda:0")
	}
	if got, want := c.Bytes(ConditionDeviceFreeBytes), int64(1<<34); got != want {
		t.Errorf("device_free_bytes = %d, want %d", got, want)
	}
	if got, want := c.Bytes(ConditionDeviceTotalBytes), int64(1<<37); got != want {
		t.Errorf("device_total_bytes = %d, want %d", got, want)
	}
	if c.Has(ConditionHostAvailableBytes) {
		t.Error("a device readout recorded a host kind")
	}
}

// cond builds a Conditions from kind/value pairs; a nil map means nothing read.
func cond(vals map[ConditionKind]int64) Conditions {
	var c Conditions
	for k, v := range vals {
		if err := c.Record(k, v); err != nil {
			panic(err)
		}
	}
	return c
}

func rcpt(name string, round int, digest string, before Conditions) Receipt {
	return Receipt{Name: name, Round: round, Complete: true, Digest: digest, Before: before}
}

const giB = 1 << 30

func TestSameArmDisagreements(t *testing.T) {
	quiet := map[ConditionKind]int64{
		ConditionHostAvailableBytes: 100 * giB,
		ConditionHostPageCacheBytes: 4 * giB,
		ConditionRetainedBytes:      0,
	}
	// The motivating failure: a neighbouring arm retained ~72.56 GiB, so the
	// second run of this arm started under materially less host memory.
	pressed := map[ConditionKind]int64{
		ConditionHostAvailableBytes: 27 * giB,
		ConditionHostPageCacheBytes: 4 * giB,
		ConditionRetainedBytes:      77918175232,
	}

	tests := []struct {
		name           string
		receipts       []Receipt
		tolerance      int64
		wantCount      int
		wantConfounded bool
		wantChanged    []ConditionKind
		wantUnobserved []ConditionKind
		wantCompared   int
	}{
		{
			name: "disagreement under changed pressure is confounded",
			receipts: []Receipt{
				rcpt("moe", 1, "sha:aaa", cond(quiet)),
				rcpt("moe", 3, "sha:bbb", cond(pressed)),
			},
			wantCount:      1,
			wantConfounded: true,
			wantChanged:    []ConditionKind{ConditionHostAvailableBytes, ConditionRetainedBytes},
			wantCompared:   3,
		},
		{
			name: "disagreement under identical conditions is not confounded",
			receipts: []Receipt{
				rcpt("moe", 1, "sha:aaa", cond(quiet)),
				rcpt("moe", 3, "sha:bbb", cond(quiet)),
			},
			wantCount:      1,
			wantConfounded: false,
			wantCompared:   3,
		},
		{
			name: "agreement is not reported at all",
			receipts: []Receipt{
				rcpt("moe", 1, "sha:aaa", cond(quiet)),
				rcpt("moe", 3, "sha:aaa", cond(pressed)),
			},
			wantCount: 0,
		},
		{
			name: "no conditions recorded stays confounded",
			receipts: []Receipt{
				rcpt("moe", 1, "sha:aaa", Conditions{}),
				rcpt("moe", 3, "sha:bbb", Conditions{}),
			},
			wantCount:      1,
			wantConfounded: true,
			wantCompared:   0,
		},
		{
			name: "kind read in only one round is unobserved, not unchanged",
			receipts: []Receipt{
				rcpt("moe", 1, "sha:aaa", cond(map[ConditionKind]int64{ConditionHostAvailableBytes: 100 * giB})),
				rcpt("moe", 3, "sha:bbb", cond(map[ConditionKind]int64{
					ConditionHostAvailableBytes: 100 * giB,
					ConditionDeviceFreeBytes:    50 * giB,
				})),
			},
			wantCount:      1,
			wantConfounded: true,
			wantUnobserved: []ConditionKind{ConditionDeviceFreeBytes},
			wantCompared:   1,
		},
		{
			name: "drift within tolerance does not confound",
			receipts: []Receipt{
				rcpt("moe", 1, "sha:aaa", cond(map[ConditionKind]int64{ConditionHostAvailableBytes: 100 * giB})),
				rcpt("moe", 3, "sha:bbb", cond(map[ConditionKind]int64{ConditionHostAvailableBytes: 100*giB - 1*giB})),
			},
			tolerance:      2 * giB,
			wantCount:      1,
			wantConfounded: false,
			wantCompared:   1,
		},
		{
			name: "drift just past tolerance confounds",
			receipts: []Receipt{
				rcpt("moe", 1, "sha:aaa", cond(map[ConditionKind]int64{ConditionHostAvailableBytes: 100 * giB})),
				rcpt("moe", 3, "sha:bbb", cond(map[ConditionKind]int64{ConditionHostAvailableBytes: 100*giB + 2*giB + 1})),
			},
			tolerance:      2 * giB,
			wantCount:      1,
			wantConfounded: true,
			wantChanged:    []ConditionKind{ConditionHostAvailableBytes},
			wantCompared:   1,
		},
		{
			name: "different arms differing is not a same-arm disagreement",
			receipts: []Receipt{
				rcpt("baseline", 1, "sha:aaa", cond(quiet)),
				rcpt("candidate", 1, "sha:zzz", cond(quiet)),
			},
			wantCount: 0,
		},
		{
			name: "incomplete receipts are ignored",
			receipts: []Receipt{
				rcpt("moe", 1, "sha:aaa", cond(quiet)),
				{Name: "moe", Round: 2, Complete: false, Digest: "sha:bbb", Before: cond(pressed)},
			},
			wantCount: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SameArmDisagreements(tt.receipts, tt.tolerance)
			if len(got) != tt.wantCount {
				t.Fatalf("got %d disagreements, want %d: %v", len(got), tt.wantCount, got)
			}
			if tt.wantCount == 0 {
				return
			}
			d := got[0]
			if d.Confounded() != tt.wantConfounded {
				t.Errorf("Confounded() = %v, want %v (%s)", d.Confounded(), tt.wantConfounded, d)
			}
			if len(d.Compared) != tt.wantCompared {
				t.Errorf("Compared = %v, want %d kinds", d.Compared, tt.wantCompared)
			}
			var changed []ConditionKind
			for _, c := range d.Changed {
				changed = append(changed, c.Kind)
				if c.Delta != c.After-c.Before {
					t.Errorf("%s: Delta = %d, want %d", c.Kind, c.Delta, c.After-c.Before)
				}
			}
			if !kindsEqual(changed, tt.wantChanged) {
				t.Errorf("Changed kinds = %v, want %v", changed, tt.wantChanged)
			}
			if !kindsEqual(d.Unobserved, tt.wantUnobserved) {
				t.Errorf("Unobserved = %v, want %v", d.Unobserved, tt.wantUnobserved)
			}
		})
	}
}

func kindsEqual(got, want []ConditionKind) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

// TestSameArmDisagreementsRoundIdentity checks that a disagreement names the
// two rounds it compared, so a finding can be traced back to the run log.
func TestSameArmDisagreementsRoundIdentity(t *testing.T) {
	receipts := []Receipt{
		rcpt("moe", 1, "sha:aaa", cond(map[ConditionKind]int64{ConditionRetainedBytes: 0})),
		rcpt("moe", 2, "sha:aaa", cond(map[ConditionKind]int64{ConditionRetainedBytes: 0})),
		rcpt("moe", 5, "sha:bbb", cond(map[ConditionKind]int64{ConditionRetainedBytes: 77918175232})),
	}
	got := SameArmDisagreements(receipts, 0)
	if len(got) != 1 {
		t.Fatalf("got %d disagreements, want 1", len(got))
	}
	if got[0].Rounds != [2]int{1, 5} {
		t.Errorf("Rounds = %v, want [1 5]", got[0].Rounds)
	}
	if got[0].Digests != [2]string{"sha:aaa", "sha:bbb"} {
		t.Errorf("Digests = %v, want [sha:aaa sha:bbb]", got[0].Digests)
	}
	if !strings.Contains(got[0].String(), "retained_bytes") {
		t.Errorf("String() = %q, want it to name the changed condition", got[0].String())
	}
}

func TestInterleaveObservesConditions(t *testing.T) {
	var calls int
	observe := func(ctx context.Context) Conditions {
		calls++
		c := Conditions{Observed: timeAt(calls)}
		if err := c.Record(ConditionRetainedBytes, int64(calls)*giB); err != nil {
			t.Fatal(err)
		}
		return c
	}

	arms := []Arm{
		{Name: "a", Run: func(ctx context.Context) (Receipt, error) {
			return Receipt{Complete: true, Digest: "d"}, nil
		}},
		{Name: "b", Run: func(ctx context.Context) (Receipt, error) {
			// An arm that records its own conditions keeps them.
			r := Receipt{Complete: true, Digest: "d"}
			if err := r.Before.Record(ConditionDeviceFreeBytes, 7); err != nil {
				return r, err
			}
			return r, nil
		}},
	}

	receipts, err := InterleaveWithOptions(context.Background(), arms, 1, Options{Observe: observe})
	if err != nil {
		t.Fatalf("InterleaveWithOptions: %v", err)
	}
	if calls != 4 {
		t.Errorf("observe called %d times, want 4 (before and after each of 2 arms)", calls)
	}
	byArm := ByArm(receipts)

	a := byArm["a"][0]
	if !a.Before.Has(ConditionRetainedBytes) || !a.After.Has(ConditionRetainedBytes) {
		t.Fatalf("arm a: Before=%v After=%v, want both observed", a.Before.Values, a.After.Values)
	}
	if a.Before.Bytes(ConditionRetainedBytes) >= a.After.Bytes(ConditionRetainedBytes) {
		t.Errorf("arm a: Before %d not taken before After %d",
			a.Before.Bytes(ConditionRetainedBytes), a.After.Bytes(ConditionRetainedBytes))
	}
	if !a.Before.Observed.Before(a.After.Observed) {
		t.Errorf("arm a: Observed %v not before %v", a.Before.Observed, a.After.Observed)
	}

	b := byArm["b"][0]
	if got := b.Before.Bytes(ConditionDeviceFreeBytes); got != 7 {
		t.Errorf("arm b: Run-recorded Before was overwritten: device_free_bytes = %d, want 7", got)
	}
	if b.Before.Has(ConditionRetainedBytes) {
		t.Error("arm b: Run-recorded Before was merged with the observer's")
	}
	if !b.After.Has(ConditionRetainedBytes) {
		t.Error("arm b: After not observed")
	}
}

// TestInterleaveDefaultObserverStamps checks that the default observer runs and
// leaves a timestamp even where it can read nothing.
func TestInterleaveDefaultObserverStamps(t *testing.T) {
	arms := []Arm{{Name: "a", Run: func(ctx context.Context) (Receipt, error) {
		return Receipt{Complete: true, Digest: "d"}, nil
	}}}
	receipts, err := Interleave(context.Background(), arms, 1)
	if err != nil {
		t.Fatalf("Interleave: %v", err)
	}
	r := receipts[0]
	if r.Before.Observed.IsZero() || r.After.Observed.IsZero() {
		t.Errorf("default observer left Observed zero: Before=%v After=%v", r.Before.Observed, r.After.Observed)
	}
	if r.Before.Device != "" {
		t.Errorf("default observer recorded device %q", r.Before.Device)
	}
}

// timeAt returns distinct, increasing timestamps for a fake observer.
func timeAt(n int) time.Time {
	return time.Date(2026, 9, 22, 0, 0, n, 0, time.UTC)
}
