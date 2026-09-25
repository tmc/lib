package main

import (
	"testing"
	"testing/fstest"
	"time"
)

func testHost(fsys fstest.MapFS, now *time.Time) *host {
	h := newHost(fsys)
	h.now = func() time.Time { return *now }
	h.pageSize = 4096
	h.userName = func(uid string) (string, bool) {
		if uid == "1000" {
			return "alice", true
		}
		return "", false
	}
	return h
}

// stat returns a /proc/pid/stat line with the given CPU ticks and
// resident pages, and a command name that holds ") (".
func stat(utime, stime, rss string) string {
	return "42 (a) (b) S 1 42 42 0 -1 4194304 0 0 0 0 " + utime + " " + stime +
		" 0 0 20 0 1 0 100 1000000 " + rss + " 0\n"
}

func TestHostFill(t *testing.T) {
	fsys := fstest.MapFS{
		"42/status":  {Data: []byte("Name:\tpython\nUid:\t1001\t1000\t1000\t1000\n")},
		"42/cmdline": {Data: []byte("python\x00train.py\x00--fast\x00")},
		"42/stat":    {Data: []byte(stat("100", "50", "256"))},
		"7/status":   {Data: []byte("Uid:\t0\t0\t0\t0\n")},
	}
	now := time.Unix(1000, 0)
	h := testHost(fsys, &now)

	d := &device{procs: []process{{pid: 42}, {pid: 7}}}
	h.fill(d)
	h.forget()
	p := d.procs[0]
	if p.user != some("alice") {
		t.Errorf("user = %+v, want alice", p.user)
	}
	if p.cmdline != some("python train.py --fast") {
		t.Errorf("cmdline = %+v", p.cmdline)
	}
	if p.cpuUtil != some[uint32](0) {
		t.Errorf("first cpuUtil = %+v, want 0", p.cpuUtil)
	}
	if p.hostMem != some[uint64](256*4096) {
		t.Errorf("hostMem = %+v, want %d", p.hostMem, 256*4096)
	}
	q := d.procs[1]
	if q.user.ok || q.cmdline.ok || q.cpuUtil.ok || q.hostMem.ok {
		t.Errorf("unreadable process: %+v", q)
	}

	// 2s later the process used 100 more ticks: 1s of CPU, 50%.
	now = now.Add(2 * time.Second)
	fsys["42/stat"] = &fstest.MapFile{Data: []byte(stat("150", "100", "256"))}
	d.procs = []process{{pid: 42}}
	h.fill(d)
	h.forget()
	if got := d.procs[0].cpuUtil; got != some[uint32](50) {
		t.Errorf("second cpuUtil = %+v, want 50", got)
	}
	if _, ok := h.cache[7]; ok {
		t.Errorf("pid 7 not forgotten")
	}
}

// A process on two devices is measured once per refresh.
func TestHostFillShared(t *testing.T) {
	fsys := fstest.MapFS{"42/stat": {Data: []byte(stat("100", "0", "1"))}}
	now := time.Unix(1000, 0)
	h := testHost(fsys, &now)
	h.fill(&device{procs: []process{{pid: 42}}})
	h.forget()
	now = now.Add(time.Second)
	fsys["42/stat"] = &fstest.MapFile{Data: []byte(stat("200", "0", "1"))}
	a := &device{procs: []process{{pid: 42}}}
	b := &device{procs: []process{{pid: 42}}}
	h.fill(a)
	h.fill(b)
	h.forget()
	if a.procs[0].cpuUtil != some[uint32](100) || b.procs[0].cpuUtil != a.procs[0].cpuUtil {
		t.Errorf("cpuUtil = %+v and %+v, want 100 twice", a.procs[0].cpuUtil, b.procs[0].cpuUtil)
	}
}

func TestHostSystemMemory(t *testing.T) {
	tests := []struct {
		meminfo     string
		total, free uint64
		ok          bool
	}{
		{"MemTotal:       16384 kB\nMemFree: 1 kB\nMemAvailable:    4096 kB\n", 16384 * 1024, 4096 * 1024, true},
		{"MemTotal: 16 kB\nMemAvailable: 16 kB\n", 16 * 1024, 16 * 1024, true},
		{"MemTotal: 16384 kB\nMemFree: 1 kB\n", 0, 0, false},
		{"MemAvailable: 1 kB\n", 0, 0, false},
		{"MemTotal: 0 kB\nMemAvailable: 0 kB\n", 0, 0, false},
		{"MemTotal: 1 kB\nMemAvailable: 2 kB\n", 0, 0, false},
		{"MemTotal: 12 MB\nMemAvailable: 1 MB\n", 0, 0, false},
	}
	for _, tt := range tests {
		h := newHost(fstest.MapFS{"meminfo": {Data: []byte(tt.meminfo)}})
		total, free, ok := h.systemMemory()
		if total != tt.total || free != tt.free || ok != tt.ok {
			t.Errorf("systemMemory(%q) = %d, %d, %v, want %d, %d, %v", tt.meminfo, total, free, ok, tt.total, tt.free, tt.ok)
		}
	}
	if _, _, ok := newHost(fstest.MapFS{}).systemMemory(); ok {
		t.Errorf("systemMemory without meminfo reports ok")
	}
}
