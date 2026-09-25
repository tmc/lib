package main

import (
	"bufio"
	"bytes"
	"io/fs"
	"math"
	"os"
	"os/user"
	"strconv"
	"strings"
	"time"
)

// A host reads what the operating system knows about processes and
// memory from a /proc file system.
type host struct {
	fsys     fs.FS
	now      func() time.Time
	hz       float64 // clock ticks per second
	pageSize uint64
	userName func(uid string) (string, bool)

	cache map[int]*hostProc
	seen  map[int]bool
}

// A hostProc is what host remembers about a process between refreshes.
type hostProc struct {
	user, cmdline opt[string]
	cpuTime       float64 // seconds; negative if unknown
	at            time.Time

	// The figures of the last fill, for a process on several devices.
	cpuUtil opt[uint32]
	hostMem opt[uint64]
}

func newHost(fsys fs.FS) *host {
	return &host{
		fsys:     fsys,
		now:      time.Now,
		hz:       100, // USER_HZ on every Linux architecture Go supports
		pageSize: uint64(os.Getpagesize()),
		userName: lookupUser,
		cache:    make(map[int]*hostProc),
		seen:     make(map[int]bool),
	}
}

func lookupUser(uid string) (string, bool) {
	u, err := user.LookupId(uid)
	if err != nil {
		return "", false
	}
	return u.Username, true
}

// fill sets the user, command line, CPU use, and resident memory of the
// processes on d. The user and command line are read once per process.
// CPU use is the share of one CPU the process used since the last fill,
// and 0 the first time a process is seen.
func (h *host) fill(d *device) {
	for i := range d.procs {
		p := &d.procs[i]
		c := h.cache[p.pid]
		if c == nil {
			c = &hostProc{cpuTime: -1}
			c.user = h.user(p.pid)
			c.cmdline = h.cmdline(p.pid)
			h.cache[p.pid] = c
		}
		p.user = c.user
		p.cmdline = c.cmdline
		if h.seen[p.pid] {
			p.cpuUtil = c.cpuUtil
			p.hostMem = c.hostMem
			continue
		}
		h.seen[p.pid] = true
		c.cpuUtil = opt[uint32]{}
		c.hostMem = opt[uint64]{}

		ticks, rss, ok := h.stat(p.pid)
		if !ok {
			c.cpuTime = -1
			p.cpuUtil, p.hostMem = c.cpuUtil, c.hostMem
			continue
		}
		now := h.now()
		cpu := ticks / h.hz
		if c.cpuTime >= 0 && now.After(c.at) {
			pct := math.Round(100 * (cpu - c.cpuTime) / now.Sub(c.at).Seconds())
			c.cpuUtil = some(uint32(max(pct, 0)))
		} else {
			c.cpuUtil = some(uint32(0))
		}
		c.hostMem = some(rss * h.pageSize)
		c.cpuTime = cpu
		c.at = now
		p.cpuUtil, p.hostMem = c.cpuUtil, c.hostMem
	}
}

// forget drops the cached processes that no device listed since the
// last call.
func (h *host) forget() {
	for pid := range h.cache {
		if !h.seen[pid] {
			delete(h.cache, pid)
		}
	}
	clear(h.seen)
}

// user returns the name of the user running pid, from the effective UID in
// /proc/pid/status.
func (h *host) user(pid int) opt[string] {
	data, err := fs.ReadFile(h.fsys, strconv.Itoa(pid)+"/status")
	if err != nil {
		return opt[string]{}
	}
	s := bufio.NewScanner(bytes.NewReader(data))
	for s.Scan() {
		rest, ok := strings.CutPrefix(s.Text(), "Uid:")
		if !ok {
			continue
		}
		f := strings.Fields(rest)
		if len(f) < 2 {
			break
		}
		if name, ok := h.userName(f[1]); ok {
			return some(name)
		}
		break
	}
	return opt[string]{}
}

// cmdline returns the command line of pid with its arguments separated by
// spaces.
func (h *host) cmdline(pid int) opt[string] {
	data, err := fs.ReadFile(h.fsys, strconv.Itoa(pid)+"/cmdline")
	if err != nil {
		return opt[string]{}
	}
	data = bytes.TrimSuffix(data, []byte{0})
	return some(string(bytes.ReplaceAll(data, []byte{0}, []byte{' '})))
}

// stat returns the user plus system CPU time of pid in clock ticks and
// its resident set size in pages, from /proc/pid/stat.
func (h *host) stat(pid int) (ticks float64, rss uint64, ok bool) {
	data, err := fs.ReadFile(h.fsys, strconv.Itoa(pid)+"/stat")
	if err != nil {
		return 0, 0, false
	}
	// The command name is in parentheses and may hold spaces or
	// parentheses itself; the fields after it start at the last ')'.
	i := bytes.LastIndexByte(data, ')')
	if i < 0 {
		return 0, 0, false
	}
	f := strings.Fields(string(data[i+1:]))
	// f[0] is field 3 (state); utime and stime are fields 14 and 15,
	// rss is field 24.
	if len(f) < 22 {
		return 0, 0, false
	}
	utime, err1 := strconv.ParseUint(f[11], 10, 64)
	stime, err2 := strconv.ParseUint(f[12], 10, 64)
	pages, err3 := strconv.ParseInt(f[21], 10, 64)
	if err1 != nil || err2 != nil || err3 != nil || pages < 0 {
		return 0, 0, false
	}
	return float64(utime + stime), uint64(pages), true
}

// systemMemory returns the MemTotal and MemAvailable lines of
// /proc/meminfo in bytes, as nvtop's set_unified_system_memory_info reads
// them. It reports false unless both are present, the total is nonzero,
// and the available memory does not exceed it.
func (h *host) systemMemory() (total, avail uint64, ok bool) {
	data, err := fs.ReadFile(h.fsys, "meminfo")
	if err != nil {
		return 0, 0, false
	}
	var haveTotal, haveAvail bool
	s := bufio.NewScanner(bytes.NewReader(data))
	for s.Scan() {
		key, rest, _ := strings.Cut(s.Text(), ":")
		f := strings.Fields(rest)
		if len(f) != 2 || f[1] != "kB" {
			continue
		}
		kb, err := strconv.ParseUint(f[0], 10, 64)
		if err != nil {
			continue
		}
		switch key {
		case "MemTotal":
			total, haveTotal = kb*1024, true
		case "MemAvailable":
			avail, haveAvail = kb*1024, true
		}
	}
	if !haveTotal || !haveAvail || total == 0 || avail > total {
		return 0, 0, false
	}
	return total, avail, true
}
