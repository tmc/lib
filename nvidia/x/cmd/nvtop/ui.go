package main

import (
	"cmp"
	"slices"
	"strings"
	"syscall"
	"time"
)

// A field is a column of the process list.
type field int

const (
	fieldPID field = iota
	fieldUser
	fieldDev
	fieldType
	fieldGPU
	fieldEnc
	fieldDec
	fieldMem
	fieldCPU
	fieldHostMem
	fieldCommand
	numFields
)

var fieldNames = [numFields]string{
	"PID", "USER", "DEV", "TYPE", "GPU", "ENC", "DEC", "GPU MEM", "CPU", "HOST MEM", "Command",
}

// fieldWidths are the column widths; USER grows to the longest name.
var fieldWidths = [numFields]int{7, 4, 3, 8, 4, 4, 4, 14, 6, 9, 0}

// displayed reports whether f is shown. As in nvtop's default
// configuration, the encoder and decoder columns are not.
func displayed(f field) bool { return f != fieldEnc && f != fieldDec }

// An overlay is the option window over the left of the process list.
type overlay int

const (
	overlayNone overlay = iota
	overlayKill
	overlaySort
)

const overlayWidth = 13

// A row is a process in the list, with the index of its device.
type row struct {
	dev int
	p   *process
}

// A ui is the state of the interactive interface.
type ui struct {
	opts options
	devs []*device
	pid  int // nvtop's own, left out of the list

	// history holds, per device, the plotted GPU and memory utilization.
	history [][2]series

	// Encoder and decoder meters hide once idle for opts.encodeHide.
	encSeen, decSeen []time.Time

	sortBy   field
	sortDesc bool

	rows      []row // the sorted list drawn last
	selected  int   // index in rows
	offset    int   // first row shown
	offsetCol int   // first column shown
	selPID    int   // pid of rows[selected], or -1

	overlay     overlay
	optSelected int
	optOffset   int

	// In the kill overlay, digits typed in a row select a signal by
	// number; number holds the digits so far.
	lastDigit bool
	number    int

	kill        func(pid int, sig syscall.Signal) error
	now         func() time.Time
	quit        bool
	redraw      bool // the terminal must be redrawn from scratch
	largestName int  // longest device name, at least 4

	// Whether any device reports ECC error counts, has NVLink, or has
	// active NVLink links. The device headers make room for these.
	ecc, nvlink, nvlinkActive bool

	// With -x, what shows in place of the process list, and the event
	// log with the number of events and whether any was a fault, per
	// device.
	view        view
	events      []event
	eventOffset int // newest events scrolled past
	eventCount  []int
	faulted     []bool

	// The process shown in detail, as last listed, and the plotted
	// history of every process.
	host        *host // where to read process details; nil if none
	detailKey   procKey
	detailProc  process
	procHistory map[procKey]*[2]series
}

func newUI(devs []*device, opts options, pid int, now func() time.Time) *ui {
	u := &ui{
		opts:        opts,
		devs:        devs,
		pid:         pid,
		history:     make([][2]series, len(devs)),
		encSeen:     make([]time.Time, len(devs)),
		decSeen:     make([]time.Time, len(devs)),
		sortBy:      fieldMem,
		sortDesc:    true,
		selPID:      -1,
		eventCount:  make([]int, len(devs)),
		faulted:     make([]bool, len(devs)),
		procHistory: make(map[procKey]*[2]series),
		kill:        sendSignal,
		now:         now,
	}
	// The meters start hidden: their last activity is set in the past
	// by more than the hiding time, as nvtop does.
	if opts.encodeHide > 0 {
		past := now().Add(-(time.Duration(opts.encodeHide.Minutes()) + 1) * time.Minute)
		for i := range devs {
			u.encSeen[i], u.decSeen[i] = past, past
		}
	}
	u.largestName = 4
	for _, d := range devs {
		if d.name.ok {
			u.largestName = max(u.largestName, width(d.name.v))
		}
		u.ecc = u.ecc || d.ecc
		u.nvlink = u.nvlink || d.nvlink.present()
		u.nvlinkActive = u.nvlinkActive || d.nvlink.links > 0
	}
	return u
}

// interval is the refresh interval: -d, or one second.
func (u *ui) interval() time.Duration {
	if u.opts.delay > 0 {
		return u.opts.delay
	}
	return time.Second
}

// frozen reports whether the process list must not change, because the
// user is choosing a signal to send to the selected process.
func (u *ui) frozen() bool { return u.overlay == overlayKill }

// record appends the plotted values of each device to its history.
// Missing values count as 0.
func (u *ui) record() {
	for i, d := range u.devs {
		u.history[i][0].push(d.gpuUtil.v)
		u.history[i][1].push(d.memUtil.v)
	}
	if u.opts.ext {
		u.recordProcs()
	}
}

// collect returns the processes of every device, without nvtop itself,
// in list order.
func (u *ui) collect() []row {
	var rows []row
	for i, d := range u.devs {
		for j := range d.procs {
			if d.procs[j].pid == u.pid {
				continue
			}
			rows = append(rows, row{i, &d.procs[j]})
		}
	}
	u.sortRows(rows)
	return rows
}

// sortRows sorts rows by u.sortBy. Unlike nvtop, whose comparators are
// not consistent orders, it sorts stably: missing values go last in both
// directions, and ties keep device and pid order.
func (u *ui) sortRows(rows []row) {
	slices.SortStableFunc(rows, func(a, b row) int {
		c := u.compare(a, b)
		if c == 0 {
			return cmp.Or(cmp.Compare(a.dev, b.dev), cmp.Compare(a.p.pid, b.p.pid))
		}
		return c
	})
}

func (u *ui) compare(a, b row) int {
	dir := 1
	if u.sortDesc {
		dir = -1
	}
	num := func(x, y opt[uint64]) int {
		switch {
		case x.ok && y.ok:
			return dir * cmp.Compare(x.v, y.v)
		case x.ok:
			return -1
		case y.ok:
			return 1
		}
		return 0
	}
	str := func(x, y opt[string]) int {
		switch {
		case x.ok && y.ok:
			return dir * strings.Compare(x.v, y.v)
		case x.ok:
			return -1
		case y.ok:
			return 1
		}
		return 0
	}
	u32 := func(v opt[uint32]) opt[uint64] { return opt[uint64]{uint64(v.v), v.ok} }
	p, q := a.p, b.p
	switch u.sortBy {
	case fieldPID:
		return dir * cmp.Compare(p.pid, q.pid)
	case fieldUser:
		return str(p.user, q.user)
	case fieldDev:
		return dir * cmp.Compare(a.dev, b.dev)
	case fieldType:
		return dir * cmp.Compare(p.kind, q.kind)
	case fieldGPU:
		return num(u32(p.gpuUtil), u32(q.gpuUtil))
	case fieldEnc:
		return num(u32(p.encUtil), u32(q.encUtil))
	case fieldDec:
		return num(u32(p.decUtil), u32(q.decUtil))
	case fieldMem:
		return num(p.gpuMem, q.gpuMem)
	case fieldCPU:
		return num(u32(p.cpuUtil), u32(q.cpuUtil))
	case fieldHostMem:
		return num(p.hostMem, q.hostMem)
	case fieldCommand:
		return str(p.cmdline, q.cmdline)
	}
	return 0
}

// handle applies a key press, following nvtop's main loop and
// interface_key.
func (u *ui) handle(k key) {
	if u.opts.ext && u.handleExt(k) {
		u.lastDigit = false
		return
	}
	switch k {
	case 'q':
		u.quit = true
	case keyEsc:
		if u.overlay == overlayNone {
			u.quit = true
		}
		u.overlay = overlayNone
	case keyF10:
		if u.overlay == overlayNone {
			u.quit = true
		}
	case keyF9:
		if !u.opts.noProcs && u.overlay == overlayNone {
			u.overlay = overlayKill
			u.optSelected = 0
			u.lastDigit = false
			u.number = 0
		}
	case keyF6:
		if !u.opts.noProcs && u.overlay == overlayNone {
			u.overlay = overlaySort
			u.optSelected = 0
		}
	case 'l', keyRight:
		if u.overlay == overlayNone {
			u.offsetCol += 4
		}
	case 'h', keyLeft:
		if u.overlay == overlayNone && u.offsetCol >= 4 {
			u.offsetCol -= 4
		}
	case 'k', keyUp:
		if u.overlay != overlayNone {
			if u.optSelected > 0 {
				u.optSelected--
			}
		} else if u.selected > 0 {
			u.selected--
		}
	case 'j', keyDown:
		if u.overlay != overlayNone {
			u.optSelected++
		} else {
			u.selected++
		}
	case '+':
		u.sortDesc = false
	case '-':
		u.sortDesc = true
	case keyEnter:
		switch u.overlay {
		case overlayKill:
			if u.optSelected > 0 && u.optSelected < len(signals) && u.selPID > 0 {
				u.kill(u.selPID, signals[u.optSelected].sig)
			}
		case overlaySort:
			u.changeSort()
		}
		u.overlay = overlayNone
	case keyF5, keyCtrlL:
		u.redraw = true
	case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		if u.overlay == overlayKill {
			d := int(k - '0')
			if n := u.number*10 + d; u.lastDigit && n < len(signals) {
				d = n
			}
			u.number = d
			u.optSelected = d
		}
	}
	u.lastDigit = '0' <= k && k <= '9'
}

// changeSort sorts by the column selected in the sort overlay.
func (u *ui) changeSort() {
	if u.optSelected == 0 {
		return
	}
	i := 0
	for f := range numFields {
		if !displayed(f) {
			continue
		}
		if i == u.optSelected-1 {
			u.sortBy = f
			return
		}
		i++
	}
}

// fitSelection follows update_selected_offset_with_window_size: it
// keeps *sel below n and scrolls *off so that *sel is among the rows
// shown.
func fitSelection(sel, off *int, rows, n int) {
	if n == 0 {
		return
	}
	if *sel > n-1 {
		*sel = n - 1
	}
	if *off > *sel {
		*off = *sel
	}
	if *off+rows-1 < *sel {
		*off = *sel - rows + 1
	}
	for rows > n-*off && *off != 0 {
		*off--
	}
}
