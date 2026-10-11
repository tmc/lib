package main

import (
	"slices"
	"syscall"
	"testing"
	"time"
)

var testNow = time.Date(2026, 9, 24, 12, 0, 0, 0, time.UTC)

func testUI(opts options) *ui {
	if opts.encodeHide == 0 {
		opts.encodeHide = 30 * time.Second
	}
	return newUI(testDevices(), opts, 1, func() time.Time { return testNow })
}

func pids(rows []row) []int {
	var p []int
	for _, r := range rows {
		p = append(p, r.p.pid)
	}
	return p
}

func TestSort(t *testing.T) {
	tests := []struct {
		by   field
		desc bool
		want []int
	}{
		{fieldMem, true, []int{1234, 4321, 999}},
		{fieldMem, false, []int{999, 4321, 1234}},
		{fieldPID, false, []int{999, 1234, 4321}},
		{fieldUser, false, []int{1234, 4321, 999}},
		{fieldType, false, []int{999, 1234, 4321}},
		{fieldCPU, true, []int{1234, 4321, 999}},
		{fieldCommand, false, []int{999, 4321, 1234}},
		// Missing values go last in both directions.
		{fieldGPU, true, []int{1234, 4321, 999}},
		{fieldGPU, false, []int{4321, 1234, 999}},
	}
	for _, tt := range tests {
		u := testUI(options{})
		u.sortBy, u.sortDesc = tt.by, tt.desc
		if got := pids(u.collect()); !slices.Equal(got, tt.want) {
			t.Errorf("sort by %s desc=%v: %v, want %v", fieldNames[tt.by], tt.desc, got, tt.want)
		}
	}
}

func TestCollectSkipsSelf(t *testing.T) {
	u := newUI(testDevices(), options{}, 999, func() time.Time { return testNow })
	if got := pids(u.collect()); slices.Contains(got, 999) {
		t.Errorf("collect includes nvtop's own pid: %v", got)
	}
}

func TestHandle(t *testing.T) {
	tests := []struct {
		name    string
		opts    options
		keys    []key
		quit    bool
		overlay overlay
		sortBy  field
		desc    bool
		sel     int
		col     int
	}{
		{name: "q quits", keys: []key{'q'}, quit: true, sortBy: fieldMem, desc: true},
		{name: "esc quits", keys: []key{keyEsc}, quit: true, sortBy: fieldMem, desc: true},
		{name: "f10 quits", keys: []key{keyF10}, quit: true, sortBy: fieldMem, desc: true},
		{name: "esc closes overlay", keys: []key{keyF9, keyEsc}, sortBy: fieldMem, desc: true},
		{name: "f10 in overlay does nothing", keys: []key{keyF6, keyF10}, overlay: overlaySort, sortBy: fieldMem, desc: true},
		{name: "f9 opens kill", keys: []key{keyF9}, overlay: overlayKill, sortBy: fieldMem, desc: true},
		{name: "no overlay without processes", opts: options{noProcs: true}, keys: []key{keyF9, keyF6}, sortBy: fieldMem, desc: true},
		{name: "sort by pid", keys: []key{keyF6, 'j', keyEnter}, sortBy: fieldPID, desc: true},
		{name: "sort by cpu", keys: []key{keyF6, keyDown, keyDown, keyDown, keyDown, keyDown, keyDown, keyDown, keyEnter}, sortBy: fieldCPU, desc: true},
		{name: "sort cancel", keys: []key{keyF6, keyEnter}, sortBy: fieldMem, desc: true},
		{name: "ascending", keys: []key{'+'}, sortBy: fieldMem},
		{name: "descending", keys: []key{'+', '-'}, sortBy: fieldMem, desc: true},
		{name: "selection", keys: []key{'j', 'j', 'k', keyDown}, sortBy: fieldMem, desc: true, sel: 2},
		{name: "selection floor", keys: []key{'k', keyUp}, sortBy: fieldMem, desc: true},
		{name: "columns", keys: []key{'l', keyRight, 'h'}, sortBy: fieldMem, desc: true, col: 4},
		{name: "column floor", keys: []key{'h', keyLeft}, sortBy: fieldMem, desc: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			u := testUI(tt.opts)
			for _, k := range tt.keys {
				u.handle(k)
			}
			if u.quit != tt.quit || u.overlay != tt.overlay || u.sortBy != tt.sortBy ||
				u.sortDesc != tt.desc || u.selected != tt.sel || u.offsetCol != tt.col {
				t.Errorf("after %q: quit %v overlay %v sort %s desc %v selected %d col %d",
					tt.keys, u.quit, u.overlay, fieldNames[u.sortBy], u.sortDesc, u.selected, u.offsetCol)
			}
		})
	}
}

func TestKill(t *testing.T) {
	type sent struct {
		pid int
		sig syscall.Signal
	}
	var got []sent
	u := testUI(options{})
	u.kill = func(pid int, sig syscall.Signal) error {
		got = append(got, sent{pid, sig})
		return nil
	}
	u.render(newScreen(120, 40, false)) // selects the first process
	if u.selPID != 1234 {
		t.Fatalf("selPID = %d, want 1234", u.selPID)
	}

	// Cancel sends nothing.
	u.handle(keyF9)
	u.handle(keyEnter)
	if len(got) != 0 || u.overlay != overlayNone {
		t.Fatalf("cancel sent %v, overlay %v", got, u.overlay)
	}

	// The list does not change under the kill overlay.
	u.handle(keyF9)
	if !u.frozen() {
		t.Errorf("not frozen under the kill overlay")
	}
	for range 9 {
		u.handle('j')
	}
	u.handle(keyEnter)
	var want []sent
	if len(signals) > 9 {
		want = []sent{{1234, signals[9].sig}}
	}
	if !slices.Equal(got, want) {
		t.Errorf("sent %v, want %v", got, want)
	}
	if u.frozen() {
		t.Errorf("still frozen after sending")
	}
}

func TestSignalDigits(t *testing.T) {
	tests := []struct {
		keys []key
		want int
	}{
		{[]key{'9'}, 9},
		{[]key{'1', '5'}, 15},
		{[]key{'1', '5', '2'}, 2}, // 152 names no signal: start over
		{[]key{'9', '9'}, 9},
		{[]key{'1', 'j', '5'}, 5}, // another key ends the number
	}
	for _, tt := range tests {
		u := testUI(options{})
		u.handle(keyF9)
		for _, k := range tt.keys {
			u.handle(k)
		}
		want := tt.want
		if len(signals) == 1 {
			want = 0 // Unsupported platforms offer only Cancel.
		}
		if u.optSelected != want {
			t.Errorf("after F9 %q: selected %d, want %d", tt.keys, u.optSelected, want)
		}
	}

	// Digits do nothing outside the kill overlay.
	u := testUI(options{})
	u.handle('3')
	if u.optSelected != 0 || u.selected != 0 {
		t.Errorf("digit outside overlay moved the selection")
	}
}

func TestFitSelection(t *testing.T) {
	tests := []struct {
		sel, off, rows, n int
		wantSel, wantOff  int
	}{
		{0, 0, 5, 10, 0, 0},
		{7, 0, 5, 10, 7, 3},  // scroll down to show the selection
		{2, 5, 5, 10, 2, 2},  // scroll up
		{12, 0, 5, 10, 9, 5}, // clamp to the last row
		{3, 4, 5, 6, 3, 1},   // do not leave blank rows below
		{3, 3, 5, 0, 3, 3},   // empty list: untouched
	}
	for _, tt := range tests {
		sel, off := tt.sel, tt.off
		fitSelection(&sel, &off, tt.rows, tt.n)
		if sel != tt.wantSel || off != tt.wantOff {
			t.Errorf("fitSelection(%d, %d, %d, %d) = %d, %d; want %d, %d",
				tt.sel, tt.off, tt.rows, tt.n, sel, off, tt.wantSel, tt.wantOff)
		}
	}
}

func TestEncoderHiding(t *testing.T) {
	u := testUI(options{})
	d := u.devs[0]
	enc, dec := u.showEncDec(0, d, testNow)
	if !enc || dec {
		t.Errorf("active encoder, idle decoder: shown %v %v, want true false", enc, dec)
	}
	d.encUtil = some[uint32](0)
	if enc, _ := u.showEncDec(0, d, testNow.Add(29*time.Second)); !enc {
		t.Errorf("encoder hidden 29s after use")
	}
	if enc, _ := u.showEncDec(0, d, testNow.Add(31*time.Second)); enc {
		t.Errorf("encoder shown 31s after use")
	}

	always := testUI(options{encodeHide: -time.Second})
	if enc, dec := always.showEncDec(1, always.devs[1], testNow); !enc || !dec {
		t.Errorf("negative hiding time: shown %v %v, want true true", enc, dec)
	}
}
