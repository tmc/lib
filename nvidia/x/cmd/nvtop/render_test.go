package main

import (
	"strings"
	"testing"
	"time"
)

func TestRender(t *testing.T) {
	tests := []struct {
		name string
		w, h int
		opts options
		keys []key
	}{
		{name: "default", w: 120, h: 40},
		{name: "small", w: 80, h: 24},
		{name: "tiny", w: 40, h: 10},
		{name: "wide", w: 200, h: 50},
		{name: "no-plot", w: 120, h: 40, opts: options{noPlot: true}},
		{name: "no-processes", w: 120, h: 40, opts: options{noProcs: true}},
		{name: "reverse", w: 120, h: 40, opts: options{reverse: true}},
		{name: "always-encdec", w: 120, h: 40, opts: options{encodeHide: -1}},
		{name: "kill", w: 120, h: 40, keys: []key{keyF9, 'j', 'j'}},
		{name: "sort", w: 120, h: 40, keys: []key{keyF6, 'j', 'j', 'j', 'j', 'j', 'j', 'j', keyEnter, '+'}},
		{name: "sort-overlay", w: 120, h: 40, keys: []key{keyF6}},
		{name: "scrolled", w: 120, h: 40, keys: []key{'j', 'l', 'l'}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			u := testUI(tt.opts)
			// A ramp for the GPU line, a constant for the memory one.
			for i := range 200 {
				u.history[0][0].push(uint32(i % 101))
				u.history[0][1].push(45)
				u.history[1][0].push(uint32(100 - i%101))
				u.history[1][1].push(0)
			}
			s := newScreen(tt.w, tt.h, true)
			u.render(s) // the first draw sets the selection
			for _, k := range tt.keys {
				u.handle(k)
			}
			u.render(s)
			golden(t, "render/"+tt.name+".golden", []byte(s.String()))
		})
	}
}

// datacenterDevices returns two devices with ECC counters and NVLink:
// one with 18 active links, and one with no bridge whose memory has had
// uncorrected errors.
func datacenterDevices() []*device {
	return []*device{
		{
			index:      0,
			name:       some("NVIDIA H100 80GB HBM3"),
			gpuClock:   some[uint32](1980),
			memClock:   some[uint32](2619),
			temp:       some[uint32](40),
			power:      some[uint32](120000),
			powerMax:   some[uint32](700000),
			ecc:        true,
			eccCorr:    some[uint64](1234),
			eccUncorr:  some[uint64](0),
			nvlink:     nvlink{version: 4, links: 18},
			nvlinkRate: some[uint64](3 << 20),
		},
		{
			index:       1,
			name:        some("NVIDIA H100 80GB HBM3"),
			memClockMax: some[uint32](2619),
			fan:         some[uint32](30),
			ecc:         true,
			eccCorr:     some[uint64](5),
			eccUncorr:   some[uint64](3),
			nvlink:      nvlink{version: 4},
		},
	}
}

func TestRenderDatacenter(t *testing.T) {
	u := newUI(datacenterDevices(), options{encodeHide: 30 * time.Second}, 1, func() time.Time { return testNow })
	s := newScreen(120, 20, true)
	u.render(s)
	golden(t, "render/datacenter.golden", []byte(s.String()))
	checkColors(t, s, []colorCheck{
		// As in nvtop, a two-digit link count has no space before it.
		{"NVL418x", 0, cyan},
		{"NVL418x", 3, magenta},
		{"NVL418x", 4, noColor},
		{"NVL4 0x", 3, magenta},
		{"ECC 999/ 0", 0, cyan},
		{"ECC 999/ 0", 5, noColor},
		{"ECC   5/ 3", 4, red},
		{"ECC   5/ 3", 9, red},
	})
}

// A colorCheck says that the cell off cells into text has color want.
type colorCheck struct {
	text string
	off  int
	want color
}

func checkColors(t *testing.T, s *screen, checks []colorCheck) {
	t.Helper()
	for _, tt := range checks {
		x, y, ok := find(s, tt.text)
		if !ok {
			t.Errorf("%q not drawn", tt.text)
			continue
		}
		if got := s.cells[y*s.w+x+tt.off].st.fg; got != tt.want {
			t.Errorf("%q column %d color %d, want %d", tt.text, tt.off, got, tt.want)
		}
	}
}

// find returns the cell at which text starts on s.
func find(s *screen, text string) (x, y int, ok bool) {
	for y := range s.h {
		var line []rune
		for _, c := range s.cells[y*s.w : (y+1)*s.w] {
			line = append(line, c.r)
		}
		if i := strings.Index(string(line), text); i >= 0 {
			return len([]rune(string(line)[:i])), y, true
		}
	}
	return 0, 0, false
}

func TestHeaderWidth(t *testing.T) {
	tests := []struct {
		name   string
		ecc    bool
		nvlink nvlink
		want   int
	}{
		{"plain", false, nvlink{}, 64},
		{"ecc", true, nvlink{}, 75},
		{"nvlink without links", false, nvlink{version: 4}, 94},
		{"nvlink", false, nvlink{version: 4, links: 2}, 91},
		{"ecc and nvlink", true, nvlink{version: 4, links: 2}, 101},
	}
	for _, tt := range tests {
		d := &device{name: some("H100"), ecc: tt.ecc, nvlink: tt.nvlink}
		u := newUI([]*device{d}, options{}, 1, func() time.Time { return testNow })
		if got := u.headerWidth(); got != tt.want {
			t.Errorf("%s: headerWidth = %d, want %d", tt.name, got, tt.want)
		}
	}
}

func TestDataRate(t *testing.T) {
	tests := []struct {
		kb   uint64
		want string
	}{
		{0, "0.000 KiB/s"},
		{999, "999.0 KiB/s"},
		{1000, "0.977 MiB/s"},
		{3 << 20, "3.000 GiB/s"},
		{50 << 20, "50.00 GiB/s"},
		{1 << 50, "1024.0 PiB/s"},
	}
	for _, tt := range tests {
		if got := dataRate(tt.kb); got != tt.want {
			t.Errorf("dataRate(%d) = %q, want %q", tt.kb, got, tt.want)
		}
	}
}

func TestFlush(t *testing.T) {
	s := newScreen(10, 2, true)
	w := s.window(0, 0, 10, 2)
	w.print(0, 0, "ab", fg(red))
	w.print(0, 1, "cd", standout(cyan))
	got := string(s.flush(nil))
	for _, want := range []string{"\x1b[H\x1b[2J", "\x1b[0;31m", "ab", "\x1b[0;7;36m", "cd"} {
		if !strings.Contains(got, want) {
			t.Errorf("first flush %q lacks %q", got, want)
		}
	}

	// Only the changed line is rewritten.
	w.print(0, 1, "x", plain)
	got = string(s.flush(nil))
	if strings.Contains(got, "ab") || !strings.Contains(got, "x") {
		t.Errorf("second flush %q", got)
	}
	if got := string(s.flush(nil)); got != "" {
		t.Errorf("unchanged flush = %q, want nothing", got)
	}

	mono := newScreen(4, 1, false)
	mono.window(0, 0, 4, 1).print(0, 0, "ab", standout(green))
	if got := string(mono.flush(nil)); strings.Contains(got, "32m") || !strings.Contains(got, "7m") {
		t.Errorf("monochrome flush %q: want standout without color", got)
	}
}

func TestPrintWide(t *testing.T) {
	tests := []struct {
		in        string
		wantText  string // screen.String
		wantFlush string // the row's characters as written
	}{
		{"a中b", "a中b", "a中b      "},
		{"éx", "ex", "ex        "},
		{"\u009b2J", "M-^[2J", "M-^[2J    "},
		{"a\tb", "a       b", "a       b "},
		{"123456789中", "123456789", "123456789 "}, // no room for the second column
	}
	for _, tt := range tests {
		s := newScreen(10, 1, false)
		n := s.window(0, 0, 10, 1).print(0, 0, tt.in, plain)
		if got := strings.TrimSuffix(s.String(), "\n"); got != tt.wantText {
			t.Errorf("print(%q): screen %q, want %q", tt.in, got, tt.wantText)
		}
		out := string(s.flush(nil))
		out = out[strings.Index(out, "1;1H\x1b[0m")+len("1;1H\x1b[0m"):]
		out = strings.TrimSuffix(out, "\x1b[0m")
		if out != tt.wantFlush {
			t.Errorf("print(%q): flushed %q, want %q (next column %d)", tt.in, out, tt.wantFlush, n)
		}
	}

	// Overwriting half of a wide character blanks the other half.
	s := newScreen(4, 1, false)
	w := s.window(0, 0, 4, 1)
	w.print(0, 0, "中文", plain)
	w.print(1, 0, "x", plain)
	out := string(s.flush(nil))
	if !strings.Contains(out, " x文") {
		t.Errorf("half-overwritten wide character flushed as %q", out)
	}
}

// The plot lines and legends take nvtop's default plot colors in order:
// cyan, yellow, green, blue.
func TestPlotColors(t *testing.T) {
	u := testUI(options{})
	s := newScreen(120, 40, true)
	u.render(s)
	for _, tt := range []struct {
		legend string
		want   color
	}{
		{"GPU0 %", cyan},
		{"GPU0 mem%", yellow},
	} {
		found := false
		for y := range s.h {
			row := s.cells[y*s.w : (y+1)*s.w]
			var line []rune
			for _, c := range row {
				line = append(line, c.r)
			}
			x := strings.Index(string(line), tt.legend)
			if x < 0 {
				continue
			}
			found = true
			x = len([]rune(string(line)[:x]))
			if got := row[x].st.fg; got != tt.want {
				t.Errorf("legend %q color %d, want %d", tt.legend, got, tt.want)
			}
			break
		}
		if !found {
			t.Errorf("legend %q not drawn", tt.legend)
		}
	}
}
