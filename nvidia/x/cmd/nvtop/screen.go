package main

import (
	"bytes"
	"fmt"
	"strings"
	"unicode"
)

// A color is one of the six colors nvtop draws with, or none.
type color uint8

const (
	noColor color = iota
	cyan
	red
	green
	yellow
	blue
	magenta
)

// A style is how a cell is drawn: a foreground color and whether it
// is in standout (reverse video).
type style struct {
	fg       color
	standout bool
}

var plain style

func fg(c color) style { return style{fg: c} }

func standout(c color) style { return style{fg: c, standout: true} }

type cell struct {
	r  rune
	st style
}

// A screen is a grid of cells that flush draws on a terminal, rewriting
// only the lines that changed since the last flush.
type screen struct {
	w, h  int
	cells []cell
	prev  []cell // what the terminal shows; nil forces a full redraw
	color bool
}

func newScreen(w, h int, color bool) *screen {
	s := &screen{w: w, h: h, color: color}
	s.cells = make([]cell, w*h)
	s.clear()
	return s
}

// clear blanks every cell.
func (s *screen) clear() {
	for i := range s.cells {
		s.cells[i] = cell{r: ' '}
	}
}

// invalidate makes the next flush redraw every line.
func (s *screen) invalidate() { s.prev = nil }

// String returns the text of the screen, one line per row with trailing
// blanks removed. Tests compare it to golden files.
func (s *screen) String() string {
	var b strings.Builder
	for y := range s.h {
		var line []rune
		for _, c := range s.cells[y*s.w : (y+1)*s.w] {
			if c.r != 0 {
				line = append(line, c.r)
			}
		}
		b.WriteString(strings.TrimRight(string(line), " "))
		b.WriteByte('\n')
	}
	return b.String()
}

// flush appends to buf the escape sequences that bring a terminal showing
// the previous flush up to date, and returns the result.
func (s *screen) flush(buf []byte) []byte {
	b := bytes.NewBuffer(buf)
	full := len(s.prev) != len(s.cells)
	if full {
		s.prev = make([]cell, len(s.cells))
		b.WriteString("\x1b[H\x1b[2J")
	}
	for y := range s.h {
		row := s.cells[y*s.w : (y+1)*s.w]
		old := s.prev[y*s.w : (y+1)*s.w]
		if !full && equalCells(row, old) {
			continue
		}
		fmt.Fprintf(b, "\x1b[%d;1H", y+1)
		cur := plain
		b.WriteString("\x1b[0m")
		for x, c := range row {
			if c.st != cur {
				b.WriteString(s.sgr(c.st))
				cur = c.st
			}
			// A wide character is written only with its filler after
			// it, and the filler only as part of it; a half left by
			// later drawing is a blank.
			switch {
			case c.r == 0:
				if x == 0 || runeWidth(row[x-1].r) != 2 {
					b.WriteByte(' ')
				}
			case runeWidth(c.r) == 2 && (x+1 == len(row) || row[x+1].r != 0):
				b.WriteByte(' ')
			default:
				b.WriteRune(c.r)
			}
		}
		b.WriteString("\x1b[0m")
		copy(old, row)
	}
	return b.Bytes()
}

func equalCells(a, b []cell) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// sgr returns the escape sequence that selects st.
func (s *screen) sgr(st style) string {
	seq := "\x1b[0"
	if st.standout {
		seq += ";7"
	}
	if s.color && st.fg != noColor {
		seq += fmt.Sprintf(";%d", ansiColor[st.fg])
	}
	return seq + "m"
}

var ansiColor = [...]int{
	cyan:    36,
	red:     31,
	green:   32,
	yellow:  33,
	blue:    34,
	magenta: 35,
}

// A window is a rectangle of a screen, like an ncurses window. Drawing
// outside it is clipped; nothing wraps.
type window struct {
	s          *screen
	x, y, w, h int
}

func (s *screen) window(x, y, w, h int) *window {
	return &window{s: s, x: x, y: y, w: max(w, 0), h: max(h, 0)}
}

func (w *window) at(x, y int) *cell {
	if x < 0 || y < 0 || x >= w.w || y >= w.h {
		return nil
	}
	sx, sy := w.x+x, w.y+y
	if sx < 0 || sy < 0 || sx >= w.s.w || sy >= w.s.h {
		return nil
	}
	return &w.s.cells[sy*w.s.w+sx]
}

// print writes str at (x, y) in st and returns the column after it.
// As in ncurses, a tab advances to the next multiple of 8 columns, control
// characters show as ^X and M-^X, and wide characters take two columns.
// Characters of no width are dropped.
func (w *window) print(x, y int, str string, st style) int {
	for _, r := range str {
		switch {
		case r == '\t':
			for {
				w.put(x, y, ' ', st)
				x++
				if x%8 == 0 {
					break
				}
			}
		case r < 0x20 || r == 0x7f:
			w.put(x, y, '^', st)
			w.put(x+1, y, r^0x40, st)
			x += 2
		case r >= 0x80 && r < 0xa0:
			x = w.print(x, y, "M-^"+string((r-0x80)^0x40), st)
		default:
			switch runeWidth(r) {
			case 1:
				w.put(x, y, r, st)
				x++
			case 2:
				// The second column holds a filler cell; see flush.
				// With no second column, the character is a blank.
				if w.at(x+1, y) == nil {
					w.put(x, y, ' ', st)
				} else {
					w.put(x, y, r, st)
					w.put(x+1, y, 0, st)
				}
				x += 2
			}
		}
	}
	return x
}

// printf is print with formatting.
func (w *window) printf(x, y int, st style, format string, args ...any) int {
	return w.print(x, y, fmt.Sprintf(format, args...), st)
}

// put writes r at (x, y) in st.
func (w *window) put(x, y int, r rune, st style) {
	if c := w.at(x, y); c != nil {
		*c = cell{r, st}
	}
}

// chgat sets the style of n cells from (x, y), as mvwchgat does.
// A negative n means to the end of the line.
func (w *window) chgat(x, y, n int, st style) {
	if n < 0 {
		n = w.w - x
	}
	for i := range n {
		if c := w.at(x+i, y); c != nil {
			c.st = st
		}
	}
}

// between sets the style of the cells of row y in [start, end), clamped
// to the window, as nvtop's set_attribute_between does.
func (w *window) between(y, start, end int, st style) {
	if start >= w.w || end < 0 {
		return
	}
	start = max(start, 0)
	end = min(end, w.w)
	w.chgat(start, y, end-start, st)
}

// hline draws n copies of r from (x, y) rightward.
func (w *window) hline(x, y, n int, r rune, st style) {
	for i := range n {
		w.put(x+i, y, r, st)
	}
}

// vline draws n copies of r from (x, y) downward.
func (w *window) vline(x, y, n int, r rune, st style) {
	for i := range n {
		w.put(x, y+i, r, st)
	}
}

// width returns the number of columns s takes.
func width(s string) int {
	n := 0
	for _, r := range s {
		n += runeWidth(r)
	}
	return n
}

// runeWidth returns the number of columns r takes on a terminal: 0 for
// combining and format characters, 2 for East Asian wide and fullwidth
// characters and emoji, and 1 otherwise. The wide ranges are the main
// blocks of Unicode's East Asian Width property, not all of it.
func runeWidth(r rune) int {
	if r == 0 || unicode.In(r, unicode.Mn, unicode.Me, unicode.Cf) {
		return 0
	}
	for _, g := range wideRanges {
		if r < g[0] {
			break
		}
		if r <= g[1] {
			return 2
		}
	}
	return 1
}

var wideRanges = [][2]rune{
	{0x1100, 0x115f},   // Hangul Jamo initials
	{0x2e80, 0x303e},   // CJK radicals, punctuation
	{0x3041, 0x33ff},   // kana, CJK symbols
	{0x3400, 0x4dbf},   // CJK extension A
	{0x4e00, 0x9fff},   // CJK unified ideographs
	{0xa000, 0xa4cf},   // Yi
	{0xac00, 0xd7a3},   // Hangul syllables
	{0xf900, 0xfaff},   // CJK compatibility ideographs
	{0xfe30, 0xfe4f},   // CJK compatibility forms
	{0xff00, 0xff60},   // fullwidth forms
	{0xffe0, 0xffe6},   // fullwidth signs
	{0x1f300, 0x1f64f}, // pictographs, emoticons
	{0x1f900, 0x1f9ff}, // supplemental pictographs
	{0x20000, 0x2fffd}, // CJK extensions B-F
	{0x30000, 0x3fffd}, // CJK extension G
}

const (
	hlineRune = '─'
	vlineRune = '│'
	ulCorner  = '┌'
	urCorner  = '┐'
	llCorner  = '└'
	lrCorner  = '┘'
	tTee      = '┬'
	bTee      = '┴'
	plusRune  = '┼'
)
