package main

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"time"
)

// Widths of the device header fields, from nvtop's sizeof_device_field.
// The name field is 11 plus the longest device name. The fan field
// narrows when a device has active NVLink links, to make room for them.
const (
	fanWidth       = 11
	narrowFanWidth = 8
	tempWidth      = 10
	powerWidth     = 15
	eccWidth       = 10
	clockWidth     = 11
	memClockWidth  = 12
	pcieWidth      = 46
)

// headerRows is the height of a device header: three lines, one more
// with -i, and one more with -x.
func (u *ui) headerRows() uint {
	return 3 + b2u(u.opts.infoBar) + b2u(u.opts.ext)
}

// The widths of the fields on a header's fourth line.
const (
	infoWidth         = 11
	nvlinkErrorsWidth = 33
)

func (u *ui) nameWidth() int { return 11 + u.largestName }

func (u *ui) fanWidth() int {
	if u.nvlinkActive {
		return narrowFanWidth
	}
	return fanWidth
}

// nvlinkStart and nvlinkWidth follow nvlink_line2_start and
// nvlink_line2_width: the NVLink field ends a header's second line.
func (u *ui) nvlinkStart() int {
	x := 6 + clockWidth + memClockWidth + tempWidth + u.fanWidth() + powerWidth
	if u.ecc {
		x += eccWidth
	}
	return x
}

const nvlinkWidth = pcieWidth - powerWidth - 3

// headerWidth follows device_length.
func (u *ui) headerWidth() int {
	line2 := clockWidth + memClockWidth + tempWidth + u.fanWidth() + powerWidth + 5
	if u.ecc {
		line2 += eccWidth + 1
	}
	if u.nvlink {
		line2 = max(line2, u.nvlinkStart()+nvlinkWidth+1)
	}
	return max(u.nameWidth()+pcieWidth+1, line2)
}

// layout returns where each part goes on a screen of the given size.
func (u *ui) layout(w, h int) layout {
	info := make([]uint, len(u.devs))
	if !u.opts.noPlot {
		for i := range info {
			info[i] = 2 // GPU and memory utilization
		}
	}
	return computeLayout(info, u.headerRows(), uint(u.headerWidth()), uint(max(h-1, 0)), uint(w), u.opts.noProcs)
}

// render draws the whole interface on s.
func (u *ui) render(s *screen) {
	s.clear()
	l := u.layout(s.w, s.h)
	now := u.now()
	for i, d := range u.devs {
		u.drawDevice(s, l.devices[i], i, d, now)
	}
	for i, r := range l.plots {
		if u.view == viewDetail && slices.Contains(l.plotDevices[i], u.detailKey.dev) {
			u.drawDetailPlot(s, r)
			continue
		}
		var lines []*series
		var legend []string
		for _, d := range l.plotDevices[i] {
			lines = append(lines, &u.history[d][0], &u.history[d][1])
			legend = append(legend, fmt.Sprintf("GPU%d %%", d), fmt.Sprintf("GPU%d mem%%", d))
		}
		u.drawPlot(s, r, lines, legend)
	}
	if !u.opts.noProcs && l.processes.h > 0 {
		switch u.view {
		case viewProcesses:
			u.drawProcesses(s, l.processes)
		case viewEvents:
			u.drawEvents(s, l.processes)
		case viewDetail:
			u.drawDetail(s, l.processes)
		}
	}
	u.drawShortcuts(s.window(0, s.h-1, s.w, 1))
}

var memoryPrefix = [...]string{" B", "Ki", "Mi", "Gi", "Ti", "Pi"}

// drawDevice follows draw_devices for device i at r.
func (u *ui) drawDevice(s *screen, r rect, i int, d *device, now time.Time) {
	const sp = 1
	nw := u.nameWidth()

	// Line 1: name and PCIe.
	w := s.window(r.x, r.y, nw, 1)
	x := w.printf(0, 0, fg(cyan), "Device %-2d", i)
	if d.name.ok {
		w.print(x, 0, "["+d.name.v+"]", plain)
	} else {
		w.print(x, 0, "[N/A]", plain)
	}
	w = s.window(r.x+sp+nw, r.y, pcieWidth, 1)
	x = w.print(0, 0, "PCIe ", fg(cyan))
	x = w.print(x, 0, "GEN ", fg(magenta))
	if d.pcieGen.ok && d.pcieWidth.ok {
		x = w.printf(x, 0, plain, "%d@%2dx", d.pcieGen.v, d.pcieWidth.v)
	} else {
		x = w.print(x, 0, "N/A", plain)
	}
	x = w.print(x, 0, " RX: ", fg(magenta))
	x = w.print(x, 0, rateOrNA(d.pcieRx), plain)
	x = w.print(x, 0, " TX: ", fg(magenta))
	w.print(x, 0, rateOrNA(d.pcieTx), plain)

	// Line 2: clocks, temperature, fan, power, ECC, NVLink.
	y := r.y + 1
	x0 := r.x
	w = s.window(x0, y, clockWidth, 1)
	if d.gpuClock.ok {
		w.printf(0, 0, plain, "GPU %dMHz", d.gpuClock.v)
	} else {
		w.print(0, 0, "GPU N/A MHz", plain)
	}
	w.chgat(0, 0, 3, fg(cyan))
	x0 += clockWidth + sp
	w = s.window(x0, y, memClockWidth, 1)
	if d.memClock.ok {
		w.printf(0, 0, plain, "MEM %dMHz", d.memClock.v)
	} else if d.memClockMax.ok {
		w.printf(0, 0, plain, "MEM %dMHz", d.memClockMax.v)
	} else {
		w.print(0, 0, "MEM N/A MHz", plain)
	}
	w.chgat(0, 0, 3, fg(cyan))
	x0 += memClockWidth + sp
	w = s.window(x0, y, tempWidth, 1)
	unit := "°C"
	if u.opts.fahrenheit {
		unit = "°F"
	}
	if d.temp.ok {
		t := d.temp.v
		shown := t
		if u.opts.fahrenheit {
			shown = toFahrenheit(t)
		}
		slow := uint32(0)
		if d.tempSlowdown.ok {
			slow = d.tempSlowdown.v
		}
		c := green
		// nvtop compares against slowdown-5 in unsigned arithmetic,
		// so a threshold under 5 always shows green.
		if slow >= 5 && t >= slow-5 {
			c = yellow
			if t >= slow {
				c = red
			}
		}
		x := w.print(0, 0, "TEMP", fg(cyan))
		x = w.printf(x, 0, fg(c), " %3d", shown)
		w.print(x, 0, unit, plain)
	} else {
		w.print(0, 0, "TEMP N/A"+unit, plain)
		w.chgat(0, 0, 4, fg(cyan))
	}
	x0 += tempWidth + sp
	w = s.window(x0, y, u.fanWidth(), 1)
	switch {
	case d.fan.ok && u.nvlinkActive:
		w.printf(0, 0, plain, "FAN %3d%%", min(d.fan.v, 100))
		w.chgat(0, 0, 3, fg(cyan))
	case d.fan.ok:
		w.printf(0, 0, plain, " FAN %3d%%  ", min(d.fan.v, 100))
		w.chgat(1, 0, 3, fg(cyan))
	case u.nvlinkActive:
		w.print(0, 0, "FAN N/A", plain)
		w.chgat(0, 0, 3, fg(cyan))
	default:
		w.print(0, 0, "  FAN N/A  ", plain)
		w.chgat(2, 0, 3, fg(cyan))
	}
	x0 += u.fanWidth() + sp
	w = s.window(x0, y, powerWidth, 1)
	switch {
	case d.power.ok && d.powerMax.ok:
		w.printf(0, 0, plain, "POW %3d / %3d W", d.power.v/1000, d.powerMax.v/1000)
	case d.power.ok:
		w.printf(0, 0, plain, "POW %3d W", d.power.v/1000)
	case d.powerMax.ok:
		w.printf(0, 0, plain, "POW N/A / %3d W", d.powerMax.v/1000)
	default:
		w.print(0, 0, "POW N/A W", plain)
	}
	w.chgat(0, 0, 3, fg(cyan))
	if u.ecc && (d.eccCorr.ok || d.eccUncorr.ok) {
		w = s.window(x0+powerWidth+sp, y, eccWidth, 1)
		w.printf(0, 0, plain, "ECC %3d/%2d", min(d.eccCorr.v, 999), min(d.eccUncorr.v, 99))
		w.chgat(0, 0, 3, fg(cyan))
		if d.eccUncorr.v > 0 {
			// An uncorrected error is a hardware fault.
			w.chgat(4, 0, -1, fg(red))
		}
	}
	if u.nvlink && d.nvlink.present() {
		w = s.window(r.x+u.nvlinkStart(), y, nvlinkWidth, 1)
		x := w.print(0, 0, "NVL", fg(cyan))
		if d.nvlink.version > 0 {
			x = w.printf(x, 0, fg(magenta), "%d", d.nvlink.version)
		} else {
			x = w.print(x, 0, "?", fg(magenta))
		}
		switch n := d.nvlink.links; {
		case n == 0:
			w.print(x, 0, " 0x", plain)
		case n < 10:
			x = w.printf(x, 0, plain, " %dx ", n)
		default:
			x = w.printf(x, 0, plain, "%dx ", n)
		}
		if d.nvlink.links > 0 && d.nvlinkRate.ok {
			w.print(x, 0, dataRate(d.nvlinkRate.v), plain)
		}
	}

	// Line 3: meters.
	y = r.y + 2
	gpuW, memW, encW, decW := meterWidths(r.w)
	showEnc, showDec := u.showEncDec(i, d, now)
	var gpuWin, memWin, encWin, decWin *window
	switch {
	case showEnc && showDec:
		gpuWin = s.window(r.x, y, gpuW, 1)
		memWin = s.window(r.x+sp+gpuW, y, memW, 1)
		encWin = s.window(r.x+2*sp+gpuW+memW, y, encW, 1)
		decWin = s.window(r.x+3*sp+gpuW+memW+encW, y, decW, 1)
	case showEnc || showDec:
		// The one meter shown goes where the decoder's would.
		gpuWin = s.window(r.x, y, gpuW+encW/2+1, 1)
		memWin = s.window(r.x+sp+gpuW+encW/2+1, y, memW+encW/2, 1)
		decWin = s.window(r.x+3*sp+gpuW+memW+encW, y, decW, 1)
		encWin = decWin
	default:
		gpuWin = s.window(r.x, y, gpuW+encW+1, 1)
		memWin = s.window(r.x+sp+gpuW+encW+1, y, memW+encW+1, 1)
	}
	if showEnc {
		meter(encWin, "ENC", d.encUtil.v, fmt.Sprintf("%d%%", d.encUtil.v))
	}
	if showDec {
		meter(decWin, "DEC", d.decUtil.v, fmt.Sprintf("%d%%", d.decUtil.v))
	}
	switch {
	case d.gpuUtil.ok && d.effLoad.ok:
		meter(gpuWin, "GPU", d.gpuUtil.v, fmt.Sprintf("%d%%(eff %d%%)", d.gpuUtil.v, d.effLoad.v))
		highlight(gpuWin, "GPU", d.gpuUtil.v, d.effLoad.v)
	case d.gpuUtil.ok:
		meter(gpuWin, "GPU", d.gpuUtil.v, fmt.Sprintf("%d%%", d.gpuUtil.v))
	default:
		meter(gpuWin, "GPU", 0, "N/A")
	}
	switch {
	case d.memTotal.ok && d.memUsed.ok:
		total, used := float64(d.memTotal.v), float64(d.memUsed.v)
		i := 0
		for ; i < 5 && total >= 1000; i++ {
			total /= 1024
			used /= 1024
		}
		pct := uint32(100 * float64(d.memUsed.v) / float64(d.memTotal.v))
		meter(memWin, "MEM", pct, fmt.Sprintf("%.3f%s/%.3f%s", used, memoryPrefix[i], total, memoryPrefix[i]))
	case d.memTotal.ok:
		total := float64(d.memTotal.v)
		i := 0
		for ; i < 5 && total >= 1000; i++ {
			total /= 1024
		}
		meter(memWin, "MEM", 0, fmt.Sprintf("N/A/%.3f%s", total, memoryPrefix[i]))
	default:
		meter(memWin, "MEM", 0, "N/A")
	}

	y = r.y + 3
	if u.opts.infoBar {
		u.drawInfoBar(s, r.x, y, d)
		y++
	}
	if u.opts.ext {
		u.drawExtLine(s.window(r.x, y, r.w, 1), d)
	}
}

// drawInfoBar follows the gpu_info_bar part of draw_devices: the
// shader cores, L2 cache, and execution engines, and the NVLink errors.
// NVML reports neither of the last two.
func (u *ui) drawInfoBar(s *screen, x, y int, d *device) {
	const sp = 1
	info := func(x int, label string, v opt[uint32]) {
		w := s.window(x, y, infoWidth, 1)
		n := w.print(0, 0, label+" ", fg(cyan))
		if v.ok {
			w.printf(n, 0, plain, "%d", v.v)
		} else {
			w.print(n, 0, "N/A", plain)
		}
	}
	info(x, "NSHC", d.cores)
	info(x+sp+infoWidth, "L2CF", opt[uint32]{})
	info(x+2*(sp+infoWidth), "NEXC", opt[uint32]{})
	if !u.nvlinkActive || !d.nvlinkErrs.ok {
		return
	}
	w := s.window(x+3*(sp+infoWidth), y, nvlinkErrorsWidth, 1)
	e := d.nvlinkErrs.v
	n := w.print(0, 0, "NVL", fg(cyan))
	count := func(label string, v uint64, c color) {
		n = w.print(n, 0, label, plain)
		a := plain
		if v > 0 {
			a = fg(c)
		}
		n = w.printf(n, 0, a, "%05d", v%100000)
	}
	count(" FL:", e.flit, red)
	count(" EE:", e.ecc, red)
	count(" CR:", e.crc, yellow)
}

// showEncDec follows encode_decode_show_select: a meter shows while its
// rate is above zero and until it has been idle for the hiding time.
func (u *ui) showEncDec(i int, d *device, now time.Time) (enc, dec bool) {
	hide := u.opts.encodeHide
	show := func(v opt[uint32], seen *time.Time) bool {
		if v.ok && v.v > 0 {
			*seen = now
			return true
		}
		return !(hide > 0 && now.Sub(*seen) > hide)
	}
	return show(d.encUtil, &u.encSeen[i]), show(d.decUtil, &u.decSeen[i])
}

// meterWidths follows alloc_device_window's split of a header's third
// line among the GPU, memory, encoder and decoder meters.
func meterWidths(total int) (gpu, mem, enc, dec int) {
	remaining := total - 3
	quot, rem := remaining/3, remaining%3
	gpu, mem, enc = quot, quot, quot
	switch rem {
	case 2:
		if enc%2 == 1 {
			enc++
		} else {
			mem++
		}
		gpu++
	case 1:
		gpu++
	}
	if enc%2 == 1 {
		mem++
		enc--
	}
	if enc/2 < 14 {
		enc += 2
		gpu--
		mem--
	}
	dec = enc / 2
	if enc%2 == 1 {
		enc++
	}
	enc /= 2
	return gpu, mem, enc, dec
}

// meter follows draw_percentage_meter: prelude, then a bracketed bar
// filled to pct in green, with text right-aligned inside.
func meter(w *window, prelude string, pct uint32, text string) {
	x := w.print(0, 0, prelude, fg(cyan))
	w.put(x, 0, '[', plain)
	x++
	inner := w.w - width(prelude) - 2
	used := int(math.Round(float64(inner) * float64(pct) / 100))
	w.hline(x, 0, used, '|', plain)
	w.hline(x+used, 0, inner-used, ' ', plain)
	w.put(x+inner, 0, ']', plain)
	w.print(x+inner-width(text), 0, text, plain)
	w.chgat(x, 0, used, fg(green))
}

// highlight follows draw_percentage_meter_with_yellow_highlight: it
// turns the first yellowPct of a meter's bar yellow.
func highlight(w *window, prelude string, pct, yellowPct uint32) {
	yellowPct = min(yellowPct, pct)
	inner := w.w - width(prelude) - 2
	n := int(math.Round(float64(inner) * float64(yellowPct) / 100))
	w.chgat(width(prelude)+1, 0, n, fg(yellow))
}

func rateOrNA(v opt[uint32]) string {
	if !v.ok {
		return "N/A"
	}
	return dataRate(uint64(v.v))
}

// dataRate follows print_data_at_scale for a rate in KB/s.
func dataRate(v uint64) string {
	f := float64(v)
	i := 1
	for ; i < 5 && f >= 1000; i++ {
		f /= 1024
	}
	var num string
	switch {
	case f >= 100:
		num = strconv.FormatFloat(f, 'f', 1, 64)
	case f >= 10:
		num = strconv.FormatFloat(f, 'f', 2, 64)
	default:
		num = strconv.FormatFloat(f, 'f', 3, 64)
	}
	return num + " " + memoryPrefix[i] + "B/s"
}

// drawPlot follows initialize_gpu_mem_plot and draw_plots for the plot
// at r showing the given lines.
func (u *ui) drawPlot(s *screen, r rect, lines []*series, legend []string) {
	outer := s.window(r.x, r.y, r.w, r.h)
	cols, rows := r.w-boxCols, r.h-2
	box(outer, 3, 0, cols+2, rows+2)
	outer.print(0, 1+rows*3/4, " 25", plain)
	outer.print(0, 1+rows/4, " 75", plain)
	outer.print(0, 1+rows/2, " 50", plain)
	outer.print(0, 1, "100", plain)
	outer.print(0, rows, "  0", plain)

	n := len(lines)
	ms := int(u.interval() / time.Millisecond)
	secs := func(num, den int) string {
		s := fmt.Sprintf("%ds", ms*cols*num/den/n/1000)
		if len(s) > 15 {
			return "err"
		}
		return s
	}
	bottom := r.h - 1
	at := func(x int, s string) { outer.print(x, bottom, s, plain) }
	if u.opts.reverse {
		at(4, "0s")
		t := secs(1, 4)
		at(4+cols/4-len(t)/2, t)
		t = secs(1, 2)
		at(4+cols/2-len(t)/2, t)
		t = secs(3, 4)
		at(4+cols*3/4-len(t)/2, t)
		t = secs(1, 1)
		at(4+cols-len(t), t)
	} else {
		at(4, secs(1, 1))
		t := secs(3, 4)
		at(4+cols/4-len(t)/2, t)
		t = secs(1, 2)
		at(4+cols/2-len(t)/2, t)
		t = secs(1, 4)
		at(4+cols*3/4-len(t)/2, t)
		at(4+cols-2, "0s")
	}

	if cols <= 0 || rows <= 0 {
		return
	}
	data := make([]float64, cols)
	maxCopy := cols / n
	for line, h := range lines {
		for j := 0; j < h.len() && j < maxCopy; j++ {
			t := maxCopy - j - 1
			if u.opts.reverse {
				t = j
			}
			data[t*n+line] = float64(h.recent(j))
		}
	}
	linePlot(s.window(r.x+4, r.y+1, cols, rows), data, n, !u.opts.reverse, legend)
}

// box follows draw_rectangle.
func box(w *window, x, y, width, height int) {
	w.hline(x+1, y, width-2, hlineRune, plain)
	w.hline(x+1, y+height-1, width-2, hlineRune, plain)
	w.vline(x, y+1, height-2, vlineRune, plain)
	w.vline(x+width-1, y+1, height-2, vlineRune, plain)
	w.put(x, y, ulCorner, plain)
	w.put(x+width-1, y, urCorner, plain)
	w.put(x, y+height-1, llCorner, plain)
	w.put(x+width-1, y+height-1, lrCorner, plain)
}

// lineColors are the colors of a plot's lines, in order.
var lineColors = [maxPlotLine]color{cyan, yellow, green, blue}

// linePlot follows nvtop_line_plot. data holds the lines interleaved:
// data[i*n+k] is sample i of line k. Each sample takes one column.
func linePlot(w *window, data []float64, n int, legendLeft bool, legend []string) {
	if len(data) == 0 || n == 0 {
		return
	}
	rows := w.h - 1
	incr := 100 / float64(rows)
	level := func(v float64) int { return int(float64(rows) - math.Round(v/incr)) }
	before := make([]int, n)
	for k := range n {
		before[k] = level(data[k])
	}
	for i := 0; i+n <= len(data) && i < w.w; i += n {
		for k := range n {
			now := level(data[i+k])
			st := fg(lineColors[k])
			x := i + k
			if now != before[k] {
				down := before[k] < now
				top, bot := min(before[k], now), max(before[k], now)
				if down {
					w.put(x, top, urCorner, st)
					w.put(x, bot, llCorner, st)
				} else {
					w.put(x, top, ulCorner, st)
					w.put(x, bot, lrCorner, st)
				}
				if bot-top > 1 {
					w.vline(x, top+1, bot-top-1, vlineRune, st)
				}
				for j := range n {
					switch {
					case j == k:
					case before[j] == bot:
						w.put(x, bot, bTee, st)
					case before[j] == top:
						w.put(x, top, tTee, st)
					case before[j] > top && before[j] < bot:
						w.put(x, before[j], plusRune, st)
					default:
						w.put(x, before[j], hlineRune, fg(lineColors[j]))
					}
				}
			} else {
				w.put(x, now, hlineRune, st)
				for j := range n {
					if j != k && before[j] != now {
						w.put(x, before[j], hlineRune, fg(lineColors[j]))
					}
				}
			}
			before[k] = now
		}
	}
	for i := 0; i < n && i < len(legend) && i < rows; i++ {
		st := fg(lineColors[i])
		l := []rune(legend[i])
		switch {
		case legendLeft:
			w.print(0, i, string(l[:min(len(l), w.w)]), st)
		case len(l) <= w.w:
			w.print(w.w-len(l), i, legend[i], st)
		default:
			w.print(0, i, string(l[:len(l)-w.w]), st)
		}
	}
}

// lineBufferSize bounds a process line, as nvtop's print buffer does.
const lineBufferSize = 8192

// drawProcesses follows draw_processes and print_processes_on_screen.
func (u *ui) drawProcesses(s *screen, r rect) {
	win := s.window(r.x, r.y, r.w, r.h)
	if u.overlay != overlayNone {
		u.drawOverlay(s.window(r.x, r.y, overlayWidth, r.h))
		win = s.window(r.x+overlayWidth, r.y, r.w-overlayWidth, r.h)
	}

	rows := u.collect()
	u.rows = rows
	if len(rows) > 0 {
		u.selected = min(u.selected, len(rows)-1)
		u.selPID = rows[u.selected].p.pid
	} else {
		u.selected = 0
		u.selPID = -1
	}
	widths := fieldWidths
	for _, r := range rows {
		if r.p.user.ok {
			widths[fieldUser] = max(widths[fieldUser], width(r.p.user.v))
		}
	}

	avail := win.h - 1
	fitSelection(&u.selected, &u.offset, avail, len(rows))
	if u.offsetCol+win.w >= lineBufferSize {
		u.offsetCol = lineBufferSize - win.w - 1
	}
	off := u.offsetCol

	var hdr []rune
	sortStart, sortEnd := 0, widths[0]
	for f := range numFields {
		if f == u.sortBy {
			sortStart = len(hdr)
			sortEnd = sortStart + widths[f]
			if f == fieldCommand {
				sortEnd = lineBufferSize - 4
			}
		}
		if displayed(f) {
			hdr = append(hdr, []rune(fmt.Sprintf("%*s ", widths[f], fieldNames[f]))...)
		}
	}
	win.print(0, 0, visible(hdr, off, win.w), plain)
	win.chgat(0, 0, -1, standout(green))
	win.between(0, sortStart-off, sortEnd-off, standout(cyan))

	typeStart := 0
	for f := fieldPID; f < fieldType; f++ {
		if displayed(f) {
			typeStart += widths[f] + 1
		}
	}
	typeEnd := typeStart + widths[fieldType]

	for i := u.offset; i < u.offset+avail && i < len(rows); i++ {
		y := i - u.offset + 1
		p := rows[i].p
		win.print(0, y, visible([]rune(u.processLine(rows[i], widths)), off, win.w), plain)
		if i == u.selected {
			win.chgat(0, y, -1, standout(cyan))
			continue
		}
		switch p.kind {
		case graphicCompute:
			win.between(y, typeStart-off, typeStart-off+4, fg(cyan))
			win.between(y, typeEnd-off-3, typeEnd-off-2, fg(yellow))
			win.between(y, typeEnd-off-1, typeEnd-off, fg(magenta))
		case graphic:
			win.between(y, typeStart-off, typeEnd-off, fg(yellow))
		default:
			win.between(y, typeStart-off, typeEnd-off, fg(magenta))
		}
	}
}

// visible returns the at most n runes of line from column off.
func visible(line []rune, off, n int) string {
	if off >= len(line) {
		return ""
	}
	line = line[off:]
	return string(line[:min(len(line), n)])
}

// processLine formats the displayed columns of r.
func (u *ui) processLine(r row, widths [numFields]int) string {
	p := r.p
	var b []byte
	col := func(f field, s string) {
		if displayed(f) {
			b = fmt.Appendf(b, "%*s ", widths[f], s)
		}
	}
	rate := func(f field, v opt[uint32]) {
		if !displayed(f) {
			return
		}
		if v.ok {
			b = fmt.Appendf(b, "%3d%% ", v.v)
		} else {
			b = append(b, "N/A  "...)
		}
	}
	col(fieldPID, truncate(strconv.Itoa(p.pid), widths[fieldPID]))
	col(fieldUser, orNA(p.user))
	col(fieldDev, truncate(strconv.Itoa(r.dev), widths[fieldDev]))
	switch p.kind {
	case graphicCompute:
		col(fieldType, "Both G+C")
	case graphic:
		col(fieldType, "Graphic")
	default:
		col(fieldType, "Compute")
	}
	rate(fieldGPU, p.gpuUtil)
	rate(fieldEnc, p.encUtil)
	rate(fieldDec, p.decUtil)
	mem := "N/A"
	if p.gpuMem.ok {
		mib := fmt.Sprintf("%6dMiB", uint32(p.gpuMem.v/(1<<20)))
		if p.memPct.ok {
			mem = truncate(mib, 9) + truncate(fmt.Sprintf(" %3d%%", p.memPct.v), 5)
		} else {
			mem = truncate(mib, widths[fieldMem]-1)
		}
	}
	col(fieldMem, mem)
	cpu := "   N/A"
	if p.cpuUtil.ok {
		cpu = truncate(fmt.Sprintf("%d%%", p.cpuUtil.v), widths[fieldCPU])
	}
	col(fieldCPU, cpu)
	host := "N/A"
	if p.hostMem.ok {
		host = truncate(fmt.Sprintf("%dMiB", p.hostMem.v/(1<<20)), widths[fieldHostMem])
	}
	col(fieldHostMem, host)
	if displayed(fieldCommand) && p.cmdline.ok {
		b = append(b, p.cmdline.v...)
	}
	return string(b)
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

func orNA(v opt[string]) string {
	if v.ok {
		return v.v
	}
	return "N/A"
}

// drawOverlay follows update_process_option_win, draw_kill_option and
// draw_sort_option.
func (u *ui) drawOverlay(w *window) {
	var title string
	var options []string
	switch u.overlay {
	case overlayKill:
		title = "Send signal:"
		for i, s := range signals {
			options = append(options, fmt.Sprintf("%2d %s", i, s.name))
		}
	case overlaySort:
		title = "Sort by     "
		options = append(options, "Cancel")
		for f := range numFields {
			if displayed(f) {
				options = append(options, fieldNames[f])
			}
		}
	}
	fitSelection(&u.optSelected, &u.optOffset, w.h-1, len(options))
	w.print(0, 0, title, standout(green))
	w.print(width(title), 0, " ", plain)
	y := 1
	for i := u.optOffset; i < len(options) && y < w.h; i++ {
		if i == u.optSelected {
			w.print(0, y, fmt.Sprintf("%-*s", overlayWidth-1, options[i]), standout(cyan))
		} else {
			w.print(0, y, options[i], plain)
		}
		y++
	}
}

// drawShortcuts follows draw_process_shortcuts. The Setup (F2) and Save
// Config (F12) entries are left out: this nvtop has neither.
func (u *ui) drawShortcuts(w *window) {
	var keys [][2]string
	switch u.overlay {
	case overlayNone:
		if !u.opts.noProcs && u.view == viewProcesses {
			keys = append(keys, [2]string{"F6", "Sort"}, [2]string{"F9", "Kill"})
		}
		keys = append(keys, [2]string{"F10", "Quit"})
		keys = append(keys, u.extShortcuts()...)
	case overlayKill:
		keys = [][2]string{{"Enter", "Send"}, {"ESC", "Cancel"}}
	case overlaySort:
		keys = [][2]string{{"Enter", "Sort"}, {"ESC", "Cancel"}, {"+", "Ascending"}, {"-", "Descending"}}
	}
	x := 0
	for _, k := range keys {
		x = w.print(x, 0, k[0], plain)
		x = w.print(x, 0, fmt.Sprintf("%-8s", k[1]), standout(cyan))
	}
	w.chgat(x, 0, -1, standout(cyan))
}
