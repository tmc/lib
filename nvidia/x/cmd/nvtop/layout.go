package main

// This file ports nvtop's interface_layout_selection.c. It computes in
// unsigned arithmetic, as the original does, so that a terminal too small
// for the layout wraps the same way.

// A rect is a window position and size.
type rect struct{ x, y, w, h int }

// A layout is where each part of the interface goes.
type layout struct {
	devices []rect
	plots   []rect
	// plotDevices lists, per plot, the devices drawn on it.
	plotDevices [][]int
	processes   rect
}

const (
	boxCols     = 5 // columns a plot needs for its frame and axis labels
	minPlotRows = 7
	maxPlotRows = 23 // when the process list shares the screen
	maxPlotLine = 4  // lines on one plot
)

func minPlotCols(info uint) uint { return boxCols + 10*info }

func minProcessRows(rows, ndev uint) uint {
	return 1 + max(5, min(rows/4, ndev*3))
}

// computeLayout follows compute_sizes_from_layout. info[i] is the number
// of values plotted for device i; rows excludes the shortcut line.
func computeLayout(info []uint, headerRows, headerCols, rows, cols uint, hideProcs bool) layout {
	ndev := uint(len(info))
	perRow := max(1, cols/headerCols)
	headerStacks := max(1, ndev/perRow+b2u(ndev%perRow > 0))
	if ndev%headerStacks == 0 {
		perRow = ndev / headerStacks
	}
	headerMin := headerStacks * headerRows
	procMin := minProcessRows(rows, ndev)
	if rows < headerMin+procMin {
		if rows >= headerMin+2 {
			procMin = rows - headerMin
		} else {
			headerMin = rows
			procMin = 0
		}
	}
	if hideProcs {
		procMin = 0
	}
	headerRowsUsed := headerMin
	procRows := procMin
	plotRows := rows - headerMin - procMin

	plotOf := make([]uint, ndev)
	plotStack := make([]uint, ndev)
	nplots, stacks := placePlots(plotRows, cols, info, plotOf, plotStack)

	// Spare lines separate the header rows.
	transferable := plotRows - stacks*minPlotRows
	spare := headerStacks - 1
	spaceBetweenStacks := false
	if transferable >= spare {
		headerRowsUsed += spare
		plotRows -= spare
		spaceBetweenStacks = true
	}

	if stacks > 0 {
		for stacks < nplots && plotRows/(stacks+1) >= 11 && (stacks+1)*minPlotRows <= plotRows {
			stacks++
		}
	}

	infoPer := make([]uint, nplots)
	for p := range nplots {
		for d := range ndev {
			if plotOf[d] == p {
				infoPer[p] += info[d]
			}
		}
	}
	stackCols := make([]uint, stacks)
	for s := range stacks {
		for p := range nplots {
			if plotStack[p] == s {
				stackCols[s] += minPlotCols(infoPer[p])
			}
		}
	}
	balanceStacks(cols, stacks, nplots, infoPer, stackCols, plotStack)

	var l layout
	left := cols - perRow*headerCols
	spaceBetweenCols := false
	if left > perRow {
		spaceBetweenCols = true
		left -= perRow - 1
	}
	x0 := b2u(left > 0)
	x, y := x0, uint(0)
	n := uint(0)
	for range ndev {
		l.devices = append(l.devices, rect{int(x), int(y), int(headerCols), int(headerRows)})
		n++
		if n == perRow {
			x = x0
			y += headerRows + b2u(spaceBetweenStacks)
			n = 0
		} else {
			x += headerCols + b2u(spaceBetweenCols)
		}
	}

	var procExtra uint
	if nplots > 0 {
		perStack := plotRows / stacks
		if !hideProcs && perStack > maxPlotRows {
			perStack = maxPlotRows
		}
		y := headerRowsUsed
		for s := range stacks {
			var inStack, lines uint
			for p := range nplots {
				if plotStack[p] == s {
					inStack++
					lines += infoPer[p]
				}
			}
			lineCols := cols - inStack*boxCols
			x := uint(0)
			for p := range nplots {
				if plotStack[p] != s {
					continue
				}
				maxCols := boxCols + lineCols*infoPer[p]/lines
				plotCols := maxCols - (maxCols-boxCols)%infoPer[p]
				l.plots = append(l.plots, rect{int(x), int(y), int(plotCols), int(perStack)})
				x += maxCols
			}
			y += perStack
		}
		procExtra = plotRows - perStack*stacks
		for p := range nplots {
			var devs []int
			for d := range ndev {
				if plotOf[d] == p {
					devs = append(devs, int(d))
				}
			}
			l.plotDevices = append(l.plotDevices, devs)
		}
	} else if plotRows > 0 {
		procRows += plotRows - 1
	}
	h := procRows + procExtra
	l.processes = rect{0, int(rows - h), int(cols), int(h)}
	return l
}

// placePlots follows preliminary_plot_positioning. It sets the plot
// each device is drawn on and the stack each plot is in, merging the
// plots of neighboring devices until they fit, and returns the number
// of plots and of stacks.
func placePlots(rows, cols uint, info []uint, plotOf, plotStack []uint) (nplots, stacks uint) {
	ndev := uint(len(info))
	per := make([]uint, ndev)
	plotted := false
	for i := range ndev {
		per[i] = info[i]
		plotOf[i] = i
		if per[i] > 0 {
			plotted = true
		}
	}
	search := plotted && rows >= minPlotRows
	for search {
		search = false
		id := uint(0)
		stacks = 1
		used := uint(0)
		rowsLeft := rows - minPlotRows
		for i := uint(0); i < ndev; i++ {
			if per[i] == 0 {
				continue
			}
			c := minPlotCols(per[i])
			if cols >= c+used {
				used += c
				plotStack[id] = stacks - 1
				id++
				continue
			}
			if used == 0 {
				stacks = 0
				break
			}
			if rowsLeft >= minPlotRows {
				rowsLeft -= minPlotRows
				stacks++
				used = 0
				i--
				continue
			}
			if a, b, ok := whoToMerge(per); ok {
				per[a] += per[b]
				per[b] = 0
				old := plotOf[b]
				for d := range ndev {
					if plotOf[d] == old {
						plotOf[d] = plotOf[a]
					}
				}
				search = true
			} else {
				stacks = 0
			}
			break
		}
	}
	if stacks == 0 {
		return 0, 0
	}
	// Move nonempty plots over those emptied by merges.
	for i := range ndev {
		if per[i] > 0 {
			nplots++
			continue
		}
		for j := i + 1; j < ndev; j++ {
			if per[j] == 0 {
				continue
			}
			per[i], per[j] = per[j], 0
			for d := range ndev {
				if plotOf[d] == j {
					plotOf[d] = i
				}
			}
			nplots++
			break
		}
	}
	return nplots, stacks
}

// whoToMerge follows who_to_merge: it returns the two neighboring
// nonempty plots whose merge holds the fewest lines, at most maxPlotLine,
// preferring plots at the end on ties.
func whoToMerge(per []uint) (a, b uint, ok bool) {
	best := ^uint(0)
	n := uint(len(per))
	for i := n - 1; i < n; i-- {
		if per[i] == 0 {
			continue
		}
		with := i
		for j := i - 1; j < i; j-- {
			if per[j] > 0 {
				with = j
				break
			}
		}
		m := per[i] + per[with]
		if with < i && m <= maxPlotLine && m < best {
			best = m
			a, b = with, i
		}
	}
	return a, b, best != ^uint(0)
}

// balanceStacks follows balance_info_on_stacks_preserving_plot_order:
// it moves plots to later stacks while that evens out the columns used by
// each stack.
func balanceStacks(maxCols, stacks, nplots uint, per, stackCols, plotStack []uint) {
	stacks = min(stacks, nplots)
	move := func(p, to uint) bool {
		if plotStack[p] == to {
			return false
		}
		c := minPlotCols(per[p])
		if stackCols[to]+c > maxCols {
			return false
		}
		stackCols[plotStack[p]] -= c
		stackCols[to] += c
		plotStack[p] = to
		return true
	}
	diff := func() uint {
		var sum uint
		for i := range stacks {
			for j := i + 1; j < stacks; j++ {
				if stackCols[i] > stackCols[j] {
					sum += stackCols[i] - stackCols[j]
				} else {
					sum += stackCols[j] - stackCols[i]
				}
			}
		}
		return sum
	}
	for p := nplots - 1; p < nplots; p-- {
		to := plotStack[p] + 1
		if to >= stacks {
			continue
		}
		before, from := diff(), plotStack[p]
		if !move(p, to) {
			continue
		}
		if diff() <= before {
			p = nplots // start over from the last plot
			continue
		}
		move(p, from)
	}
}

func b2u(b bool) uint {
	if b {
		return 1
	}
	return 0
}
