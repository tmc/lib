package main

// A view is what shows in place of the process list with -x.
type view int

const (
	viewProcesses view = iota
	viewEvents
)

// handleExt applies a key press that means something only with -x, and
// reports whether it did. Outside the process list, it takes every key
// but those that quit or redraw.
func (u *ui) handleExt(k key) bool {
	if u.overlay != overlayNone {
		return false
	}
	switch u.view {
	case viewProcesses:
		if k == 'e' {
			u.view = viewEvents
			u.eventOffset = 0
			return true
		}
		return false
	case viewEvents:
		switch k {
		case 'e', keyEsc:
			u.view = viewProcesses
		case 'k', keyUp:
			u.eventOffset = max(0, u.eventOffset-1)
		case 'j', keyDown:
			u.eventOffset++
		case 'q', keyF10, keyF5, keyCtrlL:
			return false
		}
		return true
	}
	return false
}

// extShortcuts are the -x entries of the shortcut bar.
func (u *ui) extShortcuts() [][2]string {
	if !u.opts.ext || u.opts.noProcs {
		return nil
	}
	switch u.view {
	case viewEvents:
		return [][2]string{{"e", "Procs"}}
	}
	return [][2]string{{"e", "Events"}}
}
