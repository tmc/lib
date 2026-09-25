package main

// A view is what shows in place of the process list with -x.
type view int

const (
	viewProcesses view = iota
	viewEvents
	viewDetail
)

// handleExt applies a key press that means something only with -x, and
// reports whether it did. Outside the process list, it takes every key
// but those that quit or redraw.
func (u *ui) handleExt(k key) bool {
	if u.overlay != overlayNone {
		return false
	}
	if u.player != nil {
		switch k {
		case ' ':
			if !u.ended {
				u.paused = !u.paused
				u.kick = !u.paused
			}
			return true
		case 'n':
			u.kick = u.paused && !u.ended
			return true
		case keyF9:
			// The processes of a recording are not ours to signal.
			return true
		}
	}
	switch u.view {
	case viewProcesses:
		switch k {
		case 'e':
			u.view = viewEvents
			u.eventOffset = 0
			return true
		case keyEnter:
			u.openDetail()
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
	case viewDetail:
		switch k {
		case keyEnter, keyEsc:
			u.view = viewProcesses
		case 'q', keyF10, keyF5, keyCtrlL:
			return false
		}
		return true
	}
	return false
}

// extShortcuts are the -x entries of the shortcut bar.
func (u *ui) extShortcuts() [][2]string {
	if !u.opts.ext {
		return nil
	}
	var keys [][2]string
	if !u.opts.noProcs {
		switch u.view {
		case viewEvents:
			keys = [][2]string{{"e", "Procs"}}
		case viewDetail:
			keys = [][2]string{{"ESC", "Back"}}
		default:
			keys = [][2]string{{"Enter", "Detail"}, {"e", "Events"}}
		}
	}
	switch {
	case u.player == nil:
	case u.ended:
		keys = append(keys, [2]string{"", "Replay ended"})
	case u.paused:
		keys = append(keys, [2]string{"Space", "Play"}, [2]string{"n", "Step"})
	default:
		keys = append(keys, [2]string{"Space", "Pause"})
	}
	return keys
}
