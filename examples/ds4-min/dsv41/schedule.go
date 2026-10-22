package dsv41

import "fmt"

// Layer describes one of the 40 V4.1 Flash layers. Owner is -1 for raw-only
// layers. Every layer has a private raw window; Owner names compressed KV only.
type Layer struct {
	Ratio         int
	Owner         int
	PublishKV     bool
	RefreshIndex  bool
	Engram        int
	CandidateMask bool
}

// Schedule returns the source-pinned scalar layer schedule.
func Schedule(index int) (Layer, error) {
	if index < 0 || index >= 40 {
		return Layer{}, fmt.Errorf("layer out of range")
	}
	l := Layer{Owner: -1, Engram: -1}
	if index >= 2 {
		l.Ratio = 2
		if index >= 20 {
			l.Ratio = 1
		}
		switch {
		case index < 8:
			l.Owner = 0
		case index < 14:
			l.Owner = 1
		case index < 20:
			l.Owner = 2
		default:
			l.Owner = 3
		}
	}
	l.PublishKV = index == 2 || index == 8 || index == 14 || index == 20
	l.RefreshIndex = l.PublishKV || index == 24 || index == 28 || index == 32 || index == 36
	if index == 1 {
		l.Engram = 0
	}
	if index == 14 {
		l.Engram = 1
	}
	l.CandidateMask = index >= 20
	return l, nil
}
