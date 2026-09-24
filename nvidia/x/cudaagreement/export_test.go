package cudaagreement

import (
	"strings"

	"github.com/tmc/lib/nvidia/x/cudamega"
)

// MegaUnits renders how e divides plan: one string per unit, its step names
// joined by commas, with " declined: reason" after a declined step.
func MegaUnits(e *MegaExecutor, plan *Plan, bufs map[string]Buffer) []string {
	var out []string
	for _, u := range e.megaUnits(plan, bufs) {
		var names []string
		for _, i := range u.steps {
			names = append(names, plan.Steps[i].Name)
		}
		s := strings.Join(names, ",")
		if u.declined != "" {
			s += " declined: " + u.declined
		}
		out = append(out, s)
	}
	return out
}

var CheckStartOrder = checkStartOrder

// MegaPrograms returns the persistent program for each run e would launch.
func MegaPrograms(e *MegaExecutor, plan *Plan, bufs map[string]Buffer) ([]*cudamega.Program, error) {
	var out []*cudamega.Program
	for _, u := range e.megaUnits(plan, bufs) {
		if u.declined != "" {
			continue
		}
		p, err := e.program(plan, u)
		if err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, nil
}
