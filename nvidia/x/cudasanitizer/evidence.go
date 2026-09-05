package cudasanitizer

import "strings"

// outputEvidence matches configured substrings without retaining the full
// stream. The tail spans writes; the head and tail also preserve the existing
// stdout + newline + stderr matching convention after both streams finish.
type outputEvidence struct {
	found      map[string]bool
	head, tail string
	limit      int
}

func newOutputEvidence(cfg Config) outputEvidence {
	e := outputEvidence{found: make(map[string]bool)}
	for _, list := range [][]string{cfg.RequireOutput, cfg.ForbidOutput} {
		for _, s := range list {
			if s == "" {
				continue
			}
			e.found[s] = false
			e.limit = max(e.limit, len(s)-1)
		}
	}
	return e
}

func (e *outputEvidence) write(p []byte) {
	if len(e.found) == 0 {
		return
	}
	text := e.tail + string(p)
	for s, found := range e.found {
		if !found && strings.Contains(text, s) {
			e.found[s] = true
		}
	}
	if n := min(e.limit-len(e.head), len(p)); n > 0 {
		e.head += string(p[:n])
	}
	n := min(e.limit, len(text))
	e.tail = strings.Clone(text[len(text)-n:])
}

func (r *Result) checkEvidence(stdout, stderr *capBuffer) {
	boundary := stdout.evidence.tail + "\n" + stderr.evidence.head
	found := func(s string) bool {
		return stdout.evidence.found[s] || stderr.evidence.found[s] || strings.Contains(boundary, s)
	}
	for _, want := range r.Config.RequireOutput {
		if want != "" && !found(want) {
			r.MissingEvidence = append(r.MissingEvidence, want)
		}
	}
	for _, bad := range r.Config.ForbidOutput {
		if bad != "" && found(bad) {
			r.ForbiddenEvidence = append(r.ForbiddenEvidence, bad)
		}
	}
}
