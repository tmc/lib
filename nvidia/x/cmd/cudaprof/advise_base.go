package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/advise"
)

// adviseAgainstBase compares the target capture at path against opts.base and
// ranks report by the comparison.
func adviseAgainstBase(report *advise.Report, target *cudacapture.Capture, path string, opts adviseOptions, limits cudacapture.Limits) error {
	var rules opRules
	if opts.ops != "" {
		f, err := os.Open(opts.ops)
		if err != nil {
			return err
		}
		rules, err = parseOpRules(f)
		f.Close()
		if err != nil {
			return fmt.Errorf("%s: %w", opts.ops, err)
		}
	}
	basePath, err := filepath.Abs(opts.base)
	if err != nil {
		return err
	}
	base, err := readCapturePathLimits(basePath, limits)
	if err != nil {
		return fmt.Errorf("read base capture %s: %w", opts.base, err)
	}
	c, err := compareCaptures(base, target, rules, nil, compareOptions{phase: opts.phase, opsPath: opts.ops})
	if err != nil {
		return fmt.Errorf("compare: %w", err)
	}
	c.Base.Capture, c.Target.Capture = basePath, path
	report.RankAgainst(newAdviseBaseline(c, rules))
	return nil
}

// adviseBaseline is a Comparison as an [advise.Baseline]. It marshals as the
// Comparison.
type adviseBaseline struct {
	c     *Comparison
	rules opRules
	ops   map[string]int64 // each op's exclusive delta
}

func newAdviseBaseline(c *Comparison, rules opRules) *adviseBaseline {
	ops := make(map[string]int64)
	for _, o := range c.Ops {
		ops[o.Op] = o.DeltaExclusiveNanos
	}
	return &adviseBaseline{c: c, rules: rules, ops: ops}
}

func (b *adviseBaseline) String() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "base    %s\n        %s\n", b.c.Base.Capture, b.c.Base.windowLine())
	fmt.Fprintf(&sb, "target  %s\n        %s\n", b.c.Target.Capture, b.c.Target.windowLine())
	for _, n := range b.c.Notes {
		fmt.Fprintf(&sb, "note: %s\n", n)
	}
	sb.WriteString(b.c.identity())
	return sb.String()
}

func (b *adviseBaseline) TermDelta(term string) int64 { return b.c.term(term).Delta }

// OpDelta reports ok false unless the comparison measured ops. A kernel no
// rule names is counted under the unmatched op.
func (b *adviseBaseline) OpDelta(kernel string) (string, int64, bool) {
	if b.c.OpStatus != "MEASURED" {
		return "", 0, false
	}
	op, _ := b.rules.kernelOp(opKey(kernel, false))
	return op, b.ops[op], true
}

func (b *adviseBaseline) MarshalJSON() ([]byte, error) { return json.Marshal(b.c) }
