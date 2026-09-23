package main

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// A window selects intervals by start time in the source clock domain. An
// interval is selected when lo <= start < hi, so a capture is partitioned by
// adjoining windows without duplicating or dropping a row. Use wholeCapture
// for the unbounded window; the zero value selects nothing.
type window struct{ lo, hi int64 }

func wholeCapture() window { return window{math.MinInt64, math.MaxInt64} }

func (w window) unbounded() bool { return w == wholeCapture() }

func (w window) String() string {
	if w.unbounded() {
		return "whole capture"
	}
	lo, hi := "", ""
	if w.lo != math.MinInt64 {
		lo = strconv.FormatInt(w.lo, 10)
	}
	if w.hi != math.MaxInt64 {
		hi = strconv.FormatInt(w.hi, 10)
	}
	return lo + ":" + hi
}

// Set parses START:END in source-clock nanoseconds. Either side may be empty.
func (w *window) Set(s string) error {
	lo, hi, ok := strings.Cut(s, ":")
	if !ok {
		return errors.New("expected START:END in source-clock nanoseconds")
	}
	*w = wholeCapture()
	var err error
	if lo != "" {
		if w.lo, err = strconv.ParseInt(lo, 10, 64); err != nil {
			return fmt.Errorf("window start: %w", err)
		}
	}
	if hi != "" {
		if w.hi, err = strconv.ParseInt(hi, 10, 64); err != nil {
			return fmt.Errorf("window end: %w", err)
		}
	}
	if w.hi <= w.lo {
		return errors.New("window end must exceed window start")
	}
	return nil
}
