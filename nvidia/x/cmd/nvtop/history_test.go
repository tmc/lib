package main

import "testing"

func TestSeries(t *testing.T) {
	var s series
	for i := range historySize + 10 {
		s.push(uint32(i))
	}
	if s.len() != historySize {
		t.Fatalf("len = %d, want %d", s.len(), historySize)
	}
	if got, want := s.recent(0), uint32(historySize+9); got != want {
		t.Errorf("recent(0) = %d, want %d", got, want)
	}
	if got, want := s.recent(historySize-1), uint32(10); got != want {
		t.Errorf("recent(%d) = %d, want %d", historySize-1, got, want)
	}

	var short series
	short.push(7)
	short.push(8)
	if short.len() != 2 || short.recent(0) != 8 || short.recent(1) != 7 {
		t.Errorf("short series: len %d, recent %d %d", short.len(), short.recent(0), short.recent(1))
	}
}
