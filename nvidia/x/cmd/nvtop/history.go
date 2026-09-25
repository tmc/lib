package main

// historySize is how many samples a series keeps: more than any plot
// is wide.
const historySize = 4096

// A series is a ring of the most recent samples of one plotted value.
type series struct {
	buf   [historySize]uint32
	start int // index of the oldest sample
	n     int
}

// push adds v as the newest sample, dropping the oldest if full.
func (s *series) push(v uint32) {
	if s.n < len(s.buf) {
		s.buf[(s.start+s.n)%len(s.buf)] = v
		s.n++
		return
	}
	s.buf[s.start] = v
	s.start = (s.start + 1) % len(s.buf)
}

// len returns the number of samples stored.
func (s *series) len() int { return s.n }

// recent returns the i'th most recent sample; recent(0) is the newest.
func (s *series) recent(i int) uint32 {
	return s.buf[(s.start+s.n-1-i)%len(s.buf)]
}
