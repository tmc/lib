package dsv4expertcache

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
)

// Budget specifies either a slot count or a byte limit, never both.
type Budget struct {
	Slots uint32
	Bytes uint64
}

// ParseBudget accepts a positive decimal slot count or a decimal GiB limit
// with a case-insensitive GB suffix, matching ds4's streaming-cache argument.
func ParseBudget(s string) (Budget, error) {
	bytes := len(s) > 2 && (s[len(s)-2] == 'g' || s[len(s)-2] == 'G') && (s[len(s)-1] == 'b' || s[len(s)-1] == 'B')
	n := s
	if bytes {
		n = s[:len(s)-2]
	}
	if n == "" {
		return Budget{}, errors.New("empty cache budget")
	}
	for i := range n {
		if n[i] < '0' || n[i] > '9' {
			return Budget{}, errors.New("invalid cache budget")
		}
	}
	v, err := strconv.ParseUint(n, 10, 64)
	if err != nil || v == 0 {
		return Budget{}, errors.New("invalid cache budget")
	}
	if bytes {
		if len(n) >= 32 || v > math.MaxUint64/(1<<30) {
			return Budget{}, errors.New("cache byte budget overflow")
		}
		return Budget{Bytes: v << 30}, nil
	}
	if v > math.MaxUint32 {
		return Budget{}, errors.New("cache slot budget overflow")
	}
	return Budget{Slots: uint32(v)}, nil
}

// SlotsForBytes returns zero when the byte limit cannot hold an expert or
// would produce a count outside the upstream uint32 range.
func SlotsForBytes(bytes, expertBytes uint64) uint32 {
	if expertBytes == 0 {
		return 0
	}
	n := bytes / expertBytes
	if n > math.MaxUint32 {
		return 0
	}
	return uint32(n)
}

// HotEntry is one unique expert and its saturated hit-count priority.
type HotEntry struct{ Layer, Expert, Priority uint32 }

// ReadHotlist reads ds4's three unsigned columns. Remaining text, including
// the optional weight, is ignored. Zero hits, duplicates and out-of-range
// entries are skipped. It stops after limit usable entries. The reader uses
// the same 255-byte fgets chunks as upstream, not an unlimited line grammar.
func ReadHotlist(r io.Reader, layers, experts, limit uint32) ([]HotEntry, error) {
	if r == nil || layers == 0 || experts == 0 || limit == 0 {
		return nil, errors.New("invalid hotlist dimensions")
	}
	b := bufio.NewReader(r)
	seen := make(map[[2]uint32]bool)
	var entries []HotEntry
	for line := 1; uint32(len(entries)) < limit; line++ {
		var chunk []byte
		for len(chunk) < 255 {
			c, err := b.ReadByte()
			if err != nil {
				if err != io.EOF {
					return nil, fmt.Errorf("read hotlist: %w", err)
				}
				break
			}
			chunk = append(chunk, c)
			if c == '\n' {
				break
			}
		}
		if len(chunk) == 0 {
			break
		}
		p := skipSpace(string(chunk))
		if p == "" || p[0] == '#' {
			continue
		}
		var values [3]uint64
		for i := range values {
			v, rest, ok := unsignedPrefix(p)
			if !ok {
				return nil, fmt.Errorf("invalid hotlist line %d", line)
			}
			values[i], p = v, skipSpace(rest)
		}
		if values[2] == 0 {
			continue
		}
		key := [2]uint32{uint32(values[0]), uint32(values[1])}
		if key[0] >= layers || key[1] >= experts || seen[key] {
			continue
		}
		priority := values[2]
		if priority > math.MaxUint32 {
			priority = math.MaxUint32
		}
		seen[key] = true
		entries = append(entries, HotEntry{key[0], key[1], uint32(priority)})
	}
	return entries, nil
}

func skipSpace(s string) string {
	for len(s) > 0 && (s[0] == ' ' || s[0] == '\t' || s[0] == '\n' || s[0] == '\r' || s[0] == '\v' || s[0] == '\f') {
		s = s[1:]
	}
	return s
}

// strtoul permits a sign and consumes only the numeric prefix. Both supported
// oracle hosts have 64-bit unsigned long; the layer/expert cast is uint32.
func unsignedPrefix(s string) (uint64, string, bool) {
	i := 0
	negative := false
	if len(s) > 0 && (s[0] == '+' || s[0] == '-') {
		negative = s[0] == '-'
		i++
	}
	start := i
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	if i == start {
		return 0, s, false
	}
	v, err := strconv.ParseUint(s[start:i], 10, 64)
	if err != nil {
		return 0, s, false
	}
	if negative {
		v = -v
	}
	return v, s[i:], true
}
