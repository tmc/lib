package nvidiasmi

import (
	"strconv"
	"strings"
)

// A Value is the text of one leaf element, such as "45 C", "1024 MiB",
// "Enabled", or "N/A". The zero Value means the element was absent.
//
// Its methods ignore surrounding white space.
type Value string

// placeholders lists the texts nvidia-smi prints in place of a reading,
// compared case-insensitively after removing enclosing brackets.
var placeholders = []string{
	"n/a",
	"not supported",
	"not available",
	"not found",
	"unknown error",
	"insufficient permissions",
	"requested functionality has been deprecated",
	"gpu requires reset",
	"gpu is lost",
}

// String returns v with surrounding white space removed.
func (v Value) String() string {
	return strings.TrimSpace(string(v))
}

// Available reports whether v holds a reading. It is false for an absent
// element and for placeholders such as "N/A", "[N/A]", "Not Supported",
// "[Not Supported]", and "Requested functionality has been deprecated".
func (v Value) Available() bool {
	s := v.String()
	if s == "" {
		return false
	}
	if len(s) >= 2 && s[0] == '[' && s[len(s)-1] == ']' {
		s = strings.TrimSpace(s[1 : len(s)-1])
	}
	for _, p := range placeholders {
		if strings.EqualFold(s, p) {
			return false
		}
	}
	return true
}

// split returns the leading decimal number of v and the text after it.
// It reports ok=false if v is unavailable or does not start with a number.
func (v Value) split() (num, unit string, ok bool) {
	if !v.Available() {
		return "", "", false
	}
	s := v.String()
	i := 0
	if i < len(s) && (s[i] == '-' || s[i] == '+') {
		i++
	}
	digits := 0
	for ; i < len(s); i++ {
		c := s[i]
		if c >= '0' && c <= '9' {
			digits++
			continue
		}
		if c != '.' {
			break
		}
	}
	if digits == 0 {
		return "", "", false
	}
	return s[:i], strings.TrimSpace(s[i:]), true
}

// Float returns the number at the start of v, ignoring any unit:
// 70.12 for "70.12 W". It reports ok=false if v is unavailable or does
// not start with a number.
func (v Value) Float() (f float64, ok bool) {
	num, _, ok := v.split()
	if !ok {
		return 0, false
	}
	f, err := strconv.ParseFloat(num, 64)
	return f, err == nil
}

// Int returns the integer at the start of v, ignoring any unit: 45 for
// "45 C" and 16 for "16x". A whole value with a 0x prefix, such as
// "0x100", is read as hexadecimal. It reports ok=false if v is
// unavailable or does not start with an integer.
func (v Value) Int() (n int64, ok bool) {
	if s := v.String(); len(s) > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X') {
		n, err := strconv.ParseInt(s[2:], 16, 64)
		return n, err == nil && v.Available()
	}
	num, _, ok := v.split()
	if !ok {
		return 0, false
	}
	n, err := strconv.ParseInt(num, 10, 64)
	return n, err == nil
}

// Unit returns the text after the leading number of v: "W" for
// "70.12 W" and "%" for "35 %". It returns "" if v has no leading number.
func (v Value) Unit() string {
	_, unit, _ := v.split()
	return unit
}

// Bytes returns v as a number of bytes. The unit must be one nvidia-smi
// uses for sizes: B, KiB, MiB, GiB, or TiB. It reports ok=false for
// any other unit, so "4000 KB/s" is not a size.
func (v Value) Bytes() (n int64, ok bool) {
	num, unit, ok := v.split()
	if !ok {
		return 0, false
	}
	var shift uint
	switch unit {
	case "B":
		shift = 0
	case "KiB":
		shift = 10
	case "MiB":
		shift = 20
	case "GiB":
		shift = 30
	case "TiB":
		shift = 40
	default:
		return 0, false
	}
	n, err := strconv.ParseInt(num, 10, 64)
	if err != nil || n < 0 || n > (1<<63-1)>>shift {
		return 0, false
	}
	return n << shift, true
}

// Bool interprets the state words nvidia-smi prints: "Enabled", "Yes",
// "Active", and "True" are true; "Disabled", "No", "Not Active", and
// "False" are false. Case is ignored. It reports ok=false for anything
// else.
func (v Value) Bool() (b, ok bool) {
	switch strings.ToLower(v.String()) {
	case "enabled", "yes", "active", "true":
		return true, true
	case "disabled", "no", "not active", "false":
		return false, true
	}
	return false, false
}

// maxListLen bounds the result of IntList.
const maxListLen = 1 << 16

// IntList parses a list of non-negative integers and inclusive ranges,
// such as "0-19,40-59", as nvidia-smi topo -m prints CPU and NUMA
// affinity. It reports ok=false for placeholders like "N/A", for
// malformed lists, and for lists of more than 65536 integers.
func (v Value) IntList() (list []int, ok bool) {
	if !v.Available() {
		return nil, false
	}
	for _, f := range strings.Split(v.String(), ",") {
		lo, hi, isRange := strings.Cut(f, "-")
		a, err := strconv.Atoi(lo)
		if err != nil || a < 0 || lo[0] == '+' {
			return nil, false
		}
		b := a
		if isRange {
			b, err = strconv.Atoi(hi)
			if err != nil || b < a || hi[0] == '+' {
				return nil, false
			}
		}
		if b-a >= maxListLen-len(list) {
			return nil, false
		}
		for i := 0; i <= b-a; i++ {
			list = append(list, a+i)
		}
	}
	return list, true
}
