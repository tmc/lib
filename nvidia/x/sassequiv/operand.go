package sassequiv

import (
	"math"
	"regexp"
	"strconv"
	"strings"
)

// splitOperands splits an operand list at top-level commas, dropping
// nvdisasm comments such as (*"SpillRefill"*).
func splitOperands(s string) []string {
	if i := strings.Index(s, "(*"); i >= 0 {
		s = s[:i]
	}
	s = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(s), ";"))
	if s == "" {
		return nil
	}
	var out []string
	depth, start := 0, 0
	for i, r := range s {
		switch r {
		case '[', '(':
			depth++
		case ']', ')':
			depth--
		case ',':
			if depth == 0 {
				out = append(out, strings.TrimSpace(s[start:i]))
				start = i + 1
			}
		}
	}
	return append(out, strings.TrimSpace(s[start:]))
}

var (
	regRE  = regexp.MustCompile(`^(U?R\d+|RZ|URZ|SRZ|U?P\d|U?PT|PR)((?:\.[A-Z0-9_]+)*)$`)
	sregRE = regexp.MustCompile(`^SR_[A-Za-z0-9_.]+$`)
)

// an operand split into its modifiers and its core.
type operand struct {
	prefix string // "-", "!", "~", "|", "-|" ...
	core   string // R5, 0x10, c[0x0][0x210], [R2.64+0x8], ...
	suffix string // ".H1", ".64", ... (".reuse" dropped)
	abs    bool
}

func parseOperand(s string) operand {
	var o operand
	s = strings.TrimSpace(s)
	for len(s) > 0 && strings.ContainsRune("-!~", rune(s[0])) {
		o.prefix += s[:1]
		s = s[1:]
	}
	if strings.HasPrefix(s, "|") && strings.Contains(s[1:], "|") {
		o.abs = true
		end := strings.LastIndex(s, "|")
		s = s[1:end] + s[end+1:]
	}
	s = strings.ReplaceAll(s, ".reuse", "")
	if m := regRE.FindStringSubmatch(s); m != nil {
		o.core, o.suffix = m[1], m[2]
		return o
	}
	o.core = s
	return o
}

// isReg reports whether core names a register, predicate, or zero.
func isReg(core string) bool { return regRE.MatchString(core) }

func isPred(core string) bool {
	return strings.HasPrefix(core, "P") || strings.HasPrefix(core, "UP")
}

// zero names the registers that read as zero.
func zero(core string) bool { return core == "RZ" || core == "URZ" || core == "SRZ" }

// regN returns register core's successor n places on, such as R5 -> R7.
func regN(core string, n int) string {
	if n == 0 || zero(core) {
		return core
	}
	i := strings.IndexFunc(core, func(r rune) bool { return r >= '0' && r <= '9' })
	if i < 0 {
		return core
	}
	k, _ := strconv.Atoi(core[i:])
	return core[:i] + strconv.Itoa(k+n)
}

// immediate returns the bit pattern of an immediate operand: hexadecimal
// as written, and decimal as a float32, which is how SASS prints float
// immediates.
func immediate(s string) (uint64, bool) {
	neg := false
	t := s
	if strings.HasPrefix(t, "-") {
		neg, t = true, t[1:]
	} else if strings.HasPrefix(t, "+") {
		t = t[1:]
	}
	if strings.HasPrefix(t, "0x") {
		v, err := strconv.ParseUint(t[2:], 16, 64)
		if err != nil {
			return 0, false
		}
		if neg {
			v = -v
		}
		return v & 0xffffffff, true
	}
	var f float64
	switch t {
	case "INF":
		f = math.Inf(1)
	case "QNAN":
		if neg {
			return 0xffc00000, true
		}
		return 0x7fc00000, true
	default:
		v, err := strconv.ParseFloat(t, 64)
		if err != nil {
			return 0, false
		}
		f = v
	}
	if neg {
		f = -f
	}
	return uint64(math.Float32bits(float32(f))), true
}

// half returns the fp16 bit pattern of an HFMA2 immediate half.
func half(s string) (uint64, bool) {
	if strings.HasPrefix(s, "0x") {
		v, err := strconv.ParseUint(s[2:], 16, 16)
		return v, err == nil
	}
	b, ok := immediate(s)
	if !ok {
		return 0, false
	}
	return uint64(float16bits(math.Float32frombits(uint32(b)))), true
}

// float16bits rounds f to the nearest IEEE half precision value.
func float16bits(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int(b>>23&0xff) - 127 + 15
	mant := b & 0x7fffff
	switch {
	case b&0x7fffffff == 0:
		return sign
	case int(b>>23&0xff) == 0xff:
		if mant != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	case exp >= 0x1f:
		return sign | 0x7c00
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint(14 - exp)
		h := mant >> shift
		rem := mant & (1<<shift - 1)
		halfway := uint32(1) << (shift - 1)
		if rem > halfway || rem == halfway && h&1 == 1 {
			h++
		}
		return sign | uint16(h)
	}
	h := uint32(exp)<<10 | mant>>13
	rem := mant & 0x1fff
	if rem > 0x1000 || rem == 0x1000 && h&1 == 1 {
		h++
	}
	return sign | uint16(h)
}
