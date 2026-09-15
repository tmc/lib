package hftokenizer

import (
	"sort"
	"strings"
)

type normalRune struct {
	r     rune
	class uint8
}

// normalizeNFC uses the Unicode 9.0.0 data used by tokenizers' normalization
// dependency. It neither changes with the Go toolchain nor inserts stream-safe
// CGJ characters into long combining sequences.
func normalizeNFC(s string) string {
	ascii := true
	for i := 0; i < len(s); i++ {
		if s[i] >= 0x80 {
			ascii = false
			break
		}
	}
	if ascii {
		return s
	}
	var runes []normalRune
	for _, r := range s {
		runes = decomposeRune(r, runes)
	}
	// Stable ordering is confined to each run of non-starters.
	for start := 0; start < len(runes); {
		if runes[start].class == 0 {
			start++
			continue
		}
		end := start + 1
		for end < len(runes) && runes[end].class != 0 {
			end++
		}
		part := runes[start:end]
		sort.SliceStable(part, func(i, j int) bool { return part[i].class < part[j].class })
		start = end
	}
	out := runes[:0]
	starter := -1
	var previous uint8
	for _, r := range runes {
		if starter >= 0 && (previous == 0 || previous < r.class) {
			if combined := composeRunes(out[starter].r, r.r); combined != 0 {
				out[starter].r = combined
				continue
			}
		}
		if r.class == 0 {
			starter = len(out)
		}
		out = append(out, r)
		previous = r.class
	}
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range out {
		b.WriteRune(r.r)
	}
	return b.String()
}

func decomposeRune(r rune, out []normalRune) []normalRune {
	const sBase, lBase, vBase, tBase = 0xac00, 0x1100, 0x1161, 0x11a7
	const tCount, nCount, sCount = 28, 21 * 28, 19 * 21 * 28
	if i := r - sBase; i >= 0 && i < sCount {
		out = append(out, normalRune{lBase + i/nCount, 0}, normalRune{vBase + (i%nCount)/tCount, 0})
		if t := i % tCount; t != 0 {
			out = append(out, normalRune{tBase + t, 0})
		}
		return out
	}
	if parts, ok := canonicalDecompositions[r]; ok {
		for _, part := range parts {
			out = decomposeRune(part, out)
		}
		return out
	}
	return append(out, normalRune{r, combiningClasses[r]})
}

func composeRunes(a, b rune) rune {
	const sBase, lBase, vBase, tBase = 0xac00, 0x1100, 0x1161, 0x11a7
	const lCount, vCount, tCount = 19, 21, 28
	if l, v := a-lBase, b-vBase; l >= 0 && l < lCount && v >= 0 && v < vCount {
		return sBase + (l*vCount+v)*tCount
	}
	if s, t := a-sBase, b-tBase; s >= 0 && s < lCount*vCount*tCount && s%tCount == 0 && t > 0 && t < tCount {
		return a + t
	}
	return canonicalCompositions[[2]rune{a, b}]
}
