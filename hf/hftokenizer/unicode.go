package hftokenizer

import "sort"

//go:generate python3 internal/genunicode/generate.py
//go:generate gofmt -w unicode_tables.go

type runeRange struct{ lo, hi rune }

func inRanges(r rune, ranges []runeRange) bool {
	i := sort.Search(len(ranges), func(i int) bool { return ranges[i].hi >= r })
	return i < len(ranges) && ranges[i].lo <= r
}
func isLetter(r rune) bool { return inRanges(r, letterRanges[:]) }
func isNumber(r rune) bool { return inRanges(r, numberRanges[:]) }
func isSpace(r rune) bool {
	return r >= 0x9 && r <= 0xd || r == 0x20 || r == 0x85 || r == 0xa0 || r == 0x1680 || r >= 0x2000 && r <= 0x200a || r == 0x2028 || r == 0x2029 || r == 0x202f || r == 0x205f || r == 0x3000
}
