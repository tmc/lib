package main

import "unicode/utf8"

// A key is a key press: a character, or one of the special keys below.
type key rune

// Special keys live in the Unicode private use area.
const (
	keyUp key = 0xE000 + iota
	keyDown
	keyLeft
	keyRight
	keyEnter
	keyEsc
	keyF1
	keyF2
	keyF3
	keyF4
	keyF5
	keyF6
	keyF7
	keyF8
	keyF9
	keyF10
	keyF11
	keyF12
	keyCtrlL
	keyUnknown
)

// csiTilde maps the number in an ESC [ n ~ sequence to its key.
var csiTilde = map[string]key{
	"11": keyF1, "12": keyF2, "13": keyF3, "14": keyF4,
	"15": keyF5, "17": keyF6, "18": keyF7, "19": keyF8,
	"20": keyF9, "21": keyF10, "23": keyF11, "24": keyF12,
}

// parseKeys splits b, one read from a terminal, into keys. An ESC that
// ends the read is the Escape key; an ESC followed by a character that
// starts no known sequence is an Alt chord, which nvtop ignores.
func parseKeys(b []byte) []key {
	var keys []key
	for len(b) > 0 {
		k, n := parseKey(b)
		b = b[n:]
		if k != keyUnknown {
			keys = append(keys, k)
		}
	}
	return keys
}

func parseKey(b []byte) (key, int) {
	switch c := b[0]; c {
	case '\r', '\n':
		return keyEnter, 1
	case 0x0c:
		return keyCtrlL, 1
	case 0x1b:
	default:
		r, n := decodeRune(b)
		return key(r), n
	}
	if len(b) == 1 {
		return keyEsc, 1
	}
	switch b[1] {
	case 'O': // SS3: xterm F1-F4 and application-mode keys
		if len(b) < 3 {
			return keyUnknown, 2
		}
		switch b[2] {
		case 'A':
			return keyUp, 3
		case 'B':
			return keyDown, 3
		case 'C':
			return keyRight, 3
		case 'D':
			return keyLeft, 3
		case 'M':
			return keyEnter, 3
		case 'P':
			return keyF1, 3
		case 'Q':
			return keyF2, 3
		case 'R':
			return keyF3, 3
		case 'S':
			return keyF4, 3
		}
		return keyUnknown, 3
	case '[':
		return parseCSI(b)
	}
	// Alt chord: skip the ESC and the character.
	_, n := decodeRune(b[1:])
	return keyUnknown, 1 + n
}

// parseCSI parses an ESC [ sequence: parameter bytes then a final byte.
func parseCSI(b []byte) (key, int) {
	i := 2
	// The Linux console sends ESC [ [ A through ESC [ [ E for F1-F5.
	if i < len(b) && b[i] == '[' {
		if i+1 < len(b) && b[i+1] >= 'A' && b[i+1] <= 'E' {
			return keyF1 + key(b[i+1]-'A'), i + 2
		}
		return keyUnknown, min(i+2, len(b))
	}
	start := i
	for i < len(b) && b[i] >= 0x30 && b[i] <= 0x3f {
		i++
	}
	if i >= len(b) {
		return keyUnknown, len(b)
	}
	params, final := string(b[start:i]), b[i]
	n := i + 1
	switch final {
	case 'A':
		return keyUp, n
	case 'B':
		return keyDown, n
	case 'C':
		return keyRight, n
	case 'D':
		return keyLeft, n
	case '~':
		if k, ok := csiTilde[params]; ok {
			return k, n
		}
	case 'P', 'Q', 'R', 'S':
		// ESC [ 1 ; m P: F1-F4 with modifiers. Without modifiers
		// only; nvtop does not bind modified keys.
		if params == "" || params == "1" {
			return keyF1 + key(final-'P'), n
		}
	}
	return keyUnknown, n
}

func decodeRune(b []byte) (rune, int) { return utf8.DecodeRune(b) }
