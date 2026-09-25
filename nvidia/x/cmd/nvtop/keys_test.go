package main

import (
	"slices"
	"testing"
)

func TestParseKeys(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want []key
	}{
		{"letters", "qjk", []key{'q', 'j', 'k'}},
		{"enter", "\r\n", []key{keyEnter, keyEnter}},
		{"ctrl-l", "\x0c", []key{keyCtrlL}},
		{"escape alone", "\x1b", []key{keyEsc}},
		{"csi arrows", "\x1b[A\x1b[B\x1b[C\x1b[D", []key{keyUp, keyDown, keyRight, keyLeft}},
		{"ss3 arrows", "\x1bOA\x1bOD", []key{keyUp, keyLeft}},
		{"ss3 function keys", "\x1bOP\x1bOS", []key{keyF1, keyF4}},
		{"ss3 keypad enter", "\x1bOM", []key{keyEnter}},
		{"tilde function keys", "\x1b[15~\x1b[17~\x1b[20~\x1b[21~\x1b[24~", []key{keyF5, keyF6, keyF9, keyF10, keyF12}},
		{"linux console", "\x1b[[A\x1b[[E", []key{keyF1, keyF5}},
		{"csi f1", "\x1b[1P", []key{keyF1}},
		{"modified f1 ignored", "\x1b[1;5P", nil},
		{"unknown tilde ignored", "\x1b[99~q", []key{'q'}},
		{"alt chord ignored", "\x1bxq", []key{'q'}},
		{"truncated csi", "\x1b[1", nil},
		{"utf-8", "é+", []key{'é', '+'}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := parseKeys([]byte(tt.in)); !slices.Equal(got, tt.want) {
				t.Errorf("parseKeys(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}
