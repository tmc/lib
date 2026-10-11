package bpfload

import "testing"

func TestValidate(t *testing.T) {
	tests := []struct {
		name  string
		spec  ProgramSpec
		valid bool
	}{
		{"missing type", ProgramSpec{Instructions: make([]byte, 8)}, false},
		{"empty", ProgramSpec{Type: 1}, false},
		{"partial instruction", ProgramSpec{Type: 1, Instructions: make([]byte, 7)}, false},
		{"nul license", ProgramSpec{Type: 1, Instructions: make([]byte, 8), License: "G\x00PL"}, false},
		{"negative log", ProgramSpec{Type: 1, Instructions: make([]byte, 8), LogSize: -1}, false},
		{"large log", ProgramSpec{Type: 1, Instructions: make([]byte, 8), LogSize: 17 << 20}, false},
		{"valid", ProgramSpec{Type: 1, Instructions: make([]byte, 8)}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := validateSpec(tt.spec)
			if (err == nil) != tt.valid {
				t.Fatalf("valid=%v, err=%v", tt.valid, err)
			}
			if tt.valid && (s.License != "GPL" || s.LogSize != 64<<10) {
				t.Fatalf("defaults: %+v", s)
			}
		})
	}
}
func TestCloseZero(t *testing.T) {
	var o Object
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}
}
