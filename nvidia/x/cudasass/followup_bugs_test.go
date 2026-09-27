package cudasass

import (
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

func TestItem1CommentPanic(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{"bare unclosed comment", "/*/"},
		{"prefix with unclosed comment", "0000000\n/*/"},
		{"instruction line with unclosed comment", ".text.k:\n /*/ NOP ;\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// None of these should panic.
			_, _ = ParseString(tt.input)
			_, _ = ParseLifeRangeCount(strings.NewReader(tt.input))
			_ = NormalizeLine(tt.input)
		})
	}
}

func TestItem2DecreasingAddresses(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool
		errMsg  string
	}{
		{
			name:    "decreasing within single section",
			input:   ".text.k:\n /*0010*/ NOP ;\n /*0000*/ EXIT ;\n",
			wantErr: true,
			errMsg:  "decreasing address",
		},
		{
			name:    "duplicate within single section",
			input:   ".text.k:\n /*0010*/ NOP ;\n /*0010*/ EXIT ;\n",
			wantErr: true,
			errMsg:  "decreasing address",
		},
		{
			name:    "decreasing in second section",
			input:   ".text.k1:\n /*0000*/ NOP ;\n /*0010*/ EXIT ;\n.text.k2:\n /*0020*/ NOP ;\n /*0010*/ EXIT ;\n",
			wantErr: true,
			errMsg:  "decreasing address",
		},
		{
			name:    "strictly increasing control",
			input:   ".text.k:\n /*0000*/ NOP ;\n /*0010*/ EXIT ;\n",
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseString(tt.input)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseString() err = %v, wantErr = %v", err, tt.wantErr)
			}
			if tt.wantErr {
				if !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("error %q does not contain %q", err.Error(), tt.errMsg)
				}
				if !strings.Contains(err.Error(), "k") {
					t.Errorf("error %q does not name section", err.Error())
				}
			}
		})
	}
}

func TestItem3NegativePressure(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool
	}{
		{
			name:    "negative GPR",
			input:   ".text.compute:\n /*0000*/ LDC R1, c[0x0][0x37c] ;  // |  -5   |        |        |         |\n",
			wantErr: true,
		},
		{
			name:    "negative Pred",
			input:   ".text.compute:\n /*0000*/ LDC R1, c[0x0][0x37c] ;  // |   1   |   -1   |        |         |\n",
			wantErr: true,
		},
		{
			name:    "negative UGPR",
			input:   ".text.compute:\n /*0000*/ LDC R1, c[0x0][0x37c] ;  // |   1   |        |   -2   |         |\n",
			wantErr: true,
		},
		{
			name:    "negative UPred",
			input:   ".text.compute:\n /*0000*/ LDC R1, c[0x0][0x37c] ;  // |   1   |        |        |   -3    |\n",
			wantErr: true,
		},
		{
			name:    "non-negative control",
			input:   ".text.compute:\n /*0000*/ LDC R1, c[0x0][0x37c] ;  // |   1   |    2   |    3   |    4    |\n",
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseLifeRangeCount(strings.NewReader(tt.input))
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseLifeRangeCount() err = %v, wantErr = %v", err, tt.wantErr)
			}
		})
	}
}

func TestItem4OrphanedInstructions(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool
	}{
		{
			name:    "instruction before any section header",
			input:   "/*0000*/ NOP ;\n.text.k:\n /*0000*/ EXIT ;\n",
			wantErr: true,
		},
		{
			name:    "instruction after SYMBOLS banner",
			input:   ".text.k:\n /*0000*/ EXIT ;\n//--------------------- SYMBOLS       --------------------------\n/*0010*/ NOP ;\n",
			wantErr: true,
		},
		{
			name:    "instructions inside section control",
			input:   ".text.k:\n /*0000*/ NOP ;\n /*0010*/ EXIT ;\n",
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseString(tt.input)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseString() err = %v, wantErr = %v", err, tt.wantErr)
			}
		})
	}
}

func TestItem5EncodeControlErrorTyping(t *testing.T) {
	tests := []struct {
		name    string
		arch    Arch
		wantErr bool
		wantIso bool
	}{
		{
			name:    "unsupported arch sm_90",
			arch:    "sm_90",
			wantErr: true,
			wantIso: true,
		},
		{
			name:    "unsupported arch empty",
			arch:    "",
			wantErr: true,
			wantIso: true,
		},
		{
			name:    "supported arch SM121a",
			arch:    SM121a,
			wantErr: false,
			wantIso: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := Encoding{Lo: 1, Hi: 1}
			c := ControlCode{Stall: 1}
			_, err := EncodeControl(e, c, tt.arch)
			if (err != nil) != tt.wantErr {
				t.Fatalf("EncodeControl() err = %v, wantErr = %v", err, tt.wantErr)
			}
			if tt.wantIso && !errors.Is(err, cudaerrors.ErrUnsupported) {
				t.Errorf("EncodeControl(%q) err = %v; want errors.Is(err, cudaerrors.ErrUnsupported)", tt.arch, err)
			}
		})
	}
}

func TestItem6MisalignedAddresses(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool
	}{
		{
			name:    "default unknown arch odd address",
			input:   ".text.k:\n /*0001*/ NOP ;\n",
			wantErr: true,
		},
		{
			name:    "default unknown arch 8-byte aligned (sm_70+ requires 16)",
			input:   ".text.k:\n /*0008*/ NOP ;\n",
			wantErr: true,
		},
		{
			name:    "explicit sm_70 misaligned",
			input:   ".target sm_70\n.text.k:\n /*0004*/ NOP ;\n",
			wantErr: true,
		},
		{
			name:    "explicit sm_121a misaligned 8-byte",
			input:   ".target sm_121a\n.text.k:\n /*0008*/ NOP ;\n",
			wantErr: true,
		},
		{
			name:    "older sm_60 arch 8-byte aligned accepted",
			input:   ".target sm_60\n.text.k:\n /*0008*/ NOP ;\n",
			wantErr: false,
		},
		{
			name:    "older sm_60 arch 4-byte misaligned rejected",
			input:   ".target sm_60\n.text.k:\n /*0004*/ NOP ;\n",
			wantErr: true,
		},
		{
			name:    "valid 16-byte aligned control",
			input:   ".text.k:\n /*0000*/ NOP ;\n /*0010*/ EXIT ;\n",
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseString(tt.input)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseString() err = %v, wantErr = %v", err, tt.wantErr)
			}
		})
	}
}
