package nvvmx

import (
	"strings"
	"testing"
)

func TestOptions(t *testing.T) {
	tests := []struct {
		name    string
		opt     Option
		want    string
		wantErr bool
	}{
		{
			name: "WithArch prefixed",
			opt:  WithArch("compute_80"),
			want: "-arch=compute_80",
		},
		{
			name: "WithArch numeric",
			opt:  WithArch("90a"),
			want: "-arch=compute_90a",
		},
		{
			name: "WithOpt 3",
			opt:  WithOpt(3),
			want: "-opt=3",
		},
		{
			name: "WithOpt 0",
			opt:  WithOpt(0),
			want: "-opt=0",
		},
		{
			name:    "WithOpt invalid",
			opt:     WithOpt(2),
			wantErr: true,
		},
		{
			name: "WithGenLTO",
			opt:  WithGenLTO(),
			want: "-gen-lto",
		},
		{
			name: "WithDebug",
			opt:  WithDebug(),
			want: "-g",
		},
		{
			name: "WithFTZ true",
			opt:  WithFTZ(true),
			want: "-ftz=1",
		},
		{
			name: "WithFTZ false",
			opt:  WithFTZ(false),
			want: "-ftz=0",
		},
		{
			name: "WithPrecSqrt true",
			opt:  WithPrecSqrt(true),
			want: "-prec-sqrt=1",
		},
		{
			name: "WithPrecDiv false",
			opt:  WithPrecDiv(false),
			want: "-prec-div=0",
		},
		{
			name: "WithFMA true",
			opt:  WithFMA(true),
			want: "-fma=1",
		},
		{
			name: "WithJumpTableDensity valid",
			opt:  WithJumpTableDensity(50),
			want: "-jump-table-density=50",
		},
		{
			name:    "WithJumpTableDensity invalid",
			opt:     WithJumpTableDensity(150),
			wantErr: true,
		},
		{
			name:    "WithFlag invalid",
			opt:     WithFlag("-invalid_option"),
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var o compileOptions
			err := tc.opt(&o)
			if (err != nil) != tc.wantErr {
				t.Fatalf("opt(&o) err = %v, wantErr %v", err, tc.wantErr)
			}
			if !tc.wantErr {
				if len(o.flags) != 1 || o.flags[0] != tc.want {
					t.Errorf("flags = %v, want [%s]", o.flags, tc.want)
				}
			}
		})
	}
}

func TestCompileWithOptions_NilOption(t *testing.T) {
	_, err := CompileWithOptions(nil, nil)
	if err == nil || !strings.Contains(err.Error(), "nil option") {
		t.Errorf("expected nil option error, got: %v", err)
	}
}

func TestCompileWithOptions_InferenceErrorPropagates(t *testing.T) {
	input := Input{Name: "module", Data: []byte{'B', 'C', 0xc0, 0xde}}
	_, err := CompileWithOptions([]Input{input}, WithAddressSpaceInference())
	if err == nil || !strings.Contains(err.Error(), `infer address spaces for "module"`) {
		t.Fatalf("CompileWithOptions error = %v, want inference failure", err)
	}
}

func TestPrepareInputs_AppliesInferenceWithoutNativeCompiler(t *testing.T) {
	ir := []byte(`@smem = internal addrspace(3) global [4 x i8] undef
define void @kernel() {
entry:
  %slot = alloca ptr, align 8
  store ptr addrspace(3) @smem, ptr %slot, align 8
  ret void
}
`)
	inputs := []Input{{Name: "module", Data: ir}}
	prepared, err := prepareInputs(inputs, true)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(inputs[0].Data), "alloca ptr addrspace(3)") {
		t.Fatal("prepareInputs changed caller input")
	}
	if !strings.Contains(string(prepared[0].Data), "alloca ptr addrspace(3)") {
		t.Fatalf("prepared input did not apply inference:\n%s", prepared[0].Data)
	}
	rewritten, report, err := InferAddressSpaces(ir)
	if err != nil {
		t.Fatal(err)
	}
	assertIndependentSlotTypes(t, rewritten, report)
}
