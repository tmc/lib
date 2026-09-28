package cudalowering

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

func TestCountSharedConversions(t *testing.T) {
	fn := &cudasass.Function{Instructions: []cudasass.Instruction{
		{Opcode: "CVTA.SHARED"},
		{Opcode: "CVTA.SHARED.U64"},
		{Opcode: "CVTA.GLOBAL"},
		{Opcode: "LD.E"},
		{Opcode: "LDG.E"},
		{Opcode: "LDS"},
		{Opcode: "ST.E"},
		{Opcode: "STG.E"},
		{Opcode: "STS"},
	}}
	if got := countSharedConversions(fn); got != 2 {
		t.Fatalf("countSharedConversions = %d, want 2", got)
	}
	if _, err := VerifyNoSharedConversions(nil, nil); err == nil {
		t.Fatal("VerifyNoSharedConversions accepted an empty slot list")
	}
}

func TestVerifyNoSharedConversions_RejectsMismatchedPair(t *testing.T) {
	ptx, err := cudaptx.Parse(strings.NewReader(`.version 8.0
.target sm_90
.address_size 64
.entry matched() { ret; }
`))
	if err != nil {
		t.Fatal(err)
	}
	sass, err := cudasass.Parse(strings.NewReader(`code for sm_90
Function : unrelated
/*0000*/ RET ;
`))
	if err != nil {
		t.Fatal(err)
	}
	report, err := VerifyNoSharedConversions(ptx, sass, "%slot")
	if err != nil {
		t.Fatal(err)
	}
	if err := report.Err(); err == nil || !strings.Contains(err.Error(), "missing sass") {
		t.Fatalf("mismatched pair error = %v, want missing sass", err)
	}
}

func TestVerifyNoSharedConversions_GlobalTrafficIsNotConversion(t *testing.T) {
	ptx, err := cudaptx.Parse(strings.NewReader(`.version 8.0
.target sm_90
.address_size 64
.entry global_only() {
  .reg .b32 %r<2>;
  ld.global.u32 %r1, [%r0];
  st.global.u32 [%r0], %r1;
  ret;
}`))
	if err != nil {
		t.Fatal(err)
	}
	sass, err := cudasass.Parse(strings.NewReader(`code for sm_90
Function : global_only
/*0000*/ LDG.E R1, [R0] ;
/*0010*/ STG.E [R0], R1 ;
/*0020*/ RET ;
`))
	if err != nil {
		t.Fatal(err)
	}
	report, err := VerifyNoSharedConversions(ptx, sass, "%slot")
	if err != nil {
		t.Fatal(err)
	}
	if report.Err() != nil {
		t.Fatalf("global traffic reported as a shared conversion: %v", report.Err())
	}
}

func TestVerifyNoSharedConversionsPTXOnly(t *testing.T) {
	for _, opcode := range []string{"cvta.shared.u64", "cvta.to.shared.u64"} {
		t.Run(opcode, func(t *testing.T) {
			ptx, err := cudaptx.Parse(strings.NewReader(`.version 8.0
.target sm_90
.address_size 64
.entry ptx_only() {
  .reg .b64 %rd<2>;
  ` + opcode + ` %rd1, %rd0;
  ret;
}`))
			if err != nil {
				t.Fatal(err)
			}
			sass, err := cudasass.Parse(strings.NewReader(`code for sm_90
Function : ptx_only
/*0000*/ RET ;
`))
			if err != nil {
				t.Fatal(err)
			}
			report, err := VerifyNoSharedConversions(ptx, sass, "@ptx_only/%slot")
			if err != nil {
				t.Fatal(err)
			}
			for _, div := range report.Divergences {
				if div.Kind == SharedConversion {
					if div.PTXCount == 0 || div.SASSCount != 0 {
						t.Fatalf("PTX-only conversion counts = PTX %d SASS %d", div.PTXCount, div.SASSCount)
					}
					return
				}
			}
			t.Fatal("PTX-only shared conversion was not reported")
		})
	}
}
