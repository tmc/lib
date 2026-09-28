package cudalowering

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// TestVerifyNoSharedConversions_Host tests the verification logic using synthetic modules.
func TestVerifyNoSharedConversions_Host(t *testing.T) {
	// Case 1: Kernel without inference has cvta.shared in PTX and conversion in SASS.
	ptxWithoutInference := `
.version 8.0
.target sm_90
.address_size 64

.entry test_kernel_without (
    .param .u64 out
) {
    .reg .b32 %r<4>;
    .reg .b64 %rd<4>;
    .shared .b32 s_data[16];

    mov.u64 %rd1, s_data;
    cvta.shared.u64 %rd2, %rd1;
    ld.u32 %r1, [%rd2];
    ret;
}
`
	sassWithoutInference := `
code for sm_90
Function : test_kernel_without
/*0000*/ CVTA.SHARED R2, R1 ;
/*0010*/ LD.E R3, [R2] ;
/*0020*/ RET ;
`
	mWithout, err := cudaptx.Parse(strings.NewReader(ptxWithoutInference))
	if err != nil {
		t.Fatalf("parse PTX without inference: %v", err)
	}
	dWithout, err := cudasass.Parse(strings.NewReader(sassWithoutInference))
	if err != nil {
		t.Fatalf("parse SASS without inference: %v", err)
	}

	reportWithout, err := VerifyNoSharedConversions(mWithout, dWithout, "%slot")
	if err != nil {
		t.Fatalf("VerifyNoSharedConversions: %v", err)
	}
	if len(reportWithout.Divergences) == 0 {
		t.Fatal("expected SharedConversion divergence without inference, got none")
	}
	foundSharedConv := false
	for _, d := range reportWithout.Divergences {
		if d.Kind == SharedConversion {
			foundSharedConv = true
			if !strings.Contains(d.Detail, "inferred slots: %slot") {
				t.Errorf("slot label missing from conversion report: %+v", d)
			}
			if d.PTXCount == 0 && d.SASSCount == 0 {
				t.Errorf("expected non-zero conversion count, got %+v", d)
			}
		}
	}
	if !foundSharedConv {
		t.Errorf("missing SharedConversion divergence in %+v", reportWithout.Divergences)
	}
	if reportWithout.Err() == nil {
		t.Error("expected non-nil Err() when divergences exist")
	}

	// Case 2: Kernel with inference has direct ld.shared and no conversions.
	ptxWithInference := `
.version 8.0
.target sm_90
.address_size 64

.entry test_kernel_with (
    .param .u64 out
) {
    .reg .b32 %r<4>;
    .shared .b32 s_data[16];

    ld.shared.u32 %r1, [s_data];
    ret;
}
`
	sassWithInference := `
code for sm_90
Function : test_kernel_with
/*0000*/ LDS R1, [R0] ;
/*0010*/ RET ;
`
	mWith, err := cudaptx.Parse(strings.NewReader(ptxWithInference))
	if err != nil {
		t.Fatalf("parse PTX with inference: %v", err)
	}
	dWith, err := cudasass.Parse(strings.NewReader(sassWithInference))
	if err != nil {
		t.Fatalf("parse SASS with inference: %v", err)
	}

	reportWith, err := VerifyNoSharedConversions(mWith, dWith, "%slot")
	if err != nil {
		t.Fatalf("VerifyNoSharedConversions: %v", err)
	}
	for _, d := range reportWith.Divergences {
		if d.Kind == SharedConversion {
			t.Errorf("unexpected SharedConversion divergence with inference: %v", d)
		}
	}
	if reportWith.Err() != nil {
		t.Errorf("unexpected error for inferred kernel: %v", reportWith.Err())
	}
}

// TestLowering_NoSharedConversionAfterInference tests that after address-space inference,
// emitted SASS contains no shared-space address conversions for the inferred slots.
// Gated with CUDA_NVVMX_NATIVE=1.

// TestLowering_ConversionPresentWithoutInference tests that without address-space inference,
// the same kernel DOES contain shared-space address conversions in SASS.
// Gated with CUDA_NVVMX_NATIVE=1.
