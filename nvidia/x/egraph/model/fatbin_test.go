package model

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

func TestClassifyComputeCapability(t *testing.T) {
	tests := []struct {
		name             string
		major, minor     int
		wantErr          bool
		wantTarget       string
		wantFamilyTarget string
		wantFamily       ArchFamily
		wantWGMMA        bool
		wantTCGen05      bool
		wantTMEM         bool
		wantCLC          bool
		wantScaleVec4X   bool
		wantWarpBlockMMA bool
		wantMMAInstrSub  string
	}{
		{
			name:             "hopper_sm90a",
			major:            9,
			minor:            0,
			wantTarget:       "sm_90a",
			wantFamilyTarget: "sm_90a",
			wantFamily:       FamilyHopperSM90,
			wantWGMMA:        true,
			wantMMAInstrSub:  "wgmma.mma_async",
		},
		{
			name:             "blackwell_dc_sm100a",
			major:            10,
			minor:            0,
			wantTarget:       "sm_100a",
			wantFamilyTarget: "sm_100f",
			wantFamily:       FamilyBlackwellDC_SM100,
			wantTCGen05:      true,
			wantTMEM:         true,
			wantCLC:          true,
			wantMMAInstrSub:  "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale",
		},
		{
			name:             "blackwell_ultra_sm103a",
			major:            10,
			minor:            3,
			wantTarget:       "sm_103a",
			wantFamilyTarget: "sm_103f",
			wantFamily:       FamilyBlackwellUltra_SM103,
			wantTCGen05:      true,
			wantTMEM:         true,
			wantCLC:          true,
			wantScaleVec4X:   true,
			wantMMAInstrSub:  "scale_vec::4X",
		},
		{
			name:             "rubin_sm107a",
			major:            10,
			minor:            7,
			wantTarget:       "sm_107a",
			wantFamilyTarget: "sm_107f",
			wantFamily:       FamilyRubin_SM107,
			wantTCGen05:      true,
			wantTMEM:         true,
			wantCLC:          true,
			wantScaleVec4X:   true,
			wantMMAInstrSub:  "scale_vec::4X",
		},
		{
			name:             "blackwell_edge_sm120a",
			major:            12,
			minor:            0,
			wantTarget:       "sm_120a",
			wantFamilyTarget: "sm_120f",
			wantFamily:       FamilyBlackwellEdge_SM120,
			wantCLC:          true,
			wantWarpBlockMMA: true,
			wantMMAInstrSub:  "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale",
		},
		{
			name:             "blackwell_spark_sm121a",
			major:            12,
			minor:            1,
			wantTarget:       "sm_121a",
			wantFamilyTarget: "sm_120f",
			wantFamily:       FamilyBlackwellEdge_SM120,
			wantCLC:          true,
			wantWarpBlockMMA: true,
			wantMMAInstrSub:  "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale",
		},
		{
			name:    "unsupported_sm86",
			major:   8,
			minor:   6,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spec, err := ClassifyComputeCapability(tt.major, tt.minor)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("ClassifyComputeCapability(%d, %d) succeeded, want error", tt.major, tt.minor)
				}
				return
			}
			if err != nil {
				t.Fatalf("ClassifyComputeCapability(%d, %d): %v", tt.major, tt.minor, err)
			}
			if spec.Target != tt.wantTarget {
				t.Errorf("Target = %q, want %q", spec.Target, tt.wantTarget)
			}
			if spec.FamilyTarget != tt.wantFamilyTarget {
				t.Errorf("FamilyTarget = %q, want %q", spec.FamilyTarget, tt.wantFamilyTarget)
			}
			if spec.Family != tt.wantFamily {
				t.Errorf("Family = %q, want %q", spec.Family, tt.wantFamily)
			}
			if spec.HasWGMMA != tt.wantWGMMA {
				t.Errorf("HasWGMMA = %v, want %v", spec.HasWGMMA, tt.wantWGMMA)
			}
			if spec.HasTCGen05 != tt.wantTCGen05 {
				t.Errorf("HasTCGen05 = %v, want %v", spec.HasTCGen05, tt.wantTCGen05)
			}
			if spec.HasTMEM != tt.wantTMEM {
				t.Errorf("HasTMEM = %v, want %v", spec.HasTMEM, tt.wantTMEM)
			}
			if spec.HasCLC != tt.wantCLC {
				t.Errorf("HasCLC = %v, want %v", spec.HasCLC, tt.wantCLC)
			}
			if spec.HasScaleVec4X != tt.wantScaleVec4X {
				t.Errorf("HasScaleVec4X = %v, want %v", spec.HasScaleVec4X, tt.wantScaleVec4X)
			}
			if spec.HasWarpBlockScaledMMA != tt.wantWarpBlockMMA {
				t.Errorf("HasWarpBlockScaledMMA = %v, want %v", spec.HasWarpBlockScaledMMA, tt.wantWarpBlockMMA)
			}
			if !strings.Contains(spec.MMAInstruction, tt.wantMMAInstrSub) {
				t.Errorf("MMAInstruction = %q, want substring %q", spec.MMAInstruction, tt.wantMMAInstrSub)
			}

			lookedUp, err := LookupArchTarget(spec.Target)
			if err != nil {
				t.Fatalf("LookupArchTarget(%q): %v", spec.Target, err)
			}
			if lookedUp != spec {
				t.Errorf("LookupArchTarget(%q) = %+v, want %+v", spec.Target, lookedUp, spec)
			}
		})
	}
}

func packFloatsLE(vals []float32) []byte {
	buf := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(v))
	}
	return buf
}

func unpackFloatsLE(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func packIntsLE(vals []int32) []byte {
	buf := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(buf[4*i:], uint32(v))
	}
	return buf
}

func unpackIntsLE(b []byte) []int32 {
	out := make([]int32, len(b)/4)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func TestBuildMultiArchBundleAndDispatch(t *testing.T) {
	cfg := ArchConfig{
		HiddenSize:        512,
		IntermediateSize:  1024,
		NumHiddenLayers:   4,
		NumAttentionHeads: 8,
		NumKeyValueHeads:  2,
		HeadDim:           64,
		RMSNormEps:        1e-5,
		WeightDType:       NVFP4,
		KVDType:           E4M3,
	}
	targets := []string{"sm_90a", "sm_100a", "sm_103a", "sm_107a", "sm_120a", "sm_121a"}
	bundle, err := BuildMultiArchBundle("decoder_layer", cfg, targets)
	if err != nil {
		t.Fatalf("BuildMultiArchBundle: %v", err)
	}
	if len(bundle.Units) != len(targets) {
		t.Fatalf("len(Units) = %d, want %d", len(bundle.Units), len(targets))
	}

	flags := bundle.NVCCFlags()
	var flatFlags []string
	for _, pair := range flags {
		flatFlags = append(flatFlags, strings.Join(pair, " "))
	}
	joinedFlags := strings.Join(flatFlags, " ")
	for _, wantFlag := range []string{
		"arch=compute_90a,code=sm_90a",
		"arch=compute_100a,code=sm_100a",
		"arch=compute_100f,code=sm_100f",
		"arch=compute_103a,code=sm_103a",
		"arch=compute_107a,code=sm_107a",
		"arch=compute_107f,code=sm_107f",
		"arch=compute_120a,code=sm_120a",
		"arch=compute_120f,code=sm_120f",
		"arch=compute_121a,code=sm_121a",
	} {
		if !strings.Contains(joinedFlags, wantFlag) {
			t.Errorf("NVCCFlags missing %q in %q", wantFlag, joinedFlags)
		}
	}

	if ArchSM107A != "sm_107a" || ArchSM107F != "sm_107f" {
		t.Fatalf("ArchSM107A=%q ArchSM107F=%q", ArchSM107A, ArchSM107F)
	}
	if spec107f, err := LookupArchTarget(string(ArchSM107F)); err != nil || spec107f.Target != "sm_107f" || spec107f.FamilyTarget != "sm_107f" {
		t.Fatalf("LookupArchTarget(sm_107f) = %+v, %v", spec107f, err)
	}

	for _, marker := range []string{
		"wgmma.mma_async",
		"tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale",
		"tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X",
		"mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale",
		"cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev)",
		"cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev)",
		"cudaErrorNoKernelImageForDevice",
	} {
		if !strings.Contains(bundle.DispatcherCUDA, marker) {
			t.Errorf("DispatcherCUDA missing marker %q", marker)
		}
	}

	if _, err := cudahostemu.Compiler(); err != nil {
		t.Skipf("skipping host emulation run: %v", err)
	}

	const cols = 64
	const eps = 1e-5
	x := make([]float32, cols)
	w := make([]float32, cols)
	normW := make([]float32, cols)
	var sumSq float64
	for i := range cols {
		x[i] = float32((i%7)-3) * 0.25
		w[i] = float32((i%5)+1) * 0.5
		normW[i] = 1.0 + float32(i%4)*0.125
		sumSq += float64(x[i] * x[i])
	}
	invRMS := float32(1.0 / math.Sqrt(sumSq/float64(cols)+eps))
	wantOut := make([]float32, cols)
	for i := range cols {
		wantOut[i] = x[i] * invRMS * normW[i] * w[i]
	}

	simCases := []struct {
		major, minor int
		wantArch     int32
		wantErrCode  int32
	}{
		{9, 0, 900, 0},
		{10, 0, 1000, 0},
		{10, 3, 1030, 0},
		{10, 7, 1070, 0},
		{12, 0, 1200, 0},
		{12, 1, 1210, 0},
		{8, 0, -1, 209}, // cudaErrorNoKernelImageForDevice
	}

	for _, sc := range simCases {
		t.Run(fmt.Sprintf("dispatch_%d_%d", sc.major, sc.minor), func(t *testing.T) {
			launchStmt := fmt.Sprintf(
				"eg_emu_set_compute_capability(%d, %d); err_code[0] = (int)dispatch_decoder_layer_by_device(0, x, w, norm_w, out, dispatched_arch, %d, 1e-5f)",
				sc.major, sc.minor, cols,
			)
			res, err := cudahostemu.Run(context.Background(), t.TempDir(), cudahostemu.Program{
				Source: bundle.DispatcherCUDA,
				Launch: launchStmt,
				Buffers: []cudahostemu.Buffer{
					{Name: "x", CType: "float", Data: packFloatsLE(x)},
					{Name: "w", CType: "float", Data: packFloatsLE(w)},
					{Name: "norm_w", CType: "float", Data: packFloatsLE(normW)},
					{Name: "out", CType: "float", Data: cudahostemu.Poison(4 * cols)},
					{Name: "dispatched_arch", CType: "int", Data: packIntsLE([]int32{0})},
					{Name: "err_code", CType: "int", Data: packIntsLE([]int32{-1})},
				},
			})
			if err != nil {
				t.Fatalf("cudahostemu.Run(%d.%d): %v", sc.major, sc.minor, err)
			}
			gotErr := unpackIntsLE(res["err_code"])[0]
			gotArch := unpackIntsLE(res["dispatched_arch"])[0]
			if gotErr != sc.wantErrCode {
				t.Fatalf("err_code = %d, want %d", gotErr, sc.wantErrCode)
			}
			if gotArch != sc.wantArch {
				t.Fatalf("dispatched_arch = %d, want %d", gotArch, sc.wantArch)
			}
			if sc.wantErrCode == 0 {
				gotOut := unpackFloatsLE(res["out"])
				for i := range cols {
					if diff := math.Abs(float64(gotOut[i] - wantOut[i])); diff > 1e-5 {
						t.Fatalf("out[%d] = %g, want %g (diff %g)", i, gotOut[i], wantOut[i], diff)
					}
				}
			}
		})
	}
}

func ExampleClassifyComputeCapability() {
	for _, cc := range [][2]int{{9, 0}, {10, 0}, {10, 3}, {12, 0}, {12, 1}} {
		spec, err := ClassifyComputeCapability(cc[0], cc[1])
		if err != nil {
			panic(err)
		}
		fmt.Printf("%s family=%s tmem=%v tcgen05=%v scale4x=%v warp_mma=%v\n",
			spec.Target, spec.FamilyTarget, spec.HasTMEM, spec.HasTCGen05, spec.HasScaleVec4X, spec.HasWarpBlockScaledMMA)
	}
	// Output:
	// sm_90a family=sm_90a tmem=false tcgen05=false scale4x=false warp_mma=false
	// sm_100a family=sm_100f tmem=true tcgen05=true scale4x=false warp_mma=false
	// sm_103a family=sm_103f tmem=true tcgen05=true scale4x=true warp_mma=false
	// sm_120a family=sm_120f tmem=false tcgen05=false scale4x=false warp_mma=true
	// sm_121a family=sm_120f tmem=false tcgen05=false scale4x=false warp_mma=true
}

func ExampleBuildMultiArchBundle() {
	cfg := ArchConfig{
		HiddenSize:        512,
		IntermediateSize:  1024,
		NumAttentionHeads: 8,
		NumKeyValueHeads:  2,
		HeadDim:           64,
		WeightDType:       NVFP4,
		KVDType:           E4M3,
	}
	bundle, err := BuildMultiArchBundle("layer", cfg, []string{"sm_90a", "sm_100a", "sm_103a", "sm_120a", "sm_121a"})
	if err != nil {
		panic(err)
	}
	fmt.Println(len(bundle.Units), bundle.Units[2].Spec.Target, len(bundle.NVCCFlags()) >= 7)
	// Output:
	// 5 sm_103a true
}
