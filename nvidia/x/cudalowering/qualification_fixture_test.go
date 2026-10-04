package cudalowering

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/nvvmx"
)

const qualificationGuard = uint32(0xa17e5eed)

// A failed launch or synchronization does not establish quiescence. Keep its
// raw resources for the enclosing dedicated context if a final drain fails.
func qualificationCleanup(complete bool, drain, free, unload func() error) error {
	if !complete {
		if err := drain(); err != nil {
			return fmt.Errorf("unresolved device work retained for context cleanup: %w", err)
		}
	}
	return errors.Join(free(), unload())
}

func TestAddrSpaceQualificationCleanup(t *testing.T) {
	injected := errors.New("injected drain failure")
	for _, tc := range []struct {
		name                string
		complete, failDrain bool
		wantCalls           []string
	}{
		{"complete", true, false, []string{"free", "unload"}},
		{"retry succeeded", false, false, []string{"drain", "free", "unload"}},
		{"unresolved", false, true, []string{"drain"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls []string
			err := qualificationCleanup(tc.complete,
				func() error {
					calls = append(calls, "drain")
					if tc.failDrain {
						return injected
					}
					return nil
				},
				func() error { calls = append(calls, "free"); return nil },
				func() error { calls = append(calls, "unload"); return nil })
			if errors.Is(err, injected) != tc.failDrain || strings.Join(calls, ",") != strings.Join(tc.wantCalls, ",") {
				t.Fatalf("cleanup %v,%v", calls, err)
			}
		})
	}
}

type qualificationImage struct {
	Name                                         string
	IRDigest, PTXDigest, CUBINDigest, SASSDigest string
	PTXConversions, SASSConversions              int
	Observation                                  qualificationObservation
}

type qualificationObservation struct {
	Report                  Report
	ReportError             string
	PTXStores, SASSStores   int
	PTXOutputs, SASSOutputs int
	CompilerForwarding      string
}

const qualificationForwarding = "after shared load erased; instruction streams consistent with store-to-load forwarding, not proof of causality or optimized performance"

func qualificationCheckObservation(name string, o qualificationObservation) (int, int, error) {
	if len(o.Report.Stats) != 1 || o.Report.Stats[0].Kernel != "kernel" || len(o.Report.Divergences) != 1 {
		return 0, 0, fmt.Errorf("expected exact matched kernel and divergence")
	}
	if o.PTXStores != 1 || o.SASSStores != 1 || o.PTXOutputs != 1 || o.SASSOutputs != 1 {
		return 0, 0, fmt.Errorf("missing or extra shared store or global output")
	}
	v := o.Report.Divergences[0]
	if v.Kernel != "kernel" || o.ReportError == "" || o.Report.Err() == nil || o.ReportError != o.Report.Err().Error() {
		return 0, 0, fmt.Errorf("unbound divergence metadata")
	}
	switch name {
	case "before":
		if v.Kind != SharedConversion || v.PTXOpcode != "cvta.shared" || v.SASSOpcode != "conversion" || v.PTXCount != 1 || v.SASSCount != 1 || v.Detail != "requested labels (not slot attribution): @kernel/%slot; observed SM121 mechanism: window=0x50 low-copy=0x70 high-copy=0x80 address-def=0x60 shared-access=0x90 generic-access=0xa0" || o.CompilerForwarding != "" {
			return 0, 0, fmt.Errorf("incorrect before conversion observation")
		}
		return 1, 1, nil
	case "after":
		if v.Kind != Erased || v.PTXOpcode != "ld.shared" || v.SASSOpcode != "LDS" || v.PTXCount != 1 || v.SASSCount != 0 || v.Detail != "" || o.CompilerForwarding != qualificationForwarding {
			return 0, 0, fmt.Errorf("incorrect after load erasure observation")
		}
		return 0, 0, nil
	default:
		return 0, 0, fmt.Errorf("unknown image arm")
	}
}

type qualificationBuild struct {
	Target       string
	Optimization int
	Images       []qualificationImage
	Tools        map[string]string
	Commands     [][]string
}

func qualificationIR() ([]byte, []byte, error) {
	a := append(addressSpaceABInput(), []byte("!nvvmir.version = !{!1}\n!1 = !{i32 2, i32 0, i32 3, i32 2}\n")...)
	a = bytes.Replace(a, []byte("e-p:64:64:64-p3:32:32:32"), []byte("e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8"), 1)
	b, _, err := nvvmx.InferAddressSpaces(a)
	return a, b, err
}

func qualificationDigest(b []byte) string { return fmt.Sprintf("%x", sha256.Sum256(b)) }

func qualificationRead(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	s, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !s.Mode().IsRegular() || s.Size() == 0 || s.Size() > 16<<20 {
		return nil, fmt.Errorf("invalid qualification file size or kind")
	}
	b, err := io.ReadAll(io.LimitReader(f, (16<<20)+1))
	if err != nil {
		return nil, err
	}
	if len(b) == 0 || len(b) > 16<<20 {
		return nil, fmt.Errorf("qualification file exceeds bound")
	}
	return b, nil
}

func qualificationInspect(name string, ptx, sass []byte) (int, int, qualificationObservation, error) {
	var o qualificationObservation
	m, err := cudaptx.Parse(bytes.NewReader(ptx))
	if err != nil {
		return 0, 0, o, err
	}
	d, err := cudasass.Parse(bytes.NewReader(sass))
	if err != nil {
		return 0, 0, o, err
	}
	r, err := VerifyNoSharedConversions(m, d, "@kernel/%slot")
	if err != nil {
		return 0, 0, o, err
	}
	if len(r.Stats) != 1 || r.Stats[0].Kernel != "kernel" {
		return 0, 0, o, fmt.Errorf("expected exactly one matched kernel")
	}
	if len(m.Kernels) != 1 || len(d.Functions) != 1 {
		return 0, 0, o, fmt.Errorf("unexpected instruction stream")
	}
	o.Report = *r
	if err := r.Err(); err != nil {
		o.ReportError = err.Error()
	}
	for _, inst := range m.Kernels[0].Instructions {
		switch inst.Opcode {
		case "st.shared.u32":
			o.PTXStores++
		case "st.global.u32":
			o.PTXOutputs++
		}
	}
	for _, inst := range d.Functions[0].Instructions {
		if inst.Opcode == "EXIT" {
			break
		}
		switch inst.Opcode {
		case "STS":
			o.SASSStores++
		case "STG.E":
			o.SASSOutputs++
		}
	}
	if name == "after" {
		o.CompilerForwarding = qualificationForwarding
	}
	pc, sc, err := qualificationCheckObservation(name, o)
	return pc, sc, o, err
}

func qualificationCheckOutput(got []uint32) error {
	if len(got) != 3 {
		return fmt.Errorf("expected three guarded words")
	}
	if got[0] != qualificationGuard || got[2] != qualificationGuard {
		return fmt.Errorf("guard changed")
	}
	if got[1] != 42 {
		return fmt.Errorf("output=%d, want 42", got[1])
	}
	return nil
}

func qualificationCheckBuild(b qualificationBuild) error {
	if b.Target != "sm_121" || b.Optimization != 0 || len(b.Images) != 2 {
		return fmt.Errorf("only the opt0 sm121 mechanism packet is admitted")
	}
	for i, name := range []string{"before", "after"} {
		v := b.Images[i]
		if v.Name != name {
			return fmt.Errorf("incorrect A/B image order")
		}
		for _, h := range []string{v.IRDigest, v.PTXDigest, v.CUBINDigest, v.SASSDigest} {
			if len(h) != 64 || strings.Trim(h, "0123456789abcdef") != "" {
				return fmt.Errorf("missing or invalid image binding")
			}
		}
		pc, sc, err := qualificationCheckObservation(name, v.Observation)
		if err != nil || pc != v.PTXConversions || sc != v.SASSConversions {
			return fmt.Errorf("incorrect bound inspection metadata: %v", err)
		}
	}
	if b.Images[0].SASSConversions <= 0 || b.Images[1].SASSConversions != 0 || b.Images[0].PTXConversions <= 0 || b.Images[1].PTXConversions != 0 {
		return fmt.Errorf("vacuous or remaining conversion")
	}
	if b.Images[0].IRDigest == b.Images[1].IRDigest || b.Images[0].CUBINDigest == b.Images[1].CUBINDigest {
		return fmt.Errorf("identical A/B images")
	}
	for _, name := range []string{"ptxas", "nvdisasm", "nvvm"} {
		if b.Tools[name] == "" {
			return fmt.Errorf("missing tool identity: %s", name)
		}
	}
	return nil
}

func TestAddrSpaceQualificationChecker(t *testing.T) {
	for _, tc := range []struct {
		name  string
		words []uint32
		want  bool
	}{
		{"healthy", []uint32{qualificationGuard, 42, qualificationGuard}, true},
		{"corrupt", []uint32{qualificationGuard, 43, qualificationGuard}, false},
		{"poison", []uint32{qualificationGuard, 0xdeadbeef, qualificationGuard}, false},
		{"guard", []uint32{0, 42, qualificationGuard}, false},
		{"empty", nil, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if (qualificationCheckOutput(tc.words) == nil) != tc.want {
				t.Fatal("incorrect checker result")
			}
		})
	}
	a, b, err := qualificationIR()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(a, []byte("addrspacecast (ptr addrspace(3) @shared_mem to ptr)")) || bytes.Equal(a, b) {
		t.Fatal("no explicit semantic A/B transformation")
	}
}

func TestAddrSpaceQualificationBuildAdmission(t *testing.T) {
	a, b, err := qualificationIR()
	if err != nil {
		t.Fatal(err)
	}
	good := qualificationBuild{Target: "sm_121", Optimization: 0, Tools: map[string]string{"ptxas": "toolhash", "nvdisasm": "toolhash", "nvvm": "toolhash"}, Images: []qualificationImage{
		{"before", qualificationDigest(a), qualificationDigest([]byte("before PTX")), qualificationDigest([]byte("before CUBIN")), qualificationDigest([]byte("before SASS")), 1, 1, qualificationTestObservation("before")},
		{"after", qualificationDigest(b), qualificationDigest([]byte("after PTX")), qualificationDigest([]byte("after CUBIN")), qualificationDigest([]byte("after SASS")), 0, 0, qualificationTestObservation("after")},
	}}
	for _, tc := range []struct {
		name   string
		mutate func(*qualificationBuild)
		want   bool
	}{
		{"healthy", func(*qualificationBuild) {}, true},
		{"vacuous optimized control", func(v *qualificationBuild) { v.Optimization = 3; v.Images[0].SASSConversions = 0 }, false},
		{"vacuous mechanism", func(v *qualificationBuild) { v.Images[0].SASSConversions = 0 }, false},
		{"PTX only", func(v *qualificationBuild) { v.Images[0].SASSConversions = 0 }, false},
		{"remaining conversion", func(v *qualificationBuild) { v.Images[1].SASSConversions = 1 }, false},
		{"missing image", func(v *qualificationBuild) { v.Images = v.Images[:1] }, false},
		{"missing binding", func(v *qualificationBuild) { v.Images[1].CUBINDigest = "" }, false},
		{"same binary", func(v *qualificationBuild) { v.Images[1].CUBINDigest = v.Images[0].CUBINDigest }, false},
		{"wrong target", func(v *qualificationBuild) { v.Target = "sm_100" }, false},
		{"missing tool", func(v *qualificationBuild) { delete(v.Tools, "nvvm") }, false},
		{"unbound observation", func(v *qualificationBuild) { v.Images[1].Observation.ReportError = "invented" }, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			v := good
			v.Images = append([]qualificationImage(nil), good.Images...)
			v.Tools = make(map[string]string)
			for k, s := range good.Tools {
				v.Tools[k] = s
			}
			tc.mutate(&v)
			if (qualificationCheckBuild(v) == nil) != tc.want {
				t.Fatal("incorrect build admission")
			}
		})
	}
}

// Exact retained default-ptxas CUDA13.0 after-image instruction streams.
const qualificationAfterPTX = `//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-36424714
// Cuda compilation tools, release 13.0, V13.0.88
// Based on NVVM 20.0.0
//

.version 9.0
.target sm_121
.address_size 64

	// .globl	kernel
// shared_mem has been demoted

.visible .entry kernel(
	.param .u64 .ptr .global .align 1 kernel_param_0,
	.param .u32 kernel_param_1
)
{
	.reg .b32 	%r<4>;
	.reg .b64 	%rd<2>;
	// demoted variable
	.shared .align 4 .u32 shared_mem;
	ld.param.u64 	%rd1, [kernel_param_0];
	ld.param.u32 	%r1, [kernel_param_1];
	st.shared.u32 	[shared_mem], %r1;
	ld.shared.u32 	%r2, [shared_mem];
	add.s32 	%r3, %r2, 7;
	st.global.u32 	[%rd1], %r3;
	ret;

}
`
const qualificationAfterSASS = `	.target	sm_121

	.elftype	@"ET_EXEC"


//--------------------- .text.kernel              --------------------------
	.section	.text.kernel,"ax",@progbits
	.align	128
        .global         kernel
        .type           kernel,@function
        .size           kernel,(.L_x_1 - kernel)
        .other          kernel,@"STO_CUDA_ENTRY STV_DEFAULT"
kernel:
.text.kernel:
        /*0000*/                   LDC R1, c[0x0][0x37c] ;
        /*0010*/                   S2UR UR5, SR_CgaCtaId ;
        /*0020*/                   LDCU.64 UR6, c[0x0][0x358] ;
        /*0030*/                   LDC.64 R2, c[0x0][0x380] ;
        /*0040*/                   UMOV UR4, 0x400 ;
        /*0050*/                   LDC R0, c[0x0][0x388] ;
        /*0060*/                   ULEA UR4, UR5, UR4, 0x18 ;
        /*0070*/                   IADD3 R5, PT, PT, R0, 0x7, RZ ;
        /*0080*/                   STS [UR4], R0 ;
        /*0090*/                   STG.E desc[UR6][R2.64], R5 ;
        /*00a0*/                   EXIT ;
.L_x_0:
        /*00b0*/                   BRA ` + "`" + `(.L_x_0);
        /*00c0*/                   NOP;
        /*00d0*/                   NOP;
        /*00e0*/                   NOP;
        /*00f0*/                   NOP;
        /*0100*/                   NOP;
        /*0110*/                   NOP;
        /*0120*/                   NOP;
        /*0130*/                   NOP;
        /*0140*/                   NOP;
        /*0150*/                   NOP;
        /*0160*/                   NOP;
        /*0170*/                   NOP;
.L_x_1:


//--------------------- SYMBOLS --------------------------

	.type		.nv.reservedSmem.offset0,@object
	.size		.nv.reservedSmem.offset0,0x4
	.type		.nv.reservedSmem.cap,@object
	.size		.nv.reservedSmem.cap,0x4
`

func qualificationTestObservation(name string) qualificationObservation {
	d := Divergence{Kind: Erased, Kernel: "kernel", PTXOpcode: "ld.shared", SASSOpcode: "LDS", PTXCount: 1}
	o := qualificationObservation{PTXStores: 1, SASSStores: 1, PTXOutputs: 1, SASSOutputs: 1, CompilerForwarding: qualificationForwarding}
	if name == "before" {
		d = Divergence{Kind: SharedConversion, Kernel: "kernel", PTXOpcode: "cvta.shared", SASSOpcode: "conversion", PTXCount: 1, SASSCount: 1, Detail: "requested labels (not slot attribution): @kernel/%slot; observed SM121 mechanism: window=0x50 low-copy=0x70 high-copy=0x80 address-def=0x60 shared-access=0x90 generic-access=0xa0"}
		o.CompilerForwarding = ""
	}
	o.Report = Report{Stats: []Stats{{Kernel: "kernel"}}, Divergences: []Divergence{d}}
	o.ReportError = o.Report.Err().Error()
	return o
}

func TestAddrSpaceQualificationErasurePolicy(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*qualificationObservation)
		want   bool
	}{
		{"exact after", func(*qualificationObservation) {}, true},
		{"extra divergence", func(o *qualificationObservation) {
			o.Report.Divergences = append(o.Report.Divergences, o.Report.Divergences[0])
		}, false},
		{"wrong kind", func(o *qualificationObservation) { o.Report.Divergences[0].Kind = Unsourced }, false},
		{"wrong kernel", func(o *qualificationObservation) { o.Report.Divergences[0].Kernel = "other" }, false},
		{"wrong opcode", func(o *qualificationObservation) { o.Report.Divergences[0].PTXOpcode = "st.shared" }, false},
		{"wrong SASS opcode", func(o *qualificationObservation) { o.Report.Divergences[0].SASSOpcode = "STS" }, false},
		{"wrong PTX count", func(o *qualificationObservation) { o.Report.Divergences[0].PTXCount = 2 }, false},
		{"wrong SASS count", func(o *qualificationObservation) { o.Report.Divergences[0].SASSCount = 1 }, false},
		{"extra detail", func(o *qualificationObservation) { o.Report.Divergences[0].Detail = "unbound" }, false},
		{"missing PTX store", func(o *qualificationObservation) { o.PTXStores = 0 }, false},
		{"missing SASS store", func(o *qualificationObservation) { o.SASSStores = 0 }, false},
		{"missing PTX output", func(o *qualificationObservation) { o.PTXOutputs = 0 }, false},
		{"missing SASS output", func(o *qualificationObservation) { o.SASSOutputs = 0 }, false},
		{"metadata mismatch", func(o *qualificationObservation) { o.ReportError = "invented" }, false},
		{"missing forwarding record", func(o *qualificationObservation) { o.CompilerForwarding = "" }, false},
		{"wrong matched kernel", func(o *qualificationObservation) { o.Report.Stats[0].Kernel = "other" }, false},
		{"missing matched kernel", func(o *qualificationObservation) { o.Report.Stats = nil }, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			o := qualificationTestObservation("after")
			tt.mutate(&o)
			if tt.name != "metadata mismatch" {
				o.ReportError = o.Report.Err().Error()
			}
			_, _, err := qualificationCheckObservation("after", o)
			if (err == nil) != tt.want {
				t.Fatalf("policy result: %v", err)
			}
		})
	}
	if _, _, err := qualificationCheckObservation("before", qualificationTestObservation("after")); err == nil {
		t.Fatal("before erasure accepted")
	}
	before := qualificationTestObservation("before")
	if _, _, err := qualificationCheckObservation("before", before); err != nil {
		t.Fatal(err)
	}
	before.Report.Divergences[0].Detail = "invented chain"
	before.ReportError = before.Report.Err().Error()
	if _, _, err := qualificationCheckObservation("before", before); err == nil {
		t.Fatal("unbound before chain accepted")
	}
}

func TestAddrSpaceQualificationObservationBinding(t *testing.T) {
	_, _, o, err := qualificationInspect("after", []byte(qualificationAfterPTX), []byte(qualificationAfterSASS))
	if err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(o)
	if err != nil {
		t.Fatal(err)
	}
	var got qualificationObservation
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(o, got) {
		t.Fatal("typed inspection metadata lost in receipt")
	}
	got.Report.Stats[0].PTXInstructions++
	if reflect.DeepEqual(o, got) {
		t.Fatal("inspection metadata substitution survived binding")
	}
}

func TestAddrSpaceQualificationInspection(t *testing.T) {
	tests := []struct {
		name      string
		ptx, sass string
		want      bool
	}{
		{"exact observed after", qualificationAfterPTX, qualificationAfterSASS, true},
		{"missing", qualificationAfterPTX, "", false},
		{"mismatched", qualificationAfterPTX, strings.ReplaceAll(qualificationAfterSASS, "kernel", "other"), false},
		{"wrong target", qualificationAfterPTX, strings.ReplaceAll(qualificationAfterSASS, "sm_121", "sm_120"), false},
		{"unknown scanner opcode", qualificationAfterPTX, strings.Replace(qualificationAfterSASS, "IADD3", "FOO", 1), false},
		{"malformed IADD3", qualificationAfterPTX, strings.Replace(qualificationAfterSASS, "PT, PT", "P0, PT", 1), false},
		{"scanner gap", qualificationAfterPTX, strings.Replace(qualificationAfterSASS, "/*0070*/", "/*0078*/", 1), false},
		{"missing SASS store", qualificationAfterPTX, strings.Replace(qualificationAfterSASS, "STS [UR4], R0", "MOV R8, R0", 1), false},
		{"missing SASS output", qualificationAfterPTX, strings.Replace(qualificationAfterSASS, "STG.E desc[UR6][R2.64], R5", "MOV R8, R5", 1), false},
		{"missing PTX store", strings.Replace(qualificationAfterPTX, "st.shared.u32", "st.global.u32", 1), qualificationAfterSASS, false},
		{"missing PTX output", strings.Replace(qualificationAfterPTX, "st.global.u32", "st.shared.u32", 1), qualificationAfterSASS, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pc, sc, o, err := qualificationInspect("after", []byte(tt.ptx), []byte(tt.sass))
			if (err == nil) != tt.want {
				t.Fatalf("counts=%d/%d err=%v", pc, sc, err)
			}
			if tt.want {
				if pc != 0 || sc != 0 || o.Report.Err() == nil || o.ReportError != o.Report.Err().Error() || o.CompilerForwarding != qualificationForwarding {
					t.Fatalf("lost divergence metadata: %+v", o)
				}
			}
		})
	}
	if _, _, _, err := qualificationInspect("before", []byte(qualificationAfterPTX), []byte(qualificationAfterSASS)); err == nil {
		t.Fatal("before erasure accepted")
	}
}
