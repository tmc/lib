package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestFunctorOf(t *testing.T) {
	tests := []struct {
		name, want string
	}{
		{"at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, false>", "CUDAFunctor_add"},
		{"at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>", "MulFunctor"},
		{"at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>, false>", "bfloat16_copy_kernel_cuda"},
		{"at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&)::{lambda(int)#1}>", "GeluCUDAKernelImpl"},
		{"at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::(anonymous namespace)::threshold_kernel_impl<c10::BFloat16>(at::TensorIteratorBase&, c10::BFloat16, c10::BFloat16)::{lambda(c10::BFloat16, c10::BFloat16)#1}>, std::array<char*, 3ul>, false>", "threshold_kernel_impl"},
		{"at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<long, long, bool, at::native::(anonymous namespace)::CompareEqFunctor<long> >, std::array<char*, 2ul>, false>", "CompareEqFunctor"},
		{"at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&)::{lambda(int)#1}>", "index_kernel_impl"},
		{"at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSelf_add<long>, std::array<char*, 2ul>, false>", "CUDAFunctorOnSelf_add"},
		{"cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x4_nn_align8>", ""},
		{"LayerNormF32", ""},
	}
	for _, tt := range tests {
		if got := functorOf(tt.name); got != tt.want {
			t.Errorf("functorOf(%.60q) = %q, want %q", tt.name, got, tt.want)
		}
	}
}

func TestParseOpsFile(t *testing.T) {
	rules, combines, err := parseOpsFile(strings.NewReader("GeGLU geglu\nelementwise ew\n\ngeglu = ew:gelu*,Mul* (unmatched)\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(rules) != 2 || len(combines) != 1 {
		t.Fatalf("rules %v combines %v", rules, combines)
	}
	c := combines[0]
	if c.Line != 4 || c.Name != "geglu" || len(c.Terms) != 2 || c.Terms[0].String() != "ew:gelu*,mul*" || c.Terms[1].String() != unmatchedOp {
		t.Errorf("combine %+v", c)
	}
	for _, tt := range []struct{ text, want string }{
		{"a a\nx = b\n", "line 2: no rule names op b"},
		{"a a\nx = a\nx = a\n", "line 3: combine x already declared on line 2"},
		{"a a\nx = a:[\n", `line 2: bad glob "["`},
		{"a a\nx =\n", "line 2: combine x has no terms"},
		{"a a\na = a\n", "line 2: combine a includes op a already"},
	} {
		if _, _, err := parseOpsFile(strings.NewReader(tt.text)); err == nil || err.Error() != tt.want {
			t.Errorf("parseOpsFile(%q) = %v, want %s", tt.text, err, tt.want)
		}
	}
	if _, err := parseOpRules(strings.NewReader("a a\nx = a\n")); err == nil || !strings.Contains(err.Error(), "compare only") {
		t.Errorf("parseOpRules took a combine line: %v", err)
	}
}

const (
	torchGelu = "at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::native::GeluType)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>, false>"
	torchMul  = "at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul>, false>"
	torchFill = "at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul>, false>"
)

func TestCompareCombines(t *testing.T) {
	rules, combines, err := parseOpsFile(strings.NewReader("GeGLU geglu\nelementwise ew\n\ngeglu = ew:gelu*,mul*\n"))
	if err != nil {
		t.Fatal(err)
	}
	base := compareCapture([]rec{
		{subtype: "kernel", name: torchGelu, start: 1100, end: 1300},
		{subtype: "kernel", name: torchMul, start: 1300, end: 1400},
		{subtype: "kernel", name: torchFill, start: 1400, end: 1450},
	}, phaseStatus(1000, 2000))
	target := compareCapture([]rec{
		{subtype: "kernel", name: "GeGLUForward", start: 1100, end: 1220},
		{subtype: "kernel", name: torchFill, start: 1300, end: 1350},
	}, phaseStatus(1000, 2000))
	c, err := compareCaptures(base, target, rules, nil, compareOptions{phase: -1, combines: combines})
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Combines) != 1 {
		t.Fatalf("combines %+v", c.Combines)
	}
	r := c.Combines[0]
	if r.BaseLaunches != 2 || r.TargetLaunches != 1 || r.BaseExclusiveNanos != 300 || r.TargetExclusiveNanos != 120 || r.DeltaExclusiveNanos != -180 {
		t.Errorf("combine row %+v", r)
	}
	var names []string
	for _, m := range r.Members {
		names = append(names, m.Op+": "+m.Kernel)
	}
	if got, want := strings.Join(names, ", "), "ew: GeluCUDAKernelImpl, geglu: GeGLUForward, ew: MulFunctor"; got != want {
		t.Errorf("members %s, want %s", got, want)
	}
	var out bytes.Buffer
	if err := c.write(&out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "geglu = ew:gelu*,mul*") {
		t.Errorf("output lacks the combine row:\n%s", out.String())
	}

	_, combines, err = parseOpsFile(strings.NewReader("GeGLU geglu\nelementwise ew\na = ew:gelu*\nb = ew:*impl\n"))
	if err != nil {
		t.Fatal(err)
	}
	_, err = compareCaptures(base, target, rules, nil, compareOptions{phase: -1, combines: combines})
	if err == nil || !strings.Contains(err.Error(), "ops lines 3 and 4 both combine kernel") {
		t.Errorf("overlapping combines: %v", err)
	}
}

func TestCompareOpsCheck(t *testing.T) {
	rules, combines, err := parseOpsFile(strings.NewReader("rms rms\nrms_fused fused\nnever ghost\nx = rms:zzz*\n"))
	if err != nil {
		t.Fatal(err)
	}
	base := compareCapture([]rec{
		{subtype: "kernel", name: "rms_norm", start: 1100, end: 1200},
		{subtype: "kernel", name: "rms_fused", start: 1200, end: 1300},
	}, phaseStatus(1000, 2000))
	target := compareCapture([]rec{
		{subtype: "kernel", name: "rms_fused", start: 1100, end: 1200},
	}, phaseStatus(1000, 2000))
	c, err := compareCaptures(base, target, rules, nil, compareOptions{phase: -1, combines: combines, opsCheck: true})
	if err != nil {
		t.Fatal(err)
	}
	r := c.OpsCheck.Rules
	if r[0].BaseKernels != 2 || r[0].BaseLaunches != 2 || r[0].TargetKernels != 1 {
		t.Errorf("line 1 %+v", r[0])
	}
	if r[1].BaseKernels != 0 || r[1].BaseShadowedKernels != 1 || r[1].TargetShadowedKernels != 1 || len(r[1].ShadowedBy) != 1 || r[1].ShadowedBy[0] != 1 {
		t.Errorf("line 2 %+v", r[1])
	}
	want := "lines 2, 3 won no kernel in either capture; combine terms on lines 4 (rms:zzz*) took no kernel in either capture"
	if err := c.OpsCheck.err(); err == nil || err.Error() != want {
		t.Errorf("err = %v, want %s", err, want)
	}
	var out bytes.Buffer
	if err := c.write(&out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "[1] NOTHING") || !strings.Contains(out.String(), "ops check:") {
		t.Errorf("output lacks the ops check:\n%s", out.String())
	}
}
