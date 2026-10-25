package hopper

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

type ffiTensorView struct {
	name    string
	dtype   DType
	data    []byte
	shape   []int
	strides []int
}

func (tv ffiTensorView) cDecl(varName string) string {
	dt := ffiDTypes[tv.dtype]
	shapeStr := fmt.Sprintf("{%d}", tv.shape[0])
	strideStr := fmt.Sprintf("{%d}", tv.strides[0])
	if len(tv.shape) == 2 {
		shapeStr = fmt.Sprintf("{%d, %d}", tv.shape[0], tv.shape[1])
		strideStr = fmt.Sprintf("{%d, %d}", tv.strides[0], tv.strides[1])
	}
	return fmt.Sprintf("tvm::ffi::TensorView %s(%s, %s, %s, host::dl_type<%s>());",
		varName, tv.name, shapeStr, strideStr, dt)
}

// runTVMFFI compiles the verbatim SourceTVMFFI header against testdata/emu.h
// and testdata/tvmffi_mock.h, runs mainBody, and returns all buffer contents.
func runTVMFFI(t *testing.T, ffiHeader, mainBody string, bufs ...emuBuf) map[string][]byte {
	t.Helper()
	cxx, err := exec.LookPath("clang++")
	if err != nil {
		cxx, err = exec.LookPath("g++")
	}
	if err != nil {
		t.Skip("no host C++ compiler; emulated TVM-FFI test not run")
	}
	emu, err := filepath.Abs("testdata/emu.h")
	if err != nil {
		t.Fatal(err)
	}
	mock, err := filepath.Abs("testdata/tvmffi_mock.h")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	var main strings.Builder
	main.WriteString(ffiHeader)
	main.WriteString("\nint main() {\n")
	for _, b := range bufs {
		if err := os.WriteFile(filepath.Join(dir, b.name+".bin"), b.data, 0o644); err != nil {
			t.Fatal(err)
		}
		fmt.Fprintf(&main, "    auto eg_b_%s = eg_emu_read(%q);\n    auto* %s = reinterpret_cast<%s*>(eg_b_%s.data());\n",
			b.name, filepath.Join(dir, b.name+".bin"), b.name, b.ctype, b.name)
	}
	main.WriteString(mainBody)
	for _, b := range bufs {
		fmt.Fprintf(&main, "    eg_emu_write(%q, %s, %d);\n", filepath.Join(dir, b.name+".out"), b.name, len(b.data))
	}
	main.WriteString("    return 0;\n}\n")
	cc := filepath.Join(dir, "main.cc")
	if err := os.WriteFile(cc, []byte(main.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(dir, "main")
	cmd := exec.Command(cxx, "-std=c++20", "-O1", "-w", "-include", emu, "-include", mock, "-o", bin, cc)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("compile TVM-FFI header: %v\n%s\n--- source ---\n%s", err, out, numbered(main.String()))
	}
	if out, err := exec.Command(bin).CombinedOutput(); err != nil {
		t.Fatalf("run TVM-FFI binary: %v\n%s", err, out)
	}
	res := make(map[string][]byte, len(bufs))
	for _, b := range bufs {
		data, err := os.ReadFile(filepath.Join(dir, b.name+".out"))
		if err != nil {
			t.Fatal(err)
		}
		res[b.name] = data
	}
	return res
}

func makeFFIBufs(views []ffiTensorView) ([]emuBuf, string, []string) {
	var ebufs []emuBuf
	var decls strings.Builder
	var tvNames []string
	for _, tv := range views {
		ebufs = append(ebufs, emuBuf{tv.name, tv.dtype.cType(), tv.data})
		tvName := "tv_" + tv.name
		decls.WriteString("    " + tv.cDecl(tvName) + "\n")
		tvNames = append(tvNames, tvName)
	}
	return ebufs, decls.String(), tvNames
}

// TestTVMFFI compiles SourceTVMFFI headers with the host emulator and TVM-FFI
// mock, verifying valid execution across cluster (rows <= 64) and wide
// (rows > 64) paths, padded inputs, multi-slot reductions, indexed outputs,
// Plan.SourceTVMFFI, and rejection of bad dtypes or strides.
func TestTVMFFI(t *testing.T) {
	r := rand.New(rand.NewPCG(31, 37))

	prenorm := clusterPreNorm(4096, BF16)
	prenorm.ClusterMaxRows = 4
	boundary := clusterBoundary(1024, BF16)
	slots := clusterSlotSum(4096, 8, BF16)
	slots.ClusterMaxRows = 4

	hdr, err := SourceTVMFFI(prenorm, boundary, slots)
	if err != nil {
		t.Fatal(err)
	}

	// 1. Cluster path (rows = 3) and Wide path (rows = 5 > ClusterMaxRows) on prenorm.
	for _, rows := range []int{3, 5} {
		t.Run(fmt.Sprintf("prenorm_rows_%d", rows), func(t *testing.T) {
			const d = 4096
			bufs := map[string]typed{
				"x":   {BF16, randVals(r, rows*d, 2)},
				"w":   {BF16, randVals(r, d, 0.2)},
				"out": {BF16, make([]float64, rows*d)},
			}
			views := []ffiTensorView{
				{"x", BF16, bufs["x"].bytes(), []int{rows, d}, []int{d, 1}},
				{"w", BF16, bufs["w"].bytes(), []int{d}, []int{1}},
				{"out", BF16, bufs["out"].bytes(), []int{rows, d}, []int{d, 1}},
			}
			ebufs, decls, tvNames := makeFFIBufs(views)
			body := fmt.Sprintf("%s    eg::row_prenorm::run(%d, %s);\n",
				decls, rows, strings.Join(tvNames, ", "))
			got := runTVMFFI(t, hdr, body, ebufs...)

			src, err := prenorm.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			wantVariant := "cluster"
			if rows > prenorm.ClusterMaxRows {
				wantVariant = "wide"
			}
			want := runRowVariant(t, prenorm, src, rows, wantVariant, bufs)
			if !bytes.Equal(got["out"], want["out"]) {
				t.Fatalf("rows=%d: TVM-FFI output differs from %s kernel", rows, wantVariant)
			}
		})
	}

	// 2. Boundary kernel with padded narrower input (branchCols = 1024, D = 8192).
	t.Run("boundary_padded_slice1024", func(t *testing.T) {
		const rows, d, sliceCols = 2, 8192, 1024
		bufs := map[string]typed{
			"resid":     {F32, randVals(r, rows*d, 1)},
			"branch":    {BF16, randVals(r, rows*sliceCols, 2)},
			"post_w":    {BF16, randVals(r, d, 0.5)},
			"alpha":     {F32, randVals(r, d, 0.5)},
			"beta":      {F32, randVals(r, d, 0.5)},
			"pre_w":     {BF16, randVals(r, d, 0.5)},
			"resid_out": {F32, make([]float64, rows*d)},
			"normed":    {BF16, make([]float64, rows*d)},
		}
		views := []ffiTensorView{
			{"resid", F32, bufs["resid"].bytes(), []int{rows, d}, []int{d, 1}},
			{"branch", BF16, bufs["branch"].bytes(), []int{rows, sliceCols}, []int{sliceCols, 1}},
			{"post_w", BF16, bufs["post_w"].bytes(), []int{d}, []int{1}},
			{"alpha", F32, bufs["alpha"].bytes(), []int{d}, []int{1}},
			{"beta", F32, bufs["beta"].bytes(), []int{d}, []int{1}},
			{"pre_w", BF16, bufs["pre_w"].bytes(), []int{d}, []int{1}},
			{"resid_out", F32, bufs["resid_out"].bytes(), []int{rows, d}, []int{d, 1}},
			{"normed", BF16, bufs["normed"].bytes(), []int{rows, d}, []int{d, 1}},
		}
		ebufs, decls, tvNames := makeFFIBufs(views)
		body := fmt.Sprintf("%s    eg::row_boundary::run(%d, %s);\n",
			decls, rows, strings.Join(tvNames, ", "))
		got := runTVMFFI(t, hdr, body, ebufs...)

		src, err := boundary.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		want := runRowVariant(t, boundary, src, rows, "cluster", bufs)
		for _, out := range []string{"resid_out", "normed"} {
			if !bytes.Equal(got[out], want[out]) {
				t.Fatalf("%s: TVM-FFI output differs from cluster kernel", out)
			}
		}
	})

	// 3. Multi-slot reduction kernel across cluster (rows = 2) and wide (rows = 5 > ClusterMaxRows) paths.
	for _, rows := range []int{2, 5} {
		t.Run(fmt.Sprintf("slot_sum_rows_%d", rows), func(t *testing.T) {
			const d, k = 4096, 8
			ids := make([]float64, rows*k)
			weights := make([]float64, rows*k)
			for i := range ids {
				if i%5 == 0 {
					ids[i] = -1
				} else {
					ids[i] = float64(i % 16)
				}
				weights[i] = 0.25 * float64(1+i%4)
			}
			bufs := map[string]typed{
				"y":       {BF16, randVals(r, rows*k*d, 2)},
				"w":       {BF16, randVals(r, d, 0.1)},
				"ids":     {I32, ids},
				"weights": {F32, weights},
				"out":     {BF16, make([]float64, rows*d)},
			}
			views := []ffiTensorView{
				{"y", BF16, bufs["y"].bytes(), []int{rows * k, d}, []int{d, 1}},
				{"w", BF16, bufs["w"].bytes(), []int{d}, []int{1}},
				{"ids", I32, bufs["ids"].bytes(), []int{rows * k}, []int{1}},
				{"weights", F32, bufs["weights"].bytes(), []int{rows * k}, []int{1}},
				{"out", BF16, bufs["out"].bytes(), []int{rows, d}, []int{d, 1}},
			}
			ebufs, decls, tvNames := makeFFIBufs(views)
			body := fmt.Sprintf("%s    eg::row_slot_sum::run(%d, %s);\n",
				decls, rows, strings.Join(tvNames, ", "))
			got := runTVMFFI(t, hdr, body, ebufs...)

			src, err := slots.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			wantVariant := "cluster"
			if rows > slots.ClusterMaxRows {
				wantVariant = "wide"
			}
			want := runRowVariant(t, slots, src, rows, wantVariant, bufs)
			if !bytes.Equal(got["out"], want["out"]) {
				t.Fatalf("rows=%d: slot_sum TVM-FFI output differs from %s kernel", rows, wantVariant)
			}
		})
	}

	// 4. Plan.SourceTVMFFI on a lowered e-graph kernel and rejection checks.
	t.Run("plan_and_rejection_checks", func(t *testing.T) {
		const rows, d = 3, 128
		b := NewBuilder()
		x := b.Input("x", BF16, rows, d)
		w := b.Input("w", BF16, d)
		ss := b.ReduceSum(b.Map(MustLambda("(mul $0 $0)"), x))
		y := b.Map(MustLambda("(mul (mul $0 (rsqrt (add (div $1 128) 1e-6))) $2)"), x, ss, w)
		expr, err := Optimize(b.G, y)
		if err != nil {
			t.Fatal(err)
		}
		p, err := Compile(expr, "plan_rmsnorm", BF16)
		if err != nil {
			t.Fatal(err)
		}
		planHdr, err := p.SourceTVMFFI()
		if err != nil {
			t.Fatal(err)
		}
		outName := p.Output.Name
		bufs := map[string]typed{
			"x":     {BF16, randVals(r, rows*d, 1)},
			"w":     {BF16, randVals(r, d, 0.5)},
			outName: {BF16, make([]float64, rows*d)},
		}
		views := []ffiTensorView{
			{"x", BF16, bufs["x"].bytes(), []int{rows, d}, []int{d, 1}},
			{"w", BF16, bufs["w"].bytes(), []int{d}, []int{1}},
			{outName, BF16, bufs[outName].bytes(), []int{rows, d}, []int{d, 1}},
		}
		ebufs, decls, tvNames := makeFFIBufs(views)
		kName := p.Kernels[0].Row.Name
		// Test valid run + dtype rejection + stride rejection in one binary.
		body := fmt.Sprintf(`%s    eg::%s::run(%d, %s);
    bool caught_dtype = false;
    try {
        tvm::ffi::TensorView bad_w(w, {%d}, {1}, host::dl_type<fp32_t>());
        eg::%s::run(%d, tv_x, bad_w, tv_%s);
    } catch (const std::exception& e) {
        caught_dtype = std::string(e.what()).find("dtype") != std::string::npos;
    }
    if (!caught_dtype) return 2;

    bool caught_stride = false;
    try {
        tvm::ffi::TensorView bad_x(x, {%d, %d}, {%d, 1}, host::dl_type<bf16_t>());
        eg::%s::run(%d, bad_x, tv_w, tv_%s);
    } catch (const std::exception& e) {
        caught_stride = std::string(e.what()).find("stride") != std::string::npos;
    }
    if (!caught_stride) return 3;
`, decls, kName, rows, strings.Join(tvNames, ", "), d, kName, rows, outName, rows, d, d*2, kName, rows, outName)

		got := runTVMFFI(t, planHdr, body, ebufs...)
		_, want := runRowKernel(t, *p.Kernels[0].Row, rows, bufs)
		agree(t, "plan_rmsnorm", BF16, decode(BF16, got[outName]), want[outName])
	})
}

// TestTVMFFILineLimit verifies that every line of SourceTVMFFI output is at
// most MaxFFILineLen (120) characters and contains no model-specific names.
func TestTVMFFILineLimit(t *testing.T) {
	hdr, err := SourceTVMFFI(
		clusterPreNorm(4096, BF16),
		clusterBoundary(0, BF16),
		clusterBoundary(1024, BF16),
		clusterSlotSum(4096, 8, BF16),
	)
	if err != nil {
		t.Fatal(err)
	}
	for i, line := range strings.Split(hdr, "\n") {
		if len(line) > MaxFFILineLen {
			t.Errorf("line %d has length %d > %d: %s", i+1, len(line), MaxFFILineLen, line)
		}
	}
	if strings.Contains(strings.ToLower(hdr), "moe8") {
		t.Errorf("SourceTVMFFI output contains model-specific name 'moe8'")
	}
}
