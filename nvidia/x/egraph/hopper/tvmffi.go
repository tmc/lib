package hopper

import (
	"fmt"
	"strings"
)

const (
	// DefaultFFINamespace is the default C++ namespace emitted by [SourceTVMFFI].
	DefaultFFINamespace = "eg"

	// MaxFFILineLen is the maximum line length in characters of [SourceTVMFFI] output.
	MaxFFILineLen = 120
)

var ffiDTypes = map[DType]string{
	BF16: "bf16_t",
	F32:  "fp32_t",
	E4M3: "fp8_e4m3_t",
	U8:   "uint8_t",
	I32:  "int32_t",
	I64:  "int64_t",
}

// SourceTVMFFI returns a self-contained CUDA header for TVM-FFI's load_jit
// loader in namespace [DefaultFFINamespace]. Each kernel K gets a host
// wrapper struct eg::K whose static run(int64_t rows, TensorView...) method
// validates tensor shapes, strides, dtypes and devices via TensorMatcher and
// launches K (or K_cluster / K_wide when K.Cluster is set) via LaunchKernel.
// Every emitted line is at most [MaxFFILineLen] characters.
func SourceTVMFFI(kernels ...RowKernel) (string, error) {
	return SourceTVMFFINamespace(DefaultFFINamespace, kernels...)
}

// SourceTVMFFINamespace is like [SourceTVMFFI] but places the emitted kernels
// and wrapper structs in namespace ns (or [DefaultFFINamespace] if ns is empty).
func SourceTVMFFINamespace(ns string, kernels ...RowKernel) (string, error) {
	if ns == "" {
		ns = DefaultFFINamespace
	}
	if len(kernels) == 0 {
		return "", fmt.Errorf("tvmffi: no kernels")
	}
	var b strings.Builder
	b.WriteString("#pragma once\n\n")
	b.WriteString("#if !defined(EG_HOST_EMU)\n")
	b.WriteString("#include <sgl_kernel/tensor.h>\n")
	b.WriteString("#include <sgl_kernel/utils.h>\n")
	b.WriteString("#include <sgl_kernel/utils.cuh>\n")
	b.WriteString("#include <dlpack/dlpack.h>\n")
	b.WriteString("#include <tvm/ffi/container/tensor.h>\n")
	b.WriteString("#endif\n\n")
	b.WriteString(Prelude)
	needCluster := false
	for _, k := range kernels {
		if k.Cluster || k.Slots > 0 {
			needCluster = true
			break
		}
	}
	if needCluster {
		b.WriteString("\n")
		b.WriteString(ClusterPrelude)
	}
	fmt.Fprintf(&b, "\nnamespace %s {\n\nnamespace detail {\n", ns)
	for _, k := range kernels {
		cuda, err := k.CUDA()
		if err != nil {
			return "", err
		}
		// Strip duplicate ClusterPrelude when already emitted above.
		cuda = strings.TrimPrefix(cuda, ClusterPrelude+"\n")
		b.WriteString("\n")
		b.WriteString(cuda)
	}
	b.WriteString("\n}  // namespace detail\n")
	for _, k := range kernels {
		host, err := k.ffiStruct()
		if err != nil {
			return "", err
		}
		b.WriteString("\n")
		b.WriteString(host)
	}
	fmt.Fprintf(&b, "\n}  // namespace %s\n", ns)
	return wrapCLines(b.String(), MaxFFILineLen), nil
}

// SourceTVMFFI returns the TVM-FFI header for k.
func (k RowKernel) SourceTVMFFI() (string, error) {
	return SourceTVMFFI(k)
}

// SourceTVMFFI returns the TVM-FFI header for all row kernels in p.
func (p *Plan) SourceTVMFFI() (string, error) {
	var rows []RowKernel
	for _, pk := range p.Kernels {
		if pk.Row == nil {
			return "", fmt.Errorf("plan %s: tvmffi only supports row kernels", p.Name)
		}
		rows = append(rows, *pk.Row)
	}
	return SourceTVMFFI(rows...)
}

type ffiParam struct {
	name     string
	dtype    DType
	writable bool
	ndim     int
	rowMult  int
	rowDiv   int
	cols     int
	stride0  int
	indexed  bool
}

func (k RowKernel) ffiParams() ([]ffiParam, error) {
	if _, err := k.check(); err != nil {
		return nil, err
	}
	var out []ffiParam
	byName := make(map[string]int)
	slotMult := 1
	if k.Slots > 0 {
		slotMult = k.Slots
	}
	for _, in := range k.Inputs {
		buf := in.buffer()
		idx, seen := byName[buf]
		if !seen {
			idx = len(out)
			byName[buf] = idx
			out = append(out, ffiParam{name: buf, dtype: in.DType, ndim: 1, rowDiv: 1})
		}
		p := &out[idx]
		switch in.Kind {
		case RowData:
			gs, rs := in.strides(k.D)
			valid := k.D
			if in.Padded > 0 {
				valid = in.Padded
			}
			ext := in.Offset + (in.group()-1)*rs + valid
			p.ndim = 2
			p.rowMult = slotMult
			p.rowDiv = in.group()
			p.cols = max(p.cols, ext)
			p.stride0 = gs
		case ColVector:
			if p.ndim == 1 {
				p.cols = max(p.cols, in.Offset+k.D)
			}
		case RowScalar:
			if p.ndim == 1 {
				p.rowMult = slotMult
				p.rowDiv = in.group()
				p.cols = max(p.cols, in.Offset)
			}
		case Scalar:
			if p.ndim == 1 {
				p.cols = max(p.cols, in.Offset+1)
			}
		}
	}
	for _, o := range k.Outputs {
		gs, rs := o.strides(k.D)
		ext := o.Offset + (o.group()-1)*rs + k.D
		if o.Index != "" {
			out = append(out, ffiParam{
				name:     o.Name,
				dtype:    o.DType,
				writable: true,
				ndim:     2,
				cols:     gs,
				stride0:  gs,
				indexed:  true,
			})
			out = append(out, ffiParam{
				name:    o.Index,
				dtype:   o.IndexDType,
				ndim:    1,
				rowMult: 1,
				rowDiv:  o.group(),
			})
		} else {
			out = append(out, ffiParam{
				name:     o.Name,
				dtype:    o.DType,
				writable: true,
				ndim:     2,
				rowMult:  1,
				rowDiv:   o.group(),
				cols:     ext,
				stride0:  gs,
			})
		}
	}
	return out, nil
}

func (k RowKernel) ffiStruct() (string, error) {
	ps, err := k.ffiParams()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	fmt.Fprintf(&b, "struct %s {\n  static void run(\n      int64_t rows", k.Name)
	for _, p := range ps {
		fmt.Fprintf(&b, ",\n      const tvm::ffi::TensorView %s", p.name)
	}
	b.WriteString(") {\n")
	b.WriteString("    using namespace host;\n")
	b.WriteString("    SymbolicDevice device;\n")
	b.WriteString("    device.set_options<kDLCUDA>();\n")
	fmt.Fprintf(&b, "    RuntimeCheck(rows >= 0, \"%s: negative rows \", rows);\n", k.Name)

	call := []string{"(int)rows"}
	for _, p := range ps {
		n := p.name
		dt := ffiDTypes[p.dtype]
		switch {
		case p.ndim == 2 && p.indexed:
			fmt.Fprintf(&b, "    TensorMatcher({-1, %d})\n", p.cols)
			fmt.Fprintf(&b, "        .with_strides({%d, 1})\n", p.stride0)
			fmt.Fprintf(&b, "        .with_dtype<%s>()\n", dt)
			fmt.Fprintf(&b, "        .with_device(device)\n")
			fmt.Fprintf(&b, "        .verify(%s);\n", n)
		case p.ndim == 2:
			fmt.Fprintf(&b, "    SymbolicSize %s_r{\"%s rows\"}, %s_c{\"%s cols\"};\n", n, n, n, n)
			fmt.Fprintf(&b, "    TensorMatcher({%s_r, %s_c})\n", n, n)
			fmt.Fprintf(&b, "        .with_strides({%d, 1})\n", p.stride0)
			fmt.Fprintf(&b, "        .with_dtype<%s>()\n", dt)
			fmt.Fprintf(&b, "        .with_device(device)\n")
			fmt.Fprintf(&b, "        .verify(%s);\n", n)
			reqRows := fmt.Sprintf("rows * %d / %d", p.rowMult, p.rowDiv)
			if p.rowMult == 1 && p.rowDiv == 1 {
				reqRows = "rows"
			}
			fmt.Fprintf(&b, "    RuntimeCheck(%s_r.unwrap() >= %s && %s_c.unwrap() >= %d,\n",
				n, reqRows, n, p.cols)
			fmt.Fprintf(&b, "                 \"%s: %s: want at least \", %s, \"x%d\");\n",
				k.Name, n, reqRows, p.cols)
		default:
			fmt.Fprintf(&b, "    SymbolicSize %s_n{\"%s len\"};\n", n, n)
			fmt.Fprintf(&b, "    TensorMatcher({%s_n})\n", n)
			b.WriteString("        .with_strides({1})\n")
			fmt.Fprintf(&b, "        .with_dtype<%s>()\n", dt)
			b.WriteString("        .with_device(device)\n")
			fmt.Fprintf(&b, "        .verify(%s);\n", n)
			reqLen := fmt.Sprintf("%d", p.cols)
			if p.rowMult > 0 {
				reqLen = fmt.Sprintf("%d + rows * %d / %d", p.cols, p.rowMult, p.rowDiv)
			}
			fmt.Fprintf(&b, "    RuntimeCheck(%s_n.unwrap() >= %s,\n", n, reqLen)
			fmt.Fprintf(&b, "                 \"%s: %s: too short\");\n", k.Name, n)
		}
		cnst := "const "
		if p.writable {
			cnst = ""
		}
		call = append(call, fmt.Sprintf("static_cast<%s%s*>(%s.data_ptr())", cnst, p.dtype.cType(), n))
	}

	b.WriteString("    if (rows == 0) return;\n")
	if k.Cluster {
		ctaThreads, wideThreads, _, _ := k.clusterGeometry()
		maxRows := k.maxClusterRows()
		fmt.Fprintf(&b, "    dim3 grid, block;\n")
		fmt.Fprintf(&b, "    auto kernel = detail::%s;\n", k.Name)
		fmt.Fprintf(&b, "    if (rows <= %d) {\n", maxRows)
		fmt.Fprintf(&b, "      grid = dim3(%d, (unsigned)rows);\n", ClusterSize)
		fmt.Fprintf(&b, "      block = dim3(%d);\n", ctaThreads)
		fmt.Fprintf(&b, "      kernel = detail::%s_cluster;\n", k.Name)
		if k.HasWide() {
			b.WriteString("    } else {\n")
			b.WriteString("      grid = dim3((unsigned)rows);\n")
			fmt.Fprintf(&b, "      block = dim3(%d);\n", wideThreads)
			fmt.Fprintf(&b, "      kernel = detail::%s_wide;\n    }\n", k.Name)
		} else {
			b.WriteString("    } else {\n")
			b.WriteString("      grid = dim3((unsigned)rows);\n")
			b.WriteString("      block = dim3(256);\n    }\n")
		}
		fmt.Fprintf(&b, "    LaunchKernel(grid, block, device.unwrap())(\n        kernel,\n        %s);\n",
			strings.Join(call, ",\n        "))
	} else {
		threads, _, _ := k.geometry()
		if threads == 32 {
			b.WriteString("    const dim3 grid(((unsigned)rows + 7) / 8), block(256);\n")
		} else {
			b.WriteString("    const dim3 grid((unsigned)rows), block(256);\n")
		}
		fmt.Fprintf(&b, "    LaunchKernel(grid, block, device.unwrap())(\n        detail::%s,\n        %s);\n",
			k.Name, strings.Join(call, ",\n        "))
	}
	b.WriteString("  }\n};\n")
	return b.String(), nil
}

// wrapCLines wraps any line in src longer than limit characters at safe C++
// token boundaries so every line of the returned string has len <= limit.
func wrapCLines(src string, limit int) string {
	var out strings.Builder
	lines := strings.Split(src, "\n")
	for idx, line := range lines {
		if idx > 0 {
			out.WriteByte('\n')
		}
		if len(line) <= limit {
			out.WriteString(line)
			continue
		}
		indent := leadingSpaces(line)
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "//") {
			out.WriteString(wrapCommentLine(indent, trimmed, limit))
			continue
		}
		isDefine := strings.HasPrefix(trimmed, "#define ")
		cur := line
		for len(cur) > limit {
			maxPos := limit
			if isDefine {
				maxPos = limit - 2
			}
			cut := findSplitPos(cur, maxPos)
			if cut <= len(indent)+4 {
				break
			}
			head := strings.TrimRight(cur[:cut], " ")
			tail := strings.TrimLeft(cur[cut:], " ")
			if isDefine {
				out.WriteString(head + " \\\n")
			} else {
				out.WriteString(head + "\n")
			}
			contIndent := indent + "    "
			if len(contIndent)+len(tail) > limit && len(indent)+4 <= 16 {
				cur = contIndent + tail
			} else {
				cur = "        " + tail
			}
		}
		out.WriteString(cur)
	}
	return out.String()
}

func leadingSpaces(s string) string {
	i := 0
	for i < len(s) && s[i] == ' ' {
		i++
	}
	return s[:i]
}

func wrapCommentLine(indent, trimmed string, limit int) string {
	words := strings.Fields(trimmed[2:])
	var b strings.Builder
	line := indent + "//"
	for i, w := range words {
		if i > 0 && len(line)+1+len(w) > limit {
			b.WriteString(line + "\n")
			line = indent + "//"
		}
		line += " " + w
	}
	b.WriteString(line)
	return b.String()
}

// findSplitPos returns a split index <= maxPos in s outside string literals.
func findSplitPos(s string, maxPos int) int {
	inStr := false
	best := -1
	for i := 0; i < len(s) && i < maxPos; i++ {
		c := s[i]
		if c == '"' && (i == 0 || s[i-1] != '\\') {
			inStr = !inStr
			continue
		}
		if inStr {
			continue
		}
		if (c == ';' || c == ',' || c == '{') && i+1 < len(s) && s[i+1] == ' ' {
			best = i + 1
		} else if c == ' ' && i > 0 {
			prev := s[i-1]
			if prev == '+' || prev == '-' || prev == '*' || prev == '/' || prev == '|' || prev == '=' || prev == ')' {
				best = i
			}
		}
	}
	if best > 0 {
		return best
	}
	// Fallback: split before '(' if no space operator was found.
	for i := min(maxPos, len(s)-1); i > 8; i-- {
		if s[i] == '(' || s[i] == ' ' {
			return i
		}
	}
	return maxPos
}
