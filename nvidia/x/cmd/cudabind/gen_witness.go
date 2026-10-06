package main

import (
	"fmt"
	"strconv"
	"strings"
)

// meaningfulBytes excludes record padding from the sentinel oracle. Padding
// is not a value and a compiler need not preserve it across a parameter copy.
func meaningfulBytes(typ string, width int64, records []*record) ([]int, error) {
	if strings.HasSuffix(typ, "]byte") {
		return nil, fmt.Errorf("sentinel ABI witness: opaque mapped type %s passed by value is unsupported", typ)
	}
	if n, elem, ok := witnessArray(typ); ok {
		if n <= 0 || width%n != 0 {
			return nil, fmt.Errorf("unsupported witness array %s", typ)
		}
		mask, err := meaningfulBytes(elem, width/n, records)
		if err != nil {
			return nil, err
		}
		var out []int
		for i := int64(0); i < n; i++ {
			for _, j := range mask {
				out = append(out, int(i*(width/n))+j)
			}
		}
		return out, nil
	}
	for _, r := range records {
		if r.goName != typ {
			continue
		}
		if r.opaque {
			return nil, fmt.Errorf("sentinel ABI witness: opaque by-value type %s is unsupported", typ)
		}
		var out []int
		for _, f := range r.fields {
			if f.cName == "" {
				continue
			}
			bytes, err := meaningfulBytes(f.typ, f.size, records)
			if err != nil {
				return nil, err
			}
			for _, j := range bytes {
				out = append(out, int(f.off)+j)
			}
		}
		return out, nil
	}
	if width <= 0 || width > 4096 {
		return nil, fmt.Errorf("sentinel ABI witness: type %s has unsupported width %d", typ, width)
	}
	out := make([]int, width)
	for i := range out {
		out[i] = i
	}
	return out, nil
}

func witnessArray(typ string) (int64, string, bool) {
	if !strings.HasPrefix(typ, "[") {
		return 0, "", false
	}
	n, elem, ok := strings.Cut(typ[1:], "]")
	if !ok {
		return 0, "", false
	}
	count, err := strconv.ParseInt(n, 10, 64)
	return count, elem, err == nil
}

func (g *gen) sentinelSpecial(expr, typ string, records []*record, moduleName string) {
	if typ == "driver.DevicePtr" {
		g.p("%s = %sSentinelPointer(t,dev)", expr, moduleName)
		return
	}
	if typ == "bool" {
		g.p("%s = true", expr)
		return
	}
	if n, elem, ok := witnessArray(typ); ok {
		for i := int64(0); i < n; i++ {
			g.sentinelSpecial(fmt.Sprintf("%s[%d]", expr, i), elem, records, moduleName)
		}
		return
	}
	for _, r := range records {
		if r.goName == typ {
			for _, f := range r.fields {
				if f.cName != "" {
					g.sentinelSpecial(expr+"."+f.goName, f.typ, records, moduleName)
				}
			}
			return
		}
	}
}

func (g *gen) parameterWitness(cfg *config, m *module, compileName, newName string) error {
	for ki, k := range m.kernels {
		if len(k.params) > 64 {
			return fmt.Errorf("kernel %s exceeds 64 witness parameters", k.name)
		}
		var declarations, args []string
		var masks [][]int
		total := 0
		for i, p := range k.params {
			if p.width <= 0 || p.width > 4096 {
				return fmt.Errorf("kernel %s: parameter %s has unsupported witness width %d", k.name, p.cName, p.width)
			}
			typ := p.typ
			if p.slice {
				typ = "driver.DevicePtr"
			}
			mask, err := meaningfulBytes(typ, p.width, m.records)
			if err != nil {
				return fmt.Errorf("kernel %s: parameter %s: %w", k.name, p.cName, err)
			}
			if len(mask) == 0 {
				return fmt.Errorf("kernel %s: parameter %s has no sentinel value bytes", k.name, p.cName)
			}
			masks = append(masks, mask)
			declarations = append(declarations, fmt.Sprintf("%s arg%d", p.cType, i))
			args = append(args, fmt.Sprintf("unsafe.Pointer(&arg%d)", i))
			total += 1 + int(p.width)
		}
		if total > 16384 {
			return fmt.Errorf("kernel %s exceeds bounded witness output", k.name)
		}
		if total == 0 {
			continue
		}
		g.p("func %sParameterSchema%d() (cudakernel.Schema,error) { return cudakernel.NewSchema(", cfg.typeName, ki)
		for _, p := range k.params {
			g.p("cudakernel.Param{Name:%q,Type:%q,ByteWidth:%d,Alignment:%d,IsPointer:%v},", p.cName, p.cType, p.width, p.abiAlignment, p.slice)
		}
		g.p(") }")
		g.p("func Test%sParameterABI%d(t *testing.T) {", upperFirst(cfg.typeName), ki)
		g.nativeGuard()
		g.p("dev := cudatest.Require(t)")
		g.p("t.Cleanup(func() { if err := dev.Close(); err != nil { t.Error(err) } })")
		g.p("if _,err := %sParameterSchema%d(); err != nil { t.Fatal(err) }", cfg.typeName, ki)
		g.p("const parameterSource = `")
		g.p("extern \"C\" __global__ void cudabind_parameter_%d(unsigned long long* out, %s) {", ki, strings.Join(declarations, ", "))
		base := 0
		for i, p := range k.params {
			g.p("out[%d] = sizeof(arg%d);", base, i)
			g.p("for (unsigned int j = 0; j < sizeof(arg%d) && j < %d; ++j) out[%d+j] = ((const unsigned char*)&arg%d)[j];", i, p.width, base+1, i)
			base += 1 + int(p.width)
		}
		g.p("}")
		g.p("`")
		g.p("cm, err := %s(dev, %sLayoutSource()+parameterSource); if err != nil { t.Fatal(err) }", compileName, cfg.typeName)
		g.p("m, output, err := %sLoadWitnessModule(t, dev, cm); if err != nil { t.Fatal(err) }; defer func() { if err := m.Close(); err != nil { t.Error(err) } }()", cfg.typeName)
		g.p("if err := %sCheckLayout(%sReadLayout(t,dev,m)); err != nil {t.Fatal(err)}", cfg.typeName, cfg.typeName)
		g.p("kernel, err := m.mod.Function(%q); if err != nil { t.Fatal(err) }", fmt.Sprintf("cudabind_parameter_%d", ki))
		g.p("if observer := %sWitnessCompilationObserver; observer != nil { observer(t, dev, kernel.Raw(), %sLayoutSource()+parameterSource, nil, cm, output) }", cfg.typeName, cfg.typeName)
		g.p("out, err := cudamem.Alloc[uint64](dev, %d); defer func() { if err := out.Free(); err != nil { t.Error(err) } }(); if err != nil { t.Fatal(err) }", total)
		g.p("want := make([]uint64, %d)", total)
		base = 0
		for i, p := range k.params {
			typ := p.typ
			if p.slice {
				typ = "driver.DevicePtr"
			}
			g.p("var arg%d %s", i, typ)
			g.p("bytes%d := unsafe.Slice((*byte)(unsafe.Pointer(&arg%d)), int(unsafe.Sizeof(arg%d)))", i, i, i)
			g.p("for j := range bytes%d { bytes%d[j] = %d }", i, i, 33+i%16)
			if typ == "bool" {
				g.p("arg%d = %v", i, i%2 == 0)
			} else {
				g.sentinelSpecial(fmt.Sprintf("arg%d", i), typ, m.records, cfg.typeName)
			}
			g.p("want[%d] = uint64(len(bytes%d))", base, i)
			for _, j := range masks[i] {
				g.p("want[%d] = uint64(bytes%d[%d])", base+1+j, i, j)
			}
			base += 1 + int(p.width)
		}
		g.p("if err := out.CopyFromHost(make([]uint64, len(want))); err != nil { t.Fatal(err) }")
		g.p("pointer := out.Ptr(); stream := dev.DefaultStream()")
		g.p("err = stream.WithHandle(func(h driver.Stream) error { return kernel.Launch([3]uint32{1,1,1}, [3]uint32{1,1,1}, 0, h, []unsafe.Pointer{unsafe.Pointer(&pointer), %s}) }); if err != nil { t.Fatal(err) }", strings.Join(args, ", "))
		g.p("if err := stream.Synchronize(t.Context()); err != nil { t.Fatal(err) }")
		g.p("got := make([]uint64, len(want)); if err := out.CopyToHost(got); err != nil { t.Fatal(err) }")
		g.p("receipt, err := json.Marshal(struct { Kind, Kernel string; Observed, Expected []uint64 }{%q, %q, got, want}); if err != nil { t.Fatal(err) }; t.Logf(\"native-readback %%s\", receipt)", "parameter-abi", k.name)
		base = 0
		for i, p := range k.params {
			g.p("if got[%d] != want[%d] { t.Fatalf(%q, got[%d], want[%d]) }", base, base, k.name+" argument "+p.cName+" size = %d; Go has %d", base, base)
			for _, j := range masks[i] {
				g.p("if got[%d] != want[%d] { t.Fatalf(%q, got[%d], want[%d]) }", base+1+j, base+1+j, fmt.Sprintf("%s argument %s byte %d = %%d; Go has %%d", k.name, p.cName, j), base+1+j, base+1+j)
			}
			base += 1 + int(p.width)
		}
		g.p("}")
	}
	return nil
}
