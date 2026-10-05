//go:build linux && native

package cudaoutputcoverage_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaoutputcoverage"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func recordedOutputLaunch(arm string) (uint32, int32, error) {
	switch arm {
	case "floor":
		return 3, 0, nil
	case "corrupt":
		return 4, 1, nil
	case "healthy":
		return 4, 0, nil
	}
	return 0, 0, fmt.Errorf("unknown output launch arm %q", arm)
}
func recordedOutputSpec(grid uint32, corrupt int32) cudaoutputcoverage.TwoFill {
	input := make([]byte, 8)
	binary.LittleEndian.PutUint32(input, 1003)
	binary.LittleEndian.PutUint32(input[4:], uint32(corrupt))
	s := cudaoutputcoverage.TwoFill{Contract: cudaoutputcoverage.Contract{Buffer: "out", Elements: 1006, Domain: []cudaoutputcoverage.Range{{Start: 0, End: 1003}}, Kind: cudaoutputcoverage.AtLeastOnce, Deterministic: true}, FillA: math.Float32bits(-101), FillB: math.Float32bits(-202), Geometry: cudaoutputcoverage.Geometry{Grid: [3]uint32{grid, 1, 1}, Block: [3]uint32{256, 1, 1}}, SourceDigest: recordedHash([]byte(recordedOutputSource)), InputDigest: recordedHash(input), Target: "sm_121", MathMode: "NVRTC default finite FP32 integer conversion", Reference: make([]uint32, 1006)}
	for i := 0; i < 1003; i++ {
		s.Reference[i] = math.Float32bits(float32(i + 1))
	}
	return s
}
func recordedOutputPolicy(s cudaoutputcoverage.TwoFill) cudatest.CUDAOutputPolicy {
	return cudatest.CUDAOutputPolicy{Target: s.Target, DType: "fp32", MathMode: s.MathMode, SourceDigest: s.SourceDigest, InputDigest: s.InputDigest, Elements: 1003, NaN: "reject", Infinity: "reject", Zero: "bits"}
}

func TestRecordedOutputLaunchCPU(t *testing.T) {
	for _, tt := range []struct {
		arm     string
		grid    uint32
		corrupt int32
	}{{"floor", 3, 0}, {"corrupt", 4, 1}, {"healthy", 4, 0}} {
		grid, corrupt, err := recordedOutputLaunch(tt.arm)
		if err != nil || grid != tt.grid || corrupt != tt.corrupt {
			t.Fatalf("%s: %d %d %v", tt.arm, grid, corrupt, err)
		}
		s := recordedOutputSpec(grid, corrupt)
		if s.Contract.Elements != 1006 || s.Contract.Domain[0] != (cudaoutputcoverage.Range{Start: 0, End: 1003}) || s.Reference[1002] != math.Float32bits(1003) || s.Reference[1003] != 0 {
			t.Fatal("extent/reference contract")
		}
	}
	for _, arm := range []string{"", "Floor", "pass", "unknown"} {
		if _, _, err := recordedOutputLaunch(arm); err == nil {
			t.Fatal("unknown arm admitted")
		}
	}
}

func TestRecordedOutputSourceCPU(t *testing.T) {
	fset := token.NewFileSet()
	original, err := parser.ParseFile(fset, "output_native_test.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var originalSource string
	ast.Inspect(original, func(n ast.Node) bool {
		v, ok := n.(*ast.ValueSpec)
		if !ok {
			return true
		}
		for i, name := range v.Names {
			if name.Name == "outputSource" {
				lit, ok := v.Values[i].(*ast.BasicLit)
				if !ok {
					t.Fatal("source literal missing")
				}
				originalSource, err = strconv.Unquote(lit.Value)
				if err != nil {
					t.Fatal(err)
				}
			}
		}
		return true
	})
	if originalSource != recordedOutputSource {
		t.Fatal("worker source differs from accepted compile fixture")
	}
	worker, err := parser.ParseFile(fset, "recorded_worker_native_test.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	positions := map[string]token.Pos{}
	ast.Inspect(worker, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		name, ok := sel.X.(*ast.Ident)
		if !ok {
			return true
		}
		key := name.Name + "." + sel.Sel.Name
		if key == "cudaschedule.RunRecordedHelper" || key == "driver.Load" || key == "cuda.NewDevice" || key == "driver.ModuleLoadData" || key == "cudaschedule.WriteOutputReceipt" {
			positions[key] = call.Pos()
		}
		return true
	})
	helper := positions["cudaschedule.RunRecordedHelper"]
	for _, key := range []string{"driver.Load", "cuda.NewDevice", "driver.ModuleLoadData", "cudaschedule.WriteOutputReceipt"} {
		if helper == 0 || positions[key] <= helper {
			t.Fatalf("worker readiness missing or after %s", key)
		}
	}
}

func TestRecordedOutputImageCPU(t *testing.T) {
	dir := t.TempDir()
	regular := filepath.Join(dir, "module.ptx")
	if err := os.WriteFile(regular, []byte("literal actual supplied PTX"), 0600); err != nil {
		t.Fatal(err)
	}
	got, err := recordedImage(regular)
	if err != nil || string(got) != "literal actual supplied PTX" {
		t.Fatal(string(got), err)
	}
	link := filepath.Join(dir, "link")
	if err := os.Symlink(regular, link); err != nil {
		t.Fatal(err)
	}
	fifo := filepath.Join(dir, "fifo")
	if err := syscall.Mkfifo(fifo, 0600); err != nil {
		t.Fatal(err)
	}
	empty := filepath.Join(dir, "empty")
	if err := os.WriteFile(empty, nil, 0600); err != nil {
		t.Fatal(err)
	}
	large := filepath.Join(dir, "large")
	if err := os.WriteFile(large, []byte(strings.Repeat("x", (4<<20)+1)), 0600); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{dir, link, fifo, empty, large} {
		if _, err := recordedImage(path); err == nil {
			t.Fatalf("unsafe input admitted: %s", path)
		}
	}
}

func FuzzRecordedOutputArmCPU(f *testing.F) {
	for _, arm := range []string{"floor", "healthy", "corrupt", "unknown"} {
		f.Add(arm)
	}
	f.Fuzz(func(t *testing.T, arm string) {
		grid, corrupt, err := recordedOutputLaunch(arm)
		if arm != "floor" && arm != "healthy" && arm != "corrupt" {
			if err == nil {
				t.Fatal("unlisted arm admitted")
			}
			return
		}
		if err != nil || grid < 3 || grid > 4 || corrupt < 0 || corrupt > 1 {
			t.Fatal("closed arm violated")
		}
	})
}

// Both option and prepared metadata files use one strict bounded reader.
func recordedReadJSON(path string, target any) error {
	b, err := recordedFile(path, 1<<20)
	if err != nil {
		return err
	}
	typ := reflect.TypeOf(target)
	if typ == nil || typ.Kind() != reflect.Pointer || typ.Elem().Kind() != reflect.Struct {
		return fmt.Errorf("recorded output: JSON target must be a struct pointer")
	}
	allowed := make(map[string]bool)
	for i := 0; i < typ.Elem().NumField(); i++ {
		field := typ.Elem().Field(i)
		if field.PkgPath == "" {
			allowed[field.Name] = true
		}
	}
	scan := json.NewDecoder(bytes.NewReader(b))
	scan.UseNumber()
	if err := recordedJSONValue(scan, allowed, 0); err != nil {
		return err
	}
	if _, err := scan.Token(); err != io.EOF {
		return fmt.Errorf("recorded output: trailing JSON")
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(target); err != nil {
		return err
	}
	if _, err := dec.Token(); err != io.EOF {
		return fmt.Errorf("recorded output: trailing JSON")
	}
	return nil
}
func recordedJSONValue(dec *json.Decoder, allowed map[string]bool, depth int) error {
	if depth > 32 {
		return fmt.Errorf("recorded output: JSON nesting exceeds limit")
	}
	tok, err := dec.Token()
	if err != nil {
		return err
	}
	delim, container := tok.(json.Delim)
	if depth == 0 && (!container || delim != '{') {
		return fmt.Errorf("recorded output: JSON object required")
	}
	if !container {
		return nil
	}
	switch delim {
	case '{':
		seen := make(map[string]bool)
		for dec.More() {
			key, err := dec.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			if !ok {
				return fmt.Errorf("recorded output: JSON key required")
			}
			folded := strings.ToLower(name)
			if seen[folded] {
				return fmt.Errorf("recorded output: duplicate JSON key %q", name)
			}
			seen[folded] = true
			if depth == 0 && !allowed[name] {
				return fmt.Errorf("recorded output: unknown JSON key %q", name)
			}
			if err := recordedJSONValue(dec, nil, depth+1); err != nil {
				return err
			}
		}
		end, err := dec.Token()
		if err != nil {
			return err
		}
		if end != json.Delim('}') {
			return fmt.Errorf("recorded output: JSON object terminator")
		}
	case '[':
		for dec.More() {
			if err := recordedJSONValue(dec, nil, depth+1); err != nil {
				return err
			}
		}
		end, err := dec.Token()
		if err != nil {
			return err
		}
		if end != json.Delim(']') {
			return fmt.Errorf("recorded output: JSON array terminator")
		}
	default:
		return fmt.Errorf("recorded output: invalid JSON delimiter")
	}
	return nil
}

func TestRecordedOutputJSONCPU(t *testing.T) {
	dir := t.TempDir()
	for _, kind := range []string{"options", "prepared"} {
		t.Run(kind, func(t *testing.T) {
			var target any
			var good, key string
			if kind == "options" {
				target = &struct{ RequireOutput bool }{}
				good = `{"RequireOutput":true}`
				key = "RequireOutput"
			} else {
				target = &struct{ PTXDigest string }{}
				good = `{"PTXDigest":"actual"}`
				key = "PTXDigest"
			}
			p := filepath.Join(dir, kind+".json")
			if err := os.WriteFile(p, []byte(good), 0600); err != nil {
				t.Fatal(err)
			}
			if err := recordedReadJSON(p, target); err != nil {
				t.Fatal(err)
			}
			for _, tt := range []struct{ name, body string }{
				{"duplicate", strings.TrimSuffix(good, "}") + `,"` + key + `":null}`},
				{"casealias", strings.TrimSuffix(good, "}") + `,"` + strings.ToLower(key) + `":null}`},
				{"escapedduplicate", strings.TrimSuffix(good, "}") + `,"` + fmt.Sprintf(`\u%04x%s`, key[0], key[1:]) + `":null}`},
				{"trailing", good + ` {}`}, {"unknown", `{"Unknown":true}`}, {"array", `[]`}, {"null", `null`},
				{"hiddenoverflow", good + strings.Repeat(" ", 1<<20) + `{}`},
				{"oversize-whitespace", good + strings.Repeat(" ", (1<<20)+1-len(good))},
			} {
				t.Run(tt.name, func(t *testing.T) {
					if err := os.WriteFile(p, []byte(tt.body), 0600); err != nil {
						t.Fatal(err)
					}
					if err := recordedReadJSON(p, target); err == nil {
						t.Fatal("malformed metadata admitted")
					}
				})
			}
			fifo := filepath.Join(dir, kind+"-fifo")
			if err := syscall.Mkfifo(fifo, 0600); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(p, []byte(good), 0600); err != nil {
				t.Fatal(err)
			}
			link := filepath.Join(dir, kind+"-link")
			if err := os.Symlink(p, link); err != nil {
				t.Fatal(err)
			}
			if err := recordedReadJSON(link, target); err == nil {
				t.Fatal("symlink metadata admitted")
			}
			if err := recordedReadJSON(fifo, target); err == nil {
				t.Fatal("FIFO metadata admitted")
			}
		})
	}
}

func FuzzRecordedOutputJSONCPU(f *testing.F) {
	for _, s := range []string{`{"RequireOutput":true}`, `{"RequireOutput":false,"RequireOutput":true}`, `{"Args":["literal"]}`, `[]`} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		if len(s) > 4096 {
			t.Skip()
		}
		p := filepath.Join(t.TempDir(), "metadata.json")
		if err := os.WriteFile(p, []byte(s), 0600); err != nil {
			t.Fatal(err)
		}
		var target struct {
			RequireOutput bool
			Args          []string
		}
		if err := recordedReadJSON(p, &target); err != nil {
			return
		}
		b, err := json.Marshal(target)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, b, 0600); err != nil {
			t.Fatal(err)
		}
		var again struct {
			RequireOutput bool
			Args          []string
		}
		if err := recordedReadJSON(p, &again); err != nil || !reflect.DeepEqual(target, again) {
			t.Fatal("accepted metadata round trip differs", err)
		}
	})
}

// Actual testbinary controller refusal, with deliberately relative packet and
// artifact paths so even the old parser cannot proceed to RunRecorded/CUDA.
func TestRecordedOutputControllerAdmissionCPU(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	for _, tt := range []struct{ name, body, want string }{
		{"duplicate", `{"RequireOutput":false,"RequireOutput":true}`, "duplicate JSON key"},
		{"oversized", `{"RequireOutput":true}` + strings.Repeat(" ", 1<<20) + `{}`, "file size/type"},
		{"fifo", "", "file size/type"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(dir, tt.name)
			if tt.name == "fifo" {
				if err := syscall.Mkfifo(path, 0600); err != nil {
					t.Fatal(err)
				}
			} else if err := os.WriteFile(path, []byte(tt.body), 0600); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, exe, "-test.run=^TestRecordedOutputNativeConsumer$")
			cmd.Env = append(os.Environ(), "RECORDED_OUTPUT_NATIVE=1", "RECORDED_OUTPUT_OPTIONS="+path, "RECORDED_OUTPUT_PACKET=relative")
			out, err := cmd.CombinedOutput()
			if ctx.Err() != nil || err == nil || !bytes.Contains(out, []byte(tt.want)) {
				t.Fatalf("actual controller refusal missing: err=%v ctx=%v output=%s", err, ctx.Err(), out)
			}
			t.Logf("CPU controller actual child refusal %s: %s", tt.name, out)
		})
	}
}

type recordedCompiledOutput struct{ Name, SourceDigest, PTXDigest string }
type recordedCompilerRegistry struct {
	Arch, NVRTCPath        string
	NVRTCMajor, NVRTCMinor int32
	Outputs                []recordedCompiledOutput
}

func recordedOutputSources() map[string]string {
	return map[string]string{
		"healthy": recordedOutputSource,
		"floor":   strings.Replace(recordedOutputSource, "if (i < (unsigned int)n)", "if (i < (unsigned int)n && i < 768)", 1),
		"corrupt": strings.Replace(recordedOutputSource, "corrupt && i == 29", "i == 29", 1),
	}
}
func recordedRegisteredOutput(dir string, loaded []byte) (recordedCompiledOutput, error) {
	if !filepath.IsAbs(dir) {
		return recordedCompiledOutput{}, fmt.Errorf("absolute registry required")
	}
	var registry recordedCompilerRegistry
	if err := recordedReadJSON(filepath.Join(dir, "registry.json"), &registry); err != nil {
		return recordedCompiledOutput{}, err
	}
	if registry.Arch != "sm_121" || registry.NVRTCPath == "" || registry.NVRTCMajor <= 0 || len(registry.Outputs) != 3 {
		return recordedCompiledOutput{}, fmt.Errorf("registered compiler/target inventory differs")
	}
	wanted := recordedOutputSources()
	names := []string{"healthy", "floor", "corrupt"}
	seen := make(map[string]bool)
	var selected recordedCompiledOutput
	for i, row := range registry.Outputs {
		if row.Name != names[i] || seen[row.PTXDigest] {
			return recordedCompiledOutput{}, fmt.Errorf("registered output inventory differs")
		}
		seen[row.PTXDigest] = true
		source, err := recordedImage(filepath.Join(dir, row.Name+".cu"))
		if err != nil {
			return recordedCompiledOutput{}, err
		}
		ptx, err := recordedImage(filepath.Join(dir, row.Name+".ptx"))
		if err != nil {
			return recordedCompiledOutput{}, err
		}
		if string(source) != wanted[row.Name] || row.SourceDigest != recordedHash(source) || row.PTXDigest != recordedHash(ptx) {
			return recordedCompiledOutput{}, fmt.Errorf("registered actual source/PTX body differs")
		}
		if row.PTXDigest == recordedHash(loaded) {
			selected = row
		}
	}
	if selected.Name == "" {
		return recordedCompiledOutput{}, fmt.Errorf("actual supplied PTX is not registered")
	}
	return selected, nil
}

func TestRecordedOutputRegistryCPU(t *testing.T) {
	dir := t.TempDir()
	sources := recordedOutputSources()
	registry := recordedCompilerRegistry{Arch: "sm_121", NVRTCPath: "/CPU-synthetic-NVRTC", NVRTCMajor: 13, NVRTCMinor: 4}
	// CPU-authored PTX strings test byte binding only, not device semantics.
	images := make(map[string][]byte)
	for _, name := range []string{"healthy", "floor", "corrupt"} {
		image := []byte("CPU synthetic actual " + name + " PTX body")
		images[name] = image
		source := []byte(sources[name])
		if err := os.WriteFile(filepath.Join(dir, name+".cu"), source, 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name+".ptx"), image, 0600); err != nil {
			t.Fatal(err)
		}
		registry.Outputs = append(registry.Outputs, recordedCompiledOutput{Name: name, SourceDigest: recordedHash(source), PTXDigest: recordedHash(image)})
	}
	write := func(r recordedCompilerRegistry) {
		b, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "registry.json"), b, 0600); err != nil {
			t.Fatal(err)
		}
	}
	write(registry)
	for _, name := range []string{"healthy", "floor", "corrupt"} {
		got, err := recordedRegisteredOutput(dir, images[name])
		if err != nil || got.Name != name || got.SourceDigest != recordedHash([]byte(sources[name])) {
			t.Fatal("loaded source binding", got, err)
		}
	}
	if _, err := recordedRegisteredOutput(dir, []byte("foreign supplied PTX")); err == nil {
		t.Fatal("unknown supplied PTX accepted")
	}
	for _, alteration := range []string{"ptx-digest", "source-digest", "duplicate", "target", "source-body", "consistent-source-body", "ptx-body"} {
		t.Run(alteration, func(t *testing.T) {
			changed := registry
			changed.Outputs = append([]recordedCompiledOutput(nil), registry.Outputs...)
			switch alteration {
			case "ptx-digest":
				changed.Outputs[1].PTXDigest = strings.Repeat("a", 64)
			case "source-digest":
				changed.Outputs[1].SourceDigest = strings.Repeat("b", 64)
			case "duplicate":
				changed.Outputs[1].PTXDigest = changed.Outputs[0].PTXDigest
			case "target":
				changed.Arch = "sm_90"
			case "source-body":
				if err := os.WriteFile(filepath.Join(dir, "floor.cu"), []byte("different"), 0600); err != nil {
					t.Fatal(err)
				}
			case "consistent-source-body":
				foreign := []byte(strings.Replace(sources["floor"], "i < 768", "i < 700", 1))
				if err := os.WriteFile(filepath.Join(dir, "floor.cu"), foreign, 0600); err != nil {
					t.Fatal(err)
				}
				changed.Outputs[1].SourceDigest = recordedHash(foreign)
			case "ptx-body":
				if err := os.WriteFile(filepath.Join(dir, "floor.ptx"), []byte("different"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			write(changed)
			if _, err := recordedRegisteredOutput(dir, images["floor"]); err == nil {
				t.Fatal("altered registry accepted")
			}
			if err := os.WriteFile(filepath.Join(dir, "floor.cu"), []byte(sources["floor"]), 0600); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, "floor.ptx"), images["floor"], 0600); err != nil {
				t.Fatal(err)
			}
			write(registry)
		})
	}
	if !strings.Contains(sources["floor"], "&& i < 768") || strings.Contains(sources["corrupt"], "corrupt && i == 29") {
		t.Fatal("source encoded defects absent")
	}
}

// Reads the actual compile-only three-source packet; it does not initialize CUDA.
func TestRecordedOutputCompiledRegistryCPU(t *testing.T) {
	dir := os.Getenv("RECORDED_OUTPUT_COMPILED_REGISTRY_DIR")
	if dir == "" {
		t.Skip("actual compile-only registry not selected")
	}
	for _, name := range []string{"healthy", "floor", "corrupt"} {
		image, err := recordedImage(filepath.Join(dir, name+".ptx"))
		if err != nil {
			t.Fatal(err)
		}
		got, err := recordedRegisteredOutput(dir, image)
		if err != nil || got.Name != name || got.SourceDigest != recordedHash([]byte(recordedOutputSources()[name])) {
			t.Fatal("actual compiler body/source binding differs", got, err)
		}
		t.Logf("CPU actual compiled %s PTX=%s source=%s bytes=%d", name, got.PTXDigest, got.SourceDigest, len(image))
	}
}

func TestRecordedOutputStopCPU(t *testing.T) {
	f, err := parser.ParseFile(token.NewFileSet(), "recorded_controller_native_test.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	seen := make(map[string]bool)
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || (fn.Name.Name != "TestRecordedOutputNativeConsumer" && fn.Name.Name != "TestRecordedOutputBaselineCandidateNative") {
			continue
		}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			stmt, ok := n.(*ast.IfStmt)
			if !ok {
				return true
			}
			not, ok := stmt.Cond.(*ast.UnaryExpr)
			if !ok || not.Op != token.NOT {
				return true
			}
			call, ok := not.X.(*ast.CallExpr)
			if !ok {
				return true
			}
			sel, ok := call.Fun.(*ast.SelectorExpr)
			if !ok || sel.Sel.Name != "Run" {
				return true
			}
			receiver, ok := sel.X.(*ast.Ident)
			if !ok || receiver.Name != "t" {
				return true
			}
			if len(stmt.Body.List) != 1 {
				t.Fatal("failed attempt must immediately return")
			}
			if _, ok := stmt.Body.List[0].(*ast.ReturnStmt); !ok {
				t.Fatal("failed attempt advances loop")
			}
			seen[fn.Name.Name] = true
			return true
		})
	}
	if len(seen) != 2 {
		t.Fatal("native attempt loops do not stop on failed subtest")
	}
}
