package main

import (
	"bytes"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// Compile the actual CLI sampling-through-stop statements with a host-only
// session. Model loading and device execution are outside this host gate.
func TestSampleTranscriptCallSite(t *testing.T) {
	set := token.NewFileSet()
	file, err := parser.ParseFile(set, "main.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var loop *ast.RangeStmt
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "run" {
			continue
		}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			r, ok := n.(*ast.RangeStmt)
			if !ok {
				return true
			}
			var b bytes.Buffer
			if err := format.Node(&b, set, r); err != nil {
				t.Fatal(err)
			}
			if strings.Contains(b.String(), "id = smp.Sample(s.logits, cfg.sampling, &rng)") {
				if loop != nil {
					t.Fatal("ambiguous sampling call site")
				}
				loop = r
			}
			return true
		})
	}
	if loop == nil {
		t.Fatal("missing sampling call site")
	}
	stop := -1
	for i, stmt := range loop.Body.List {
		b, ok := stmt.(*ast.IfStmt)
		if !ok {
			continue
		}
		var text bytes.Buffer
		if err := format.Node(&text, set, b.Cond); err != nil {
			t.Fatal(err)
		}
		if text.String() == "tok.IsStopForMode(id, cfg.thinkMode)" {
			stop = i
			break
		}
	}
	if stop < 0 {
		t.Fatal("missing production stop boundary")
	}
	loop.Body.List = loop.Body.List[:stop+1]
	var body bytes.Buffer
	if err := format.Node(&body, set, loop); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	write := func(name string, data []byte) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(filepath.Join(dir, name)), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", []byte("module hostcall\ngo 1.27\n"))
	for _, name := range []string{
		"sampler.go", "budget.go", "probabilities.go",
		"exp.go", "exp_linux_arm64.go", "exp_other.go",
		"temperature_linux_arm64.go", "temperature_other.go",
		"LICENSE.arm-expf",
	} {
		data, err := os.ReadFile(filepath.Join("sampler", name))
		if err != nil {
			t.Fatal(err)
		}
		write(filepath.Join("sampler", name), data)
	}
	data, err := os.ReadFile("sampled_transcript.go")
	if err != nil {
		t.Fatal(err)
	}
	write("sampled_transcript.go", data)
	write("subject.go", []byte(callSitePrefix+body.String()+"\nreturn nil\n}\n"))
	write("subject_test.go", []byte(callSiteTests))
	goTool, err := sampleTranscriptGoTool()
	if err != nil {
		t.Fatalf("find Go tool: %v", err)
	}
	cmd := exec.Command(goTool, "test", "-mod=readonly", "-count=1", "-timeout=30s", "-v", ".")
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("production call-site gate: %v\n%s", err, out)
	}
}

func sampleTranscriptGoTool() (string, error) {
	if root := runtime.GOROOT(); root != "" {
		return exec.LookPath(filepath.Join(root, "bin", "go"))
	}
	return exec.LookPath("go")
}

func TestSampleTranscriptGoToolUnsetGOROOT(t *testing.T) {
	t.Setenv("GOROOT", "")
	// An empty PATH directory rules out accidentally finding the tool there.
	t.Setenv("PATH", t.TempDir())
	tool, err := sampleTranscriptGoTool()
	if err != nil {
		t.Fatal(err)
	}
	if !filepath.IsAbs(tool) {
		t.Fatalf("Go tool is not absolute: %q", tool)
	}
	info, err := os.Stat(tool)
	if err != nil || info.IsDir() {
		t.Fatalf("Go tool is not a file: %q: %v", tool, err)
	}
}

const callSitePrefix = `package main
import("hostcall/sampler")
type cfgType struct { tokens int; decodeLogitsDir, dumpLogprobs, sampledTraceDir string; sampling sampler.Params; thinkMode, logprobsTopK int }
type session struct { logits []float32 }
type tokenizer struct{}
func(tokenizer)IsStopForMode(int,int)bool{return true}
type logprobStep struct{ Step,Selected int; Scores []sampler.TokenScore }
func dumpDecodeLogits(string,int,[]float32)error{return nil}
func writeSampledDecision(string,int,int,uint64,uint64,bool,[]int,[]float32)error{return nil}
func subject(transcript *sampleTranscript)error{
cfg:=cfgType{tokens:3,sampling:sampler.Default}
s:=session{logits:[]float32{1,2,3}};tok:=tokenizer{}
smp:=sampler.New(3);rng:=sampler.RNG{State:12345}
var logprobSteps []logprobStep;_ = logprobSteps
prompt:=[]int{7};var generatedIDs []int;_ = prompt;_ = generatedIDs
`

const callSiteTests = `package main
import("encoding/json";"os";"path/filepath";"testing";"hostcall/sampler")
func TestActualStopDecision(t *testing.T){
path:=filepath.Join(t.TempDir(),"decisions.jsonl");x,err:=newSampleTranscript(path);if err!=nil{t.Fatal(err)}
if err:=subject(x);err!=nil{t.Fatal(err)};if err:=x.Close();err!=nil{t.Fatal(err)}
data,err:=os.ReadFile(path);if err!=nil{t.Fatal(err)}
var got map[string]interface{}
if err:=json.Unmarshal(data,&got);err!=nil{t.Fatalf("missing actual stop decision: %v",err)}
rng:=sampler.RNG{State:12345};s:=sampler.New(3);want:=s.Sample([]float32{1,2,3},sampler.Default,&rng)
// Decode integers independently; float64 cannot represent all uint64 RNG states.
var raw map[string]json.RawMessage;if err:=json.Unmarshal(data,&raw);err!=nil{t.Fatal(err)}
var before,after uint64;if err:=json.Unmarshal(raw["rng_before"],&before);err!=nil{t.Fatal(err)};if err:=json.Unmarshal(raw["rng_after"],&after);err!=nil{t.Fatal(err)}
if got["step"]!=float64(0)||got["token"]!=float64(want)||before!=12345||after!=rng.State||got["stop"]!=true{t.Fatalf("actual stop decision mismatch: %s",data)}
}
`
