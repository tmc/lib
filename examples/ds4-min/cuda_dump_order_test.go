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
	"strings"
	"testing"
)

// TestCUDADumpOrdering compiles actual dump bodies with host-only device
// primitives. Removing either production synchronization must fail the same
// ordinary sequencing/error tests. This is not device qualification.
func TestCUDADumpOrdering(t *testing.T) {
	for _, mutant := range []string{"", "dumpBatchLayerSlice", "dumpRawKVSlot"} {
		t.Run("remove-"+mutant, func(t *testing.T) {
			set := token.NewFileSet()
			file, err := parser.ParseFile(set, "cuda_session_linux.go", nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			var source bytes.Buffer
			source.WriteString("package main\nimport (\"fmt\";\"os\";\"strconv\")\n")
			found, removed := 0, 0
			for _, decl := range file.Decls {
				fn, ok := decl.(*ast.FuncDecl)
				if !ok || (fn.Name.Name != "dumpBatchLayerSlice" && fn.Name.Name != "dumpRawKVSlot" && fn.Name.Name != "dumpPosition") {
					continue
				}
				found++
				if fn.Name.Name != "dumpPosition" {
					fn.Type.Params.List[0].Type = ast.NewIdent("fakeSlice")
				}
				if fn.Name.Name == mutant {
					var body []ast.Stmt
					for _, stmt := range fn.Body.List {
						var b bytes.Buffer
						if err := format.Node(&b, set, stmt); err != nil {
							t.Fatal(err)
						}
						if bytes.Contains(b.Bytes(), []byte("syncStream(r.stream())")) {
							removed++
							continue
						}
						body = append(body, stmt)
					}
					fn.Body.List = body
				}
				if err := format.Node(&source, set, fn); err != nil {
					t.Fatal(err)
				}
				source.WriteByte('\n')
			}
			if found != 3 || (mutant != "" && removed != 1) {
				t.Fatalf("found %d functions, removed %d sync statements", found, removed)
			}
			dir := t.TempDir()
			for name, data := range map[string][]byte{"go.mod": []byte("module dumporder\ngo 1.25\n"), "subject.go": source.Bytes(), "subject_test.go": []byte(cudaDumpHostTests)} {
				if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
					t.Fatal(err)
				}
			}
			cmd := exec.Command("go", "test", "-count=1", ".")
			cmd.Dir = dir
			cmd.Env = append(os.Environ(), "GOWORK=off", "CGO_ENABLED=0")
			out, err := cmd.CombinedOutput()
			if mutant == "" {
				if err != nil {
					t.Fatalf("ordinary gate: %v\n%s", err, out)
				}
			} else if err == nil || !strings.Contains(string(out), "order [copy]") {
				t.Fatalf("removal mutant must fail ordinary ordering assertion: %v\n%s", err, out)
			}
		})
	}
}

const cudaDumpHostTests = `package main
import("encoding/binary";"errors";"math";"os";"path/filepath";"reflect";"testing")
const cudaKVWidth=512
type fakeStream struct{order []string; err error}
type cudaRuntime struct{work *fakeStream}
func(r *cudaRuntime)stream()*fakeStream{return r.work}
func syncStream(s *fakeStream)error{s.order=append(s.order,"sync");return s.err}
type fakeSlice struct{n int; work *fakeStream; err error}
func(s fakeSlice)Len()int{return s.n}
func(s fakeSlice)Slice(a,b int)fakeSlice{return fakeSlice{b-a,s.work,s.err}}
func(s fakeSlice)CopyToHost(dst []float32)error{s.work.order=append(s.work.order,"copy");for i:=range dst{dst[i]=1};return s.err}
func float32Bytes(v []float32)[]byte{b:=make([]byte,len(v)*4);for i,x:=range v{binary.LittleEndian.PutUint32(b[i*4:],math.Float32bits(x))};return b}
func TestDump(t *testing.T){for _,method:=range []string{"layer","raw"}{for _,mode:=range []string{"order","sync-error","copy-error","disabled","empty","other-position","malformed-position"}{t.Run(method+"/"+mode,func(t *testing.T){
dir:=t.TempDir();prefix:=filepath.Join(dir,"dump");pos:="2048";n:=512;s:=&fakeStream{};copyErr:=error(nil);want:=errors.New("primitive failed")
switch mode{case "sync-error":s.err=want;case "copy-error":copyErr=want;case "disabled":prefix="";case "empty":n=0;case "other-position":pos="2047";case "malformed-position":pos="bad"}
t.Setenv("DS4_DUMP_LAYERS",prefix);t.Setenv("DS4_DUMP_POSITION",pos);r:=&cudaRuntime{s};src:=fakeSlice{n,s,copyErr}
var err error;if method=="layer"{err=r.dumpBatchLayerSlice(src,1,2048,"hc_ffn_post")}else{err=r.dumpRawKVSlot(src,1,2048)}
var order []string;switch mode{case "order","copy-error":order=[]string{"sync","copy"};case "sync-error":order=[]string{"sync"}}
if !reflect.DeepEqual(s.order,order){t.Fatalf("order %v, want %v",s.order,order)}
if mode=="sync-error"||mode=="copy-error"{if !errors.Is(err,want){t.Fatalf("error %v",err)}}else if mode=="malformed-position"{if err==nil{t.Fatal("missing position error")}}else if err!=nil{t.Fatal(err)}
files,_:=os.ReadDir(dir);wantFiles:=0;if mode=="order"{wantFiles=1};if len(files)!=wantFiles{t.Fatalf("files %d, want %d",len(files),wantFiles)}
})}}}
`
