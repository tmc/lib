package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// Compile the actual Linux admission body on the host. The fake status buffer
// records writes, so rejection must preserve both metadata and device intent.
func TestDSparkBeginRejectionPreservesFrontier(t *testing.T) {
	data, err := os.ReadFile("dspark_cuda_features_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	body := dsparkSourceFunction(t, string(data), "func (f *cudaDSparkFeatures) begin(")
	fixture := `package fixture
import("fmt";"testing")
var Flash=struct{Vocab int}{128}
type fakeStatus struct{writes int}
func(s *fakeStatus)CopyFromHost([]uint32)error{s.writes++;return nil}
type cudaRuntime struct{pos,ctx int;failed error}
type cudaRequestState struct{generation uint64}
type cudaDSparkFeatures struct{r *cudaRuntime;request *cudaRequestState;position,token int;published bool;generation uint64;captured []bool;status fakeStatus;valid bool}
func(f *cudaDSparkFeatures)bound()bool{return f!=nil&&f.valid}
func(f *cudaDSparkFeatures)invalidate(){f.published=false;f.position=-1;f.token=-1;clear(f.captured)}
func dsparkRequestOpen(*cudaRuntime)error{return nil}
func TestAdmission(t *testing.T){
 for _,kind:=range []string{"negative-position","wrong-position","negative-token","large-token","failed-runtime","unbound","context-exhausted"}{
  f:=&cudaDSparkFeatures{r:&cudaRuntime{pos:4,ctx:8},request:&cudaRequestState{generation:7},position:3,token:11,published:true,generation:6,captured:[]bool{true,true},valid:true}
  position,token:=4,12
  switch kind{case "negative-position":position=-1;case "wrong-position":position=5;case "negative-token":token=-1;case "large-token":token=128;case "failed-runtime":f.r.failed=fmt.Errorf("failed");case "unbound":f.valid=false;case "context-exhausted":f.r.ctx=4}
  if err:=f.begin(position,token);err==nil{t.Fatal("rejected admission accepted",kind)}
  if !f.published||f.position!=3||f.token!=11||f.generation!=6||!f.captured[0]||!f.captured[1]||f.status.writes!=0{t.Fatal("rejected admission changed trusted frontier",kind)}
 }
 f:=&cudaDSparkFeatures{r:&cudaRuntime{pos:4,ctx:8},request:&cudaRequestState{generation:7},published:true,captured:[]bool{true},valid:true}
 if err:=f.begin(4,12);err!=nil||f.published||f.position!=4||f.token!=12||f.generation!=7||f.captured[0]||f.status.writes!=1{t.Fatal("valid admission not initialized",err)}
}
`
	for _, mutation := range []string{"baseline", "invalidate", "context"} {
		source := body
		want := ""
		if mutation == "invalidate" {
			needle := `return fmt.Errorf("cuda dspark features: invalid request position")`
			if strings.Count(source, needle) != 1 {
				t.Fatal("admission mutation site changed")
			}
			source = strings.Replace(source, needle, "if f != nil { f.invalidate() }; "+needle, 1)
			want = "rejected admission changed trusted frontier"
		}
		if mutation == "context" {
			source = strings.Replace(source, "position >= f.r.ctx || ", "", 1)
			if source == body {
				t.Fatal("context guard mutation site missing")
			}
			want = "rejected admission accepted context-exhausted"
		}
		dir := t.TempDir()
		for name, contents := range map[string]string{"go.mod": "module fixture\n\ngo 1.23\n", "begin_test.go": fixture + "\n" + source} {
			if err := os.WriteFile(filepath.Join(dir, name), []byte(contents), 0600); err != nil {
				t.Fatal(err)
			}
		}
		cmd := exec.Command("go", "test", "-count=1", ".")
		cmd.Dir = dir
		cmd.Env = append(os.Environ(), "GOWORK=off")
		output, err := cmd.CombinedOutput()
		if mutation == "baseline" && err != nil {
			t.Fatalf("actual admission: %v\n%s", err, output)
		}
		if mutation != "baseline" && (err == nil || !strings.Contains(string(output), want)) {
			t.Fatalf("mutation did not fail on frontier preservation: %v\n%s", err, output)
		}
	}
}
