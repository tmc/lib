package cudarepro

import (
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// This controlled CPU executable exercises the protocol and actual arithmetic;
// it neither loads CUDA nor qualifies the worker's device/oracle assertions.
const cpuExportRunner = `package main
import("crypto/sha256";"encoding/json";"fmt";"os";"strings";"time")
func main(){
 var req struct{Version,Attempt int;Kernel []byte;Case json.RawMessage;OracleRequired bool;OraclePatch string}
 if err:=json.NewDecoder(os.Stdin).Decode(&req);err!=nil{panic(err)}
 var c struct{Inputs []uint64}
 if err:=json.Unmarshal(req.Case,&c);err!=nil{panic(err)}
 ref:=make([]uint64,len(c.Inputs));got:=make([]uint64,len(c.Inputs))
 for i,v:=range c.Inputs{ref[i]=v+uint64(i)+1;got[i]=ref[i]}
 mode:=os.Getenv("EXPORT_CPU_MODE")
 if mode=="timeout"{time.Sleep(time.Minute)}
 if mode=="overflow"{fmt.Print(strings.Repeat("x",(1<<20)+1));return}
 if !strings.Contains(string(req.Kernel),"FIXED") && len(got)>0 {got[0]++}
 if mode=="drift" && len(got)>1{got[0]=ref[0];got[1]++}
 pid:=os.Getpid();if mode=="pid"{pid++}
 kd:=fmt.Sprintf("%x",sha256.Sum256(req.Kernel));if mode=="digest"{kd=strings.Repeat("0",64)}
 oracle:=map[string]any{"Attached":mode!="oracle","PID":os.Getpid(),"Patch":req.OraclePatch,"Evidence":"cpu-protocol-control"}
 out:=map[string]any{"Version":1,"PID":pid,"KernelDigest":kd,"CaseDigest":fmt.Sprintf("%x",sha256.Sum256(req.Case)),"Reference":ref,"Observed":got,"Oracle":oracle}
 if mode=="arrays"{out["Observed"]=[]uint64{}}
 json.NewEncoder(os.Stdout).Encode(out)
 if mode=="trailing"{fmt.Println("{}")}
}
`

func setExportRunner(t *testing.T, opts *ExportOptions) {
	t.Helper()
	dir := testTempDir(t)
	source := filepath.Join(dir, "runner.go")
	binary := filepath.Join(dir, "runner")
	if err := os.WriteFile(source, []byte(cpuExportRunner), 0600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("go", "build", "-o", binary, source)
	cmd.Env = append(os.Environ(), "GOWORK=off")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build runner: %v\n%s", err, out)
	}
	b, err := os.ReadFile(binary)
	if err != nil {
		t.Fatal(err)
	}
	opts.RunnerPath = binary
	opts.RunnerSHA256 = fmt.Sprintf("%x", sha256.Sum256(b))
}

func TestExportRunnerSnapshot(t *testing.T) {
	finding, r, opts := sampleExportSetup()
	setExportRunner(t, &opts)
	dir := filepath.Join(testTempDir(t), "reproducer")
	if err := ExportPackage(dir, finding, r, opts); err != nil {
		t.Fatal(err)
	}
	source := fmt.Sprintf(`package reproducer
import("bytes";"encoding/json";"os";"os/exec";"testing")
func TestRunnerSnapshot(t *testing.T){
 copied,err:=copyRunner(%q,t.TempDir(),%q)
 if err!=nil{t.Fatal(err)}
 if err:=os.WriteFile(%q,[]byte("replaced original"),0700);err!=nil{t.Fatal(err)}
 req,_:=json.Marshal(runnerRequest{Version:1,Attempt:1,Kernel:[]byte("FIXED"),Case:json.RawMessage(%q)})
 cmd:=exec.Command(copied)
 cmd.Stdin=bytes.NewReader(req)
 out,err:=cmd.Output()
 if err!=nil{t.Fatalf("execute snapshot: %%v",err)}
 var response runnerResponse
 if err:=json.Unmarshal(out,&response);err!=nil{t.Fatal(err)}
 if response.PID!=cmd.Process.Pid || len(response.Observed)!=1 || response.Observed[0]!=42{t.Fatalf("snapshot response = %%s",out)}
}
`, opts.RunnerPath, opts.RunnerSHA256, opts.RunnerPath, `{"Inputs":[41]}`)
	if err := os.WriteFile(filepath.Join(dir, "snapshot_test.go"), []byte(source), 0600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("go", "test", "-count=1", "-run", "^TestRunnerSnapshot$")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOWORK=off", "EXPORT_CPU_MODE=")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("snapshot control: %v\n%s", err, out)
	}
}
