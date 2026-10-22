package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// Actual Restore, restoreSnapshot, feature restore and transaction bodies run
// with byte-array fake copies. This is not physical device rollback evidence.
func TestDSparkRestoreSourceTransactions(t *testing.T) {
	read := func(name string) string {
		b, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	outer := read("cuda_state_snapshot_linux.go")
	body := dsparkSourceFunction(t, outer, "func (s *cudaRequestState) Restore(") + "\n" + dsparkSourceFunction(t, outer, "func (s *cudaRequestState) restoreSnapshot(") + "\n" + dsparkSourceFunction(t, read("dspark_cuda_snapshot_linux.go"), "func (f *cudaDSparkFeatures) restore(") + "\n" + dsparkSourceFunction(t, read("cuda_state_identity_linux.go"), "func cudaStateTransaction(")
	body = strings.ReplaceAll(body, "snapshot dsv4.Snapshot", "snapshot any")
	cleanup := read("backend_cleanup.go")
	for _, mutation := range []string{"baseline", "rollback", "backup-owner"} {
		source := body
		if mutation == "rollback" {
			source = strings.Replace(source, "func() error { return s.restoreSnapshot(backup.(*cudaStateSnapshot)) }", "func() error { return nil }", 1)
		}
		if mutation == "backup-owner" {
			source = strings.Replace(source, "finishBackendCleanup(err, backup)", "errors.Join(err, backup.Close())", 1)
		}
		if mutation != "baseline" && source == body {
			t.Fatal("mutation missing", mutation)
		}
		program := `package main
import("errors";"fmt";"reflect";"testing";"io")
var dsv4=struct{ErrClosed,ErrUnimplemented,ErrDesynchronized error}{errors.New("closed"),errors.New("unimplemented"),errors.New("desynchronized")}
var copyError=errors.New("copy failed")
var closeError=errors.New("backup close failed")
var copies,failAt int
var persistentFailure,failBackup bool
type fakeSlice struct{data []int}
func(s fakeSlice)CopyDeviceToDevice(v fakeSlice)error{copies++;copy(s.data,v.data);if copies==failAt||(persistentFailure&&copies>=failAt){return copyError};return nil}
func(s fakeSlice)clone()fakeSlice{return fakeSlice{append([]int(nil),s.data...)}}
type fakeJournal struct{frontier,epoch int}
func(j *fakeJournal)clone()*fakeJournal{v:=*j;return &v}
type fakeHistory struct{values fakeSlice;journal *fakeJournal}
type cudaDSparkFeatures struct{features,status fakeSlice;history *fakeHistory;captured []bool;position,token int;generation uint64;published bool}
type cudaDSparkFeatureSnapshot struct{features,status,history fakeSlice;journal *fakeJournal;captured []bool;position,token int;generation uint64;published bool}
func(f *cudaDSparkFeatures)validateSnapshot(*cudaDSparkFeatureSnapshot,int,uint64)error{return nil}
type fakeState struct{value int}
func restoreCUDAState(dst *fakeState,src fakeState)error{*dst=src;return nil}
func validateCUDARestore(fakeState,fakeState,int,int)error{return nil}
func validateCUDAStateFrontier(int,int)error{return nil}
func syncStream(int)error{return nil}
type fakeObserver struct{invalidations int}
func(o *fakeObserver)invalidate(){o.invalidations++}
type cudaRuntime struct{state fakeState;logits []float32;pos,ctx int;failed error;headObserver fakeObserver}
func(*cudaRuntime)stream()int{return 0}
func(*cudaRuntime)steeringStateIdentity()int{return 0}
type cudaRequestState struct{runtime *cudaRuntime;closed bool;dspark *cudaDSparkFeatures;generation,epoch uint64;backup *cudaStateSnapshot}
type cudaStateSnapshot struct{owner *cudaRuntime;closed,closing bool;state fakeState;logits []float32;pos,steering int;dspark *cudaDSparkFeatureSnapshot;generation uint64;closes int}
func(v *cudaStateSnapshot)Close()error{v.closes++;if failBackup&&v.closes==1{return closeError};v.closed=true;return nil}
func(s *cudaRequestState)Snapshot()(io.Closer,error){f:=s.dspark;v:=&cudaStateSnapshot{owner:s.runtime,state:s.runtime.state,logits:append([]float32(nil),s.runtime.logits...),pos:s.runtime.pos,generation:s.generation,dspark:&cudaDSparkFeatureSnapshot{features:f.features.clone(),status:f.status.clone(),history:f.history.values.clone(),journal:f.history.journal.clone(),captured:append([]bool(nil),f.captured...),position:f.position,token:f.token,generation:f.generation,published:f.published}};s.backup=v;return v,nil}
func(s *cudaRequestState)completeObservedCacheReplacement(err error)error{if err==nil{s.epoch++;s.runtime.headObserver.invalidate()};return err}
` + source + `
func fixture()(*cudaRequestState,*cudaStateSnapshot){
 f:=&cudaDSparkFeatures{features:fakeSlice{[]int{1}},status:fakeSlice{[]int{2}},history:&fakeHistory{values:fakeSlice{[]int{3}},journal:&fakeJournal{4,1}},captured:[]bool{true},position:3,token:9,generation:3,published:true}
 s:=&cudaRequestState{runtime:&cudaRuntime{state:fakeState{4},logits:[]float32{5},pos:4,ctx:99},dspark:f,generation:4,epoch:7}
 v:=&cudaStateSnapshot{owner:s.runtime,state:fakeState{40},logits:[]float32{50},pos:5,generation:5,dspark:&cudaDSparkFeatureSnapshot{features:fakeSlice{[]int{10}},status:fakeSlice{[]int{20}},history:fakeSlice{[]int{30}},journal:&fakeJournal{5,2},captured:[]bool{true},position:4,token:10,generation:4,published:true}}
 return s,v
}
func TestRollback(t *testing.T){
 for _,failure:=range []int{1,2,3}{copies=0;failAt=failure;persistentFailure=false;failBackup=false;s,v:=fixture();err:=s.Restore(v)
 if !errors.Is(err,copyError){t.Fatal("copy error lost",err)}
 if s.runtime.state.value!=4||s.runtime.pos!=4||s.runtime.logits[0]!=5||s.generation!=4||s.dspark.position!=3||s.dspark.token!=9||!s.dspark.published||s.dspark.history.journal.frontier!=4||s.dspark.history.journal.epoch!=1||!reflect.DeepEqual(s.dspark.features.data,[]int{1})||!reflect.DeepEqual(s.dspark.status.data,[]int{2})||!reflect.DeepEqual(s.dspark.history.values.data,[]int{3})||s.epoch!=7||s.runtime.failed!=nil{t.Fatal("rollback did not restore trusted frontier")}
 }
 copies=0;failAt=2;persistentFailure=true;s,v:=fixture();err:=s.Restore(v)
 if !errors.Is(err,dsv4.ErrDesynchronized)||s.runtime.failed==nil||s.runtime.headObserver.invalidations!=1{t.Fatal("rollback failure not poisoned",err)}
 persistentFailure=false
}
func TestBackupOwner(t *testing.T){copies=0;failAt=0;failBackup=true;s,v:=fixture();err:=s.Restore(v)
 if !errors.Is(err,closeError){t.Fatal("backup close error lost",err)}
 var owner io.Closer;if !errors.As(err,&owner){t.Fatal("backup owner missing")}
 if err:=owner.Close();err!=nil||s.backup.closes!=2{t.Fatal("backup retry lost",err)}
 if err:=owner.Close();err!=nil||s.backup.closes!=2{t.Fatal("backup free repeated",err)}
 failBackup=false
}
`
		dir := t.TempDir()
		for name, data := range map[string]string{"backend_cleanup.go": cleanup, "restore_test.go": program} {
			if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0600); err != nil {
				t.Fatal(err)
			}
		}
		cmd := exec.Command("go", "test", "backend_cleanup.go", "restore_test.go", "-count=1")
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if mutation == "baseline" && err != nil {
			t.Fatalf("baseline: %v\n%s", err, out)
		}
		want := map[string]string{"rollback": "rollback did not restore trusted frontier", "backup-owner": "backup owner missing"}[mutation]
		if mutation != "baseline" && (err == nil || !strings.Contains(string(out), want)) {
			t.Fatalf("%s mutant not caught: %v\n%s", mutation, err, out)
		}
	}
}
