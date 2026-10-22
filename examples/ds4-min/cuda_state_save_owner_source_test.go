package main

import (
	"os"
	"strings"
	"testing"
)

func TestCUDAStateSaveSnapshotOwnerSource(t *testing.T) {
	data, err := os.ReadFile("cuda_state_persistence_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	actual := dsparkSourceFunction(t, string(data), "func saveCUDAState(")
	for _, name := range []string{"writeCUDAStateSnapshot", "writeCUDAStateCoreSnapshot"} {
		actual += dsparkSourceFunction(t, string(data), "func "+name+"(")
	}
	actual = strings.ReplaceAll(actual, "dsv4.ErrClosed", "errClosed")
	cleanup, err := os.ReadFile("backend_cleanup.go")
	if err != nil {
		t.Fatal(err)
	}
	start := strings.Index(string(cleanup), "// backendCleanupOwner")
	if start < 0 {
		t.Fatal("cleanup source missing")
	}
	fixture := `package main
import("bytes";"encoding/binary";"errors";"fmt";"io";"testing")
var errClosed=errors.New("closed")
var free,writeFailure=errors.New("snapshot close"),errors.New("writer")
const cudaStateHeaderBytes=104
const cudaStateMaxBytes=1<<20
const cudaStateFormatVersion=3
var cudaStateMagic [8]byte
type pool struct{n int}
type layer struct{attn,indexer pool}
type state struct{layers []layer}
type cudaStateSnapshot struct{state state;pos int;closes int;fail,closed,closing bool;logits []float32;steering identity;dspark *struct{};supportModel,supportQuant [32]byte}
func(s *cudaStateSnapshot)Close()error{s.closes++;if s.fail&&s.closes<3{return free};return nil}
type identity struct{}
func(identity)nativeStateIdentity()([32]byte,[32]byte,error){return [32]byte{},[32]byte{},nil}
func(identity)encode()([]byte,error){return nil,nil}
type runtime struct{failed error;cudaFoundation identity;logits []float32}
func(*runtime)steeringStateIdentity()identity{return identity{}}
type cudaRequestState struct{closed bool;runtime *runtime;snapshot *cudaStateSnapshot}
func(s *cudaRequestState)Snapshot()(io.Closer,error){return s.snapshot,nil}
type field struct{name string}
func cudaStateFields(*state,[]float32)[]field{return nil}
func cudaStateFieldBytes(field)([]byte,error){return nil,nil}
func writeCUDAStateField(*bytes.Buffer,field,[]byte)error{return nil}
func writeSessionFull(w io.Writer,b []byte)error{_,err:=w.Write(b);return err}
// Support encoding has a separate transaction gate. This owner gate must not
// accidentally enter it when testing the actual legacy encoder and caller.
func writeCUDASupportSnapshot(io.Writer,*cudaStateSnapshot,[32]byte,[32]byte)error{panic("unexpected support encoder")}
type failWriter struct{}
func(failWriter)Write([]byte)(int,error){return 0,writeFailure}
func TestSave(t *testing.T){
 for _,badWriter:=range []bool{false,true}{
  p:=&cudaStateSnapshot{fail:true};s:=&cudaRequestState{runtime:&runtime{},snapshot:p};var w io.Writer=&bytes.Buffer{}
  if badWriter{w=failWriter{}}
  err:=saveCUDAState(s,w)
  if !errors.Is(err,free)||badWriter&&!errors.Is(err,writeFailure)||p.closes!=1{t.Fatal("snapshot close error lost",err,p.closes)}
  var retained io.Closer
  if !errors.As(err,&retained)||retained==nil{t.Fatal("snapshot owner lost")}
  if !errors.Is(retained.Close(),free)||p.closes!=2{t.Fatal("retry ownership lost")}
  if err:=retained.Close();err!=nil||p.closes!=3{t.Fatal("retained snapshot not released",err,p.closes)}
  if err:=retained.Close();err!=nil||p.closes!=3{t.Fatal("snapshot double released")}
 }
 p:=&cudaStateSnapshot{};s:=&cudaRequestState{runtime:&runtime{},snapshot:p}
 if err:=saveCUDAState(s,&bytes.Buffer{});err!=nil||p.closes!=1{t.Fatal("successful save changed",err,p.closes)}
}
`
	fixture += string(cleanup)[start:]
	dsparkOwnerSourceGate(t, fixture+actual, "")
	mutant := strings.Replace(actual, "defer func() { err = finishBackendCleanup(err, snapshot) }()", "defer snapshot.Close()", 1)
	if mutant == actual {
		t.Fatal("close mutation missing")
	}
	dsparkOwnerSourceGate(t, fixture+mutant, "snapshot close error lost")
}
