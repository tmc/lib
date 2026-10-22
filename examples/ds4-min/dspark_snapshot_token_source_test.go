package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestPublishedSnapshotJournalToken(t *testing.T) {
	data, err := os.ReadFile("dspark_cuda_snapshot_linux.go")
	if err != nil {
		t.Fatal(err)
	}
	actual := dsparkSourceFunction(t, string(data), "func (f *cudaDSparkFeatures) validateSnapshot(")
	journal, err := os.ReadFile("dspark_feature_journal.go")
	if err != nil {
		t.Fatal(err)
	}
	fixture := `package main
import("fmt";"slices";"testing")
var Flash=struct{Vocab,SlidingWindow int}{128,128}
type fakeSlice int
func(s fakeSlice)Len()int{return int(s)}
type cudaRuntime struct{ctx int;dspark *cudaDSparkFeatures}
type cudaDSparkHistory struct{values fakeSlice;journal *dsparkFeatureJournal}
type cudaDSparkFeatures struct{r *cudaRuntime;features,status fakeSlice;targets []int;captured []bool;modelSHA string;history *cudaDSparkHistory}
func(f *cudaDSparkFeatures)bound()bool{return f!=nil}
type cudaDSparkFeatureSnapshot struct{owner *cudaDSparkFeatures;features,status,history fakeSlice;targets []int;captured []bool;modelSHA string;generation uint64;position,token int;published bool;journal *dsparkFeatureJournal}
func TestMismatch(t *testing.T){
 j,err:=newDSparkFeatureJournal(4,1);if err!=nil{t.Fatal(err)}
 if err:=j.append(0,11,1);err!=nil{t.Fatal(err)}
 f:=&cudaDSparkFeatures{r:&cudaRuntime{ctx:4},features:1,status:1,targets:[]int{1},captured:[]bool{true},modelSHA:"sealed",history:&cudaDSparkHistory{values:4,journal:j}}
 v:=&cudaDSparkFeatureSnapshot{owner:f,features:1,status:1,history:4,targets:[]int{1},captured:[]bool{true},modelSHA:"sealed",generation:0,position:0,token:11,published:true,journal:j.clone()}
 if err:=f.validateSnapshot(v,1,1);err!=nil{t.Fatal("valid snapshot refused",err)}
 v.token=12
 if err:=f.validateSnapshot(v,1,1);err==nil{t.Fatal("published token differs from valid journal and is admitted")}
}
`
	guard := "\n\t\tif f.history != nil {\n\t\t\trow, err := v.journal.row(v.position, v.journal.epoch)\n\t\t\tif err != nil || row.token != v.token {\n\t\t\t\treturn fmt.Errorf(\"cuda dspark restore: published token and journal mismatch\")\n\t\t\t}\n\t\t}"
	mutant := strings.Replace(actual, guard, "", 1)
	if mutant == actual {
		t.Fatal("snapshot token guard missing")
	}
	for _, lane := range []struct{ name, source, want string }{
		{"baseline", actual, ""},
		{"deleted-token-guard", mutant, "published token differs from valid journal and is admitted"},
	} {
		t.Run(lane.name, func(t *testing.T) {
			dir := t.TempDir()
			for name, data := range map[string][]byte{"probe_test.go": []byte(fixture + lane.source), "journal.go": journal} {
				if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
					t.Fatal(err)
				}
			}
			cmd := exec.Command("go", "test", "-count=1", "-v", "probe_test.go", "journal.go")
			cmd.Dir = dir
			output, err := cmd.CombinedOutput()
			t.Logf("actual validator and actual journal output:\n%s", output)
			if lane.want == "" {
				if err != nil {
					t.Fatalf("proposed fix failed: %v", err)
				}
			} else if err == nil || !strings.Contains(string(output), lane.want) {
				t.Fatalf("journal disagreement not reproduced: %v", err)
			}
		})
	}
}
