package dsv4eval

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// This is a controller oracle, not a model or token-ranking oracle. Its rank
// stub supplies the known reachable rank from thinkingRuntime's live logits.
func TestG7PinnedSoftController(t *testing.T) {
	path := os.Getenv("DS4_G7_EVAL_SOURCE")
	if path == "" {
		t.Skip("DS4_G7_EVAL_SOURCE not set")
	}
	source, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(source)) != "4092ce91692e99b6cbb846401f075969362de5d2ea62662513979e18480f9a60" {
		t.Fatal("upstream evaluator source pin changed")
	}
	start := bytes.Index(source, []byte("        if (generation_in_think && think_close_tokens.len > 0) {"))
	if start < 0 {
		t.Fatal("literal controller not found")
	}
	end := bytes.Index(source[start:], []byte("        if (token < 0)\n"))
	if end < 0 {
		t.Fatal("literal controller boundary not found")
	}
	controller := string(source[start : start+end])
	program := `#include <stdio.h>
#include <stdlib.h>
enum { EVAL_THINK_CLOSE_NONE, EVAL_THINK_CLOSE_SOFT, EVAL_THINK_CLOSE_HARD };
typedef int eval_think_close_kind;
int token_rank_in_top(void *session,int token,int max_rank){return max_rank>=2?2:0;}
int main(int argc,char **argv){
struct {int soft_limit_reply_budget,hard_limit_reply_budget,soft_limit_think_close_rank;} config={atoi(argv[1]),2,3},*cfg=&config;
int ids[]={1};struct {int len;int *v;} think_close_tokens={1,ids};
int forced_close_pos=-1,generation_in_think=1,closed=0,answered=0;void *session=NULL;
for(int generated=0;generated<8;generated++){
int remaining_budget=8-generated,close_rank=0,token=-1;eval_think_close_kind close_kind=EVAL_THINK_CLOSE_NONE;
` + controller + `
if(token<0)token=closed?(answered?3:2):0;
printf("%d %s\n",token,close_kind==EVAL_THINK_CLOSE_SOFT?"soft_forced":close_kind==EVAL_THINK_CLOSE_HARD?"hard_forced":"none");
if(token==3)break;
if(token==1){closed=1;generation_in_think=0;}
if(token==2)answered=1;
}return 0;}`
	dir := t.TempDir()
	file, bin := filepath.Join(dir, "controller.c"), filepath.Join(dir, "controller")
	if err := os.WriteFile(file, []byte(program), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("cc", "-std=c11", "-O0", file, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("compile literal controller: %v %s", err, out)
	}
	for _, soft := range []int{3, 2} {
		want, err := exec.Command(bin, strconv.Itoa(soft)).Output()
		if err != nil {
			t.Fatal(err)
		}
		var first CaseArtifacts
		if err := RunSmoke(new(bytes.Buffer), new(thinkingRuntime), runtimeCases(), RunOptions{Build: "fixture", ModelSHA256: strings.Repeat("a", 64), Context: 32, Soft: soft, Hard: 2, Rank: 3, Capture: func(c CaseArtifacts) error {
			if c.Index == 1 {
				first = c
			}
			return nil
		}}); err != nil {
			t.Fatal(err)
		}
		var got strings.Builder
		for _, d := range first.Decisions {
			fmt.Fprintf(&got, "%d %s\n", d.Chosen, d.Forced)
		}
		if got.String() != string(want) {
			t.Fatalf("soft=%d actual RunSmoke decisions differ from literal upstream controller\ngot %s\nwant %s", soft, got.String(), want)
		}
	}
	t.Logf("literal controller sha256=%x; both reachable soft and hard accepted decision sequences match", sha256.Sum256([]byte(controller)))
}

// Replaying C0 against itself protects diagnostic import and completeness. It
// is not the missing independently executed physical C-C repeat.
func TestG7RetainedC0CannotQualifyAsCC(t *testing.T) {
	root := os.Getenv("DS4_G7_C0_DIR")
	if root == "" {
		t.Skip("DS4_G7_C0_DIR not set")
	}
	raw, err := os.ReadFile(filepath.Join(root, "raw.trace"))
	if err != nil {
		t.Fatal(err)
	}
	attested, err := os.ReadFile(filepath.Join(root, "attested-visible.trace"))
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(raw)) != "2e432af8660ee22704817b4bd15d3c884d4ad4b9b7adb6d7047814d96f942779" || fmt.Sprintf("%x", sha256.Sum256(attested)) != "17d977087dab284843602049bf81bb32a8a2d64336ac707ef302887b99cefbf1" {
		t.Fatal("retained C0 fixture changed")
	}
	marker := []byte("===== CASE ")
	if !bytes.Equal(raw[bytes.Index(raw, marker):], attested[bytes.Index(attested, marker):]) {
		t.Fatal("attestation changed raw case/summary suffix")
	}
	r, err := CompareSmoke(bytes.NewReader(attested), bytes.NewReader(attested))
	if err != nil {
		t.Fatal(err)
	}
	if r.Cases != 12 || r.Ran != 8 || r.Skipped != 4 || r.StatusMatches != 12 || r.ThinkCloseMatches != 12 || r.LeftPassed != 8 || r.RightPassed != 8 || r.Pass() || r.AgreementPass() {
		t.Fatalf("diagnostic C0 replay qualified or lost denominator: %+v", r)
	}
	changed := bytes.Replace(attested, []byte("think_close: natural"), []byte("think_close: soft_forced"), 1)
	r, err = CompareSmoke(bytes.NewReader(attested), bytes.NewReader(changed))
	if err != nil || r.ThinkCloseMatches != 11 || r.Pass() || r.AgreementPass() {
		t.Fatalf("thinking-close mutation accepted: %+v %v", r, err)
	}
	t.Log("actual retained C0 replay:12 diagnostic cases,8 ran/4 skipped,12 status/close matches; agreement and quality remain false")
}

func TestG7QualityAndIncompleteBoundaries(t *testing.T) {
	base := SmokeReport{Cases: 12, Ran: 12, StatusMatches: 11, ThinkCloseMatches: 12, LeftPassed: 11, RightPassed: 11}
	for _, tc := range []struct {
		name               string
		change             func(*SmokeReport)
		agreement, quality bool
	}{
		{"complete", func(*SmokeReport) {}, true, true},
		{"incomplete", func(r *SmokeReport) { r.Ran = 11; r.Skipped = 1 }, false, false},
		{"left-quality", func(r *SmokeReport) { r.LeftPassed = 10 }, true, false},
		{"right-quality", func(r *SmokeReport) { r.RightPassed = 10 }, true, false},
		{"status", func(r *SmokeReport) { r.StatusMatches = 10 }, false, false},
		{"close", func(r *SmokeReport) { r.ThinkCloseMatches = 11 }, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := base
			tc.change(&r)
			if r.AgreementPass() != tc.agreement || r.Pass() != tc.quality {
				t.Fatalf("qualification boundary differs: %+v", r)
			}
		})
	}
}
