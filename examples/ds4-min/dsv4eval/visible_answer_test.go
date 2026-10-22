package dsv4eval

import (
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func ExampleSmokeReport_AgreementPass() {
	r := SmokeReport{Cases: 12, Ran: 12, StatusMatches: 11, ThinkCloseMatches: 12}
	fmt.Println(r.AgreementPass())
	// Output: true
}

func TestAcceptedLinesPinnedC(t *testing.T) {
	source := os.Getenv("DS4_ORACLE_SOURCE")
	if source == "" {
		t.Skip("set DS4_ORACLE_SOURCE to pinned upstream")
	}
	b, err := os.ReadFile(filepath.Join(source, "ds4_eval.c"))
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(b)) != "4092ce91692e99b6cbb846401f075969362de5d2ea62662513979e18480f9a60" {
		t.Fatal("unqualified source")
	}
	c := "#include <stdio.h>\n#include <stdlib.h>\n#include <stdbool.h>\n#include <ctype.h>\n"
	for _, signature := range []string{"static bool parse_line_spec(", "static bool compsec_answer_matches("} {
		i := strings.Index(string(b), signature)
		if i < 0 {
			t.Fatal("missing helper")
		}
		j := strings.Index(string(b)[i:], "\n}")
		if j < 0 {
			t.Fatal("unterminated helper")
		}
		c += string(b)[i:i+j+2] + "\n"
	}
	c += "int main(int argc,char **argv) { if(argc!=3)return 2; printf(\"%d\\n\",compsec_answer_matches(argv[1],argv[2]));return 0;}\n"
	dir := t.TempDir()
	path := filepath.Join(dir, "parser.c")
	binary := filepath.Join(dir, "parser")
	if err := os.WriteFile(path, []byte(c), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("cc", path, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("cc: %v: %s", err, out)
	}
	for _, tt := range []struct{ name, expected, picked string }{
		{"empty-expected", "", "3"}, {"empty-picked", "3", ""},
		{"reversed", "4-3", "3"}, {"255", "255", "255"},
		{"256", "256", "256"}, {"clamp", "255-256", "255"},
		{"outside", "3-4", "5"}, {"malformed", "3-x", "0"},
		{"bare-range", "3-", "0-3"}, {"negative-end", "3--2", "0"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			out, err := exec.Command(binary, tt.expected, tt.picked).Output()
			if err != nil {
				t.Fatal(err)
			}
			want := strings.TrimSpace(string(out)) == "1"
			if got := acceptedLines(tt.expected, tt.picked); got != want {
				t.Fatalf("got %v, actual C %v", got, want)
			}
		})
	}
}

// Literal C0 terminal fragments, raw.trace SHA-256
// 2e432af8660ee22704817b4bd15d3c884d4ad4b9b7adb6d7047814d96f942779.
// Upstream 11e8cad ds4_eval.c:2972 and :3161 define visibility and Juliet sets.
func TestGradeUpstreamVisibleAnswers(t *testing.T) {
	tests := []struct {
		name           string
		c              Case
		output, picked string
		complete, pass bool
	}{
		{"C0-choice", Case{Kind: Choice, Answer: "J"}, "Need final exactly one line.</think>Answer: J", "J", true, true},
		{"C0-Juliet", Case{Source: "NIST Juliet", Kind: LineSet, Answer: "3-4"}, "So answer line 3.</think>Answer: 3", "3", true, true},
		{"hidden-answer-mutation", Case{Kind: Choice, Answer: "J"}, "Answer: J</think>no visible answer", "", false, false},
		{"chosen-token-mutation", Case{Kind: Choice, Answer: "J"}, "reason</think>Answer: A", "A", true, false},
		{"outside-range-mutation", Case{Source: "NIST Juliet", Kind: LineSet, Answer: "3-4"}, "reason</think>Answer: 3,5", "3,5", true, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			picked, complete, pass := Grade(tt.c, tt.output)
			if picked != tt.picked || complete != tt.complete || pass != tt.pass {
				t.Fatalf("got %q/%v/%v, want %q/%v/%v", picked, complete, pass, tt.picked, tt.complete, tt.pass)
			}
		})
	}
}

func TestSmokeAgreementSeparateFromQuality(t *testing.T) {
	r := SmokeReport{Cases: 12, Ran: 12, StatusMatches: 12, ThinkCloseMatches: 12, LeftPassed: 8, RightPassed: 8}
	if !r.AgreementPass() || r.Pass() {
		t.Fatal("agreement and quality conflated")
	}
	r.Ran, r.Skipped = 8, 4
	if r.AgreementPass() {
		t.Fatal("incomplete run qualified")
	}
	r.Ran, r.Skipped, r.StatusMatches = 12, 0, 10
	if r.AgreementPass() {
		t.Fatal("status mutation qualified")
	}
}
