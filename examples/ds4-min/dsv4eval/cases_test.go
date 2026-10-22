package dsv4eval

import (
	"bytes"
	"strings"
	"testing"
)

func TestCasesTable(t *testing.T) {
	cs := Cases()
	if err := Validate(cs); err != nil {
		t.Fatal(err)
	}
	hard, err := Select(cs, Filter{Suite: Hard})
	if err != nil || len(hard) != 50 {
		t.Fatalf("hard = %d, %v", len(hard), err)
	}
	smoke, err := Select(cs, Filter{Suite: HardSmoke})
	if err != nil || len(smoke) != 12 {
		t.Fatalf("hard-smoke = %d, %v", len(smoke), err)
	}
	if _, err := Select(cs, Filter{ID: "missing"}); err == nil {
		t.Fatal("missing selection accepted")
	}
	physics, err := SelectN(cs, Filter{Source: "gpqa diamond", Domain: "physics"}, 2)
	if err != nil || len(physics) != 2 {
		t.Fatalf("filtered selection = %d, %v", len(physics), err)
	}
	if _, err := SelectN(cs, Filter{Source: "GPQA Diamond"}, 100); err == nil || !strings.Contains(err.Error(), "requested 100") {
		t.Fatalf("over-limit selection error = %v", err)
	}
	if _, err := SelectN(cs, Filter{}, -1); err == nil {
		t.Fatal("negative question count accepted")
	}
	var b bytes.Buffer
	if err := WriteList(&b, cs); err != nil {
		t.Fatal(err)
	}
	if got := len(strings.Split(strings.TrimSpace(b.String()), "\n")); got != 142 {
		t.Fatalf("list rows = %d", got)
	}
}

func TestGradeUsesLastAnswer(t *testing.T) {
	c := Case{Kind: Choice, Answer: "B", Aliases: []string{"beta"}}
	if picked, complete, pass := Grade(c, "Answer: A\nreason\nAnswer: B\n"); picked != "B" || !complete || !pass {
		t.Fatalf("grade = %q %v %v", picked, complete, pass)
	}
	if _, complete, _ := Grade(c, "no answer"); complete {
		t.Fatal("incomplete output marked complete")
	}
	if _, _, pass := Grade(Case{Kind: Integer, Answer: "70"}, "Answer: 070"); !pass {
		t.Fatal("integer normalization failed")
	}
	if _, _, pass := Grade(Case{Kind: LineSet, Answer: "a,b"}, "Answer: b,a"); !pass {
		t.Fatal("line-set normalization failed")
	}
}

func TestSelfTestExtractors(t *testing.T) {
	if err := SelfTestExtractors(); err != nil {
		t.Fatal(err)
	}
}

func TestValidateMutation(t *testing.T) {
	cs := Cases()
	cs = append([]Case(nil), cs[:len(cs)-1]...)
	if err := Validate(cs); err == nil || !strings.Contains(err.Error(), "142") {
		t.Fatalf("truncated table error = %v", err)
	}
	cs = Cases()
	cs[1].ID = cs[0].ID
	if err := Validate(cs); err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("duplicate table error = %v", err)
	}
}

func TestHardSmokeIdentityMutation(t *testing.T) {
	cs := Cases()
	for i := range cs {
		cs[i].Suites &^= HardSmoke
	}
	if err := Validate(cs); err == nil || !strings.Contains(err.Error(), "hard-smoke") {
		t.Fatalf("missing hard-smoke identities accepted: %v", err)
	}
}
