package nccltest

import (
	"strings"
	"testing"
)

const referenceOutput = `# nccl-tests version 2.20.0 nccl-library=23102
# nThread 1 nGpus 1 minBytes 65536 maxBytes 65536 step: 1(bytes) warmup iters: 5 iters: 20 agg iters: 1 validation: 2 graph: 0 unalign: 0
# size count type redop root time algbw busbw wrong time algbw busbw wrong
65536 16384 float sum -1 2.62 25.01 0.00 0 0.09 706.21 0.00 N/A
# Out of bounds values : 0 OK
# Collective test concluded: all_reduce_perf
`

func TestParseReference(t *testing.T) {
	rows, err := ParseReference(strings.NewReader(referenceOutput))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0].Count != 16384 || !rows[0].OutOfPlace.Checked || rows[0].InPlace.Checked || rows[0].Config != (ReferenceConfig{1, 1, 5, 20, 1, 2, 0, 0}) {
		t.Fatalf("got %+v", rows)
	}
	for _, tc := range []string{
		"", strings.ReplaceAll(referenceOutput, "agg iters: 1", "agg iters: 0"), strings.ReplaceAll(referenceOutput, "0 OK", "1 FAILED"),
		strings.ReplaceAll(referenceOutput, "# Collective test concluded:", "# absent:"),
		strings.ReplaceAll(referenceOutput, "2.62", "NaN"),
		strings.ReplaceAll(referenceOutput, "65536 16384", "65536 0"),
	} {
		if _, err := ParseReference(strings.NewReader(tc)); err == nil {
			t.Errorf("accepted incomplete/invalid output %q", tc)
		}
	}
}
func TestAllToAllVPattern(t *testing.T) {
	c := Case{Operation: AllToAllV, Bytes: 64}
	l, err := c.Layout(4)
	if err != nil {
		t.Fatal(err)
	}
	if l.SendCount != 12 || l.RecvCount != 12 || l.ParamCount != 4 {
		t.Fatalf("got %+v", l)
	}
	if got := expectedFloat[float32](c, l, 4, 2, 0); got != 7 {
		t.Errorf("first sender = %g", got)
	}
	if got := expectedFloat[float32](c, l, 4, 2, 4); got != 7 {
		t.Errorf("last sender = %g", got)
	}
	for rank := 0; rank < 4; rank++ {
		sent, received := 0, 0
		for peer := 0; peer < 4; peer++ {
			sent += peerElements(4, 4, rank, peer, 4)
			received += peerElements(4, 4, peer, rank, 4)
		}
		if sent != 12 || received != 12 {
			t.Fatalf("imbalanced rank %d", rank)
		}
	}
}

func TestReferenceEmptyReduction(t *testing.T) {
	out := strings.Replace(referenceOutput, "float sum -1", "float -1", 1)
	rows, err := ParseReference(strings.NewReader(out))
	if err != nil || len(rows) != 1 || rows[0].Reduction != "" {
		t.Fatalf("got %+v, %v", rows, err)
	}
}

func TestReferenceMalformedMeasurements(t *testing.T) {
	lines := strings.Split(referenceOutput, "\n")
	for _, column := range []int{5, 6, 7, 9, 10, 11} {
		for _, value := range []string{"NaN", "+Inf", "-Inf", "-1", "1e999", "bad"} {
			fields := strings.Fields(lines[3])
			fields[column] = value
			mutated := append([]string(nil), lines...)
			mutated[3] = strings.Join(fields, " ")
			if _, err := ParseReference(strings.NewReader(strings.Join(mutated, "\n"))); err == nil {
				t.Fatalf("accepted column %d value %q", column, value)
			}
		}
	}
	for _, column := range []int{8, 12} {
		for _, value := range []string{"-1", "0.5", "18446744073709551616", "bad"} {
			fields := strings.Fields(lines[3])
			fields[column] = value
			mutated := append([]string(nil), lines...)
			mutated[3] = strings.Join(fields, " ")
			if _, err := ParseReference(strings.NewReader(strings.Join(mutated, "\n"))); err == nil {
				t.Fatalf("accepted correctness column %d value %q", column, value)
			}
		}
	}
}

func FuzzParseReference(f *testing.F) {
	f.Add(referenceOutput)
	f.Add("")
	f.Add(strings.Replace(referenceOutput, "float sum -1", "float -1", 1))
	f.Fuzz(func(t *testing.T, input string) {
		rows, err := ParseReference(strings.NewReader(input))
		if err != nil {
			return
		}
		if len(rows) == 0 {
			t.Fatal("accepted empty result")
		}
		for _, row := range rows {
			if row.Count <= 0 || row.NCCLVersion <= 0 {
				t.Fatalf("accepted invalid row: %+v", row)
			}
		}
	})
}

func TestReferenceGraphControls(t *testing.T) {
	for _, value := range []string{"0", "3", "-1", "bad", "999999999999999999999999"} {
		rows, err := ParseReference(strings.NewReader(strings.ReplaceAll(referenceOutput, "graph: 0", "graph: "+value)))
		if value == "0" || value == "3" {
			if err != nil || len(rows) != 1 {
				t.Fatalf("graph %s: %v", value, err)
			}
			if (value == "3") != (rows[0].Config.GraphLaunches == 3) {
				t.Fatal("graph count lost")
			}
		} else if err == nil {
			t.Fatalf("accepted graph %s", value)
		}
	}
}

func TestReferenceOffset(t *testing.T) {
	rows, err := ParseReference(strings.NewReader(strings.ReplaceAll(referenceOutput, "unalign: 0", "unalign: 3")))
	if err != nil || len(rows) != 1 || rows[0].Config.Offset != 3 {
		t.Fatalf("offset lost: %v %v", rows, err)
	}
	for _, value := range []string{"-1", "bad", "999999999999999999999999"} {
		if _, err := ParseReference(strings.NewReader(strings.ReplaceAll(referenceOutput, "unalign: 0", "unalign: "+value))); err == nil {
			t.Fatalf("accepted offset %s", value)
		}
	}
}

func TestReferenceRejectsDuplicateControls(t *testing.T) {
	banner := strings.Split(referenceOutput, "\n")[1]
	for _, second := range []string{banner, strings.ReplaceAll(banner, "unalign: 0", "unalign: 3")} {
		input := strings.Replace(referenceOutput, banner, banner+"\n"+second, 1)
		if _, err := ParseReference(strings.NewReader(input)); err == nil {
			t.Fatal("accepted duplicate controls")
		}
	}
}

func TestReferenceRepeatedRows(t *testing.T) {
	line := "65536 16384 float sum -1 2.62 25.01 0.00 0 0.09 706.21 0.00 N/A"
	input := strings.Replace(referenceOutput, line, line+"\n"+strings.Replace(line, "2.62", "3.14", 1), 1)
	rows, err := ParseReference(strings.NewReader(input))
	if err != nil || len(rows) != 2 {
		t.Fatalf("rows=%v error=%v", rows, err)
	}
	if rows[0].OutOfPlace.Microseconds != 2.62 || rows[1].OutOfPlace.Microseconds != 3.14 {
		t.Fatal("cycle order lost")
	}
}

func TestReferenceSectionOrder(t *testing.T) {
	row := "65536 16384 float sum -1 2.62 25.01 0.00 0 0.09 706.21 0.00 N/A\n"
	summary := "# Out of bounds values : 0 OK\n"
	complete := "# Collective test concluded: all_reduce_perf\n"
	for name, input := range map[string]string{
		"early summary":        summary + referenceOutput,
		"early completion":     complete + referenceOutput,
		"duplicate summary":    strings.Replace(referenceOutput, summary, summary+summary, 1),
		"duplicate completion": referenceOutput + complete,
		"late row":             referenceOutput + row,
		"early row":            row + referenceOutput,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := ParseReference(strings.NewReader(input)); err == nil {
				t.Fatal("accepted misplaced result section")
			}
		})
	}
}
