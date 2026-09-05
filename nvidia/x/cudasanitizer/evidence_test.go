package cudasanitizer

import (
	"strings"
	"testing"
)

func TestOutputEvidenceChunks(t *testing.T) {
	patterns := []string{"needle", "tail\nhead", "absent", "世界", "\n", "", "x"}
	cfg := Config{RequireOutput: patterns, ForbidOutput: patterns}
	for _, tc := range []struct{ name, stdout, stderr string }{
		{"late matches", strings.Repeat("x", 257) + "needle世界tail", "head" + strings.Repeat("y", 129)},
		{"empty", "", ""},
		{"partial", "need", "le"},
		{"stderr", "x", strings.Repeat("y", 257) + "needle世界"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, chunk := range []int{1, 2, 3, 8, 64, 4096} {
				stdout := &capBuffer{limit: 16, evidence: newOutputEvidence(cfg)}
				stderr := &capBuffer{limit: 16, evidence: newOutputEvidence(cfg)}
				write := func(b *capBuffer, s string) {
					for len(s) > 0 {
						n := min(chunk, len(s))
						if _, err := b.Write([]byte(s[:n])); err != nil {
							t.Fatal(err)
						}
						s = s[n:]
					}
					if len(b.evidence.head) > b.evidence.limit || len(b.evidence.tail) > b.evidence.limit {
						t.Fatal("unbounded evidence retention")
					}
				}
				write(stdout, tc.stdout)
				write(stderr, tc.stderr)
				r := &Result{Config: cfg}
				r.checkEvidence(stdout, stderr)
				combined := tc.stdout + "\n" + tc.stderr
				missing, forbidden := map[string]bool{}, map[string]bool{}
				for _, s := range r.MissingEvidence {
					missing[s] = true
				}
				for _, s := range r.ForbiddenEvidence {
					forbidden[s] = true
				}
				for _, s := range patterns {
					if s == "" {
						continue
					}
					want := strings.Contains(combined, s)
					if missing[s] == want || forbidden[s] != want {
						t.Errorf("chunk %d, pattern %q: missing=%v forbidden=%v, complete-stream match=%v", chunk, s, missing[s], forbidden[s], want)
					}
				}
			}
		})
	}
}
