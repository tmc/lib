package cudasanitizer_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudasanitizer"
)

// TestAdditionalCheckersCUDA uses the maintained testdata/checkers.cu fixture.
func TestAdditionalCheckersCUDA(t *testing.T) {
	prog := hardwareProg(t, "CUDASANITIZER_TEST_CHECKERS_PROG")
	for _, tool := range []cudasanitizer.Tool{cudasanitizer.Initcheck, cudasanitizer.Synccheck} {
		for _, mode := range []string{"clean", "bad"} {
			t.Run(string(tool)+"/"+mode, func(t *testing.T) {
				marker := string(tool) + " " + mode + " launch submitted"
				required := []string{marker}
				if mode == "clean" {
					required = append(required, string(tool)+" clean completed")
				}
				res, err := cudasanitizer.Run(context.Background(), cudasanitizer.Config{
					Tool: tool, MissingTool: cudasanitizer.MissingToolFail,
					RequireCUDAInit: cudasanitizer.ToggleYes,
					RequireOutput:   required, Timeout: 45 * time.Second, LogDir: t.TempDir(),
				}, prog, string(tool), mode)
				if err != nil {
					t.Fatal(err)
				}
				t.Log(res.Details())
				// Sanitizer errors outrank missing output in Run's classification, so
				// explicitly require execution evidence for the known-bad case too.
				if len(res.MissingEvidence) != 0 || !strings.Contains(res.Stdout, marker) {
					t.Fatalf("missing execution evidence: %v", res.MissingEvidence)
				}
				want := cudasanitizer.OutcomeClean
				if mode == "bad" {
					want = cudasanitizer.OutcomeSanitizerErrors
				}
				if res.Outcome != want {
					t.Fatalf("outcome %v, want %v", res.Outcome, want)
				}
				if mode == "bad" {
					finding := "Uninitialized __global__ memory read"
					if tool == cudasanitizer.Synccheck {
						finding = "Barrier error detected"
					}
					found := false
					for _, f := range res.Log.Findings {
						if strings.Contains(f.Headline, finding) {
							found = true
						}
					}
					if !res.Log.SummaryFound || res.Log.Errors == 0 || !found {
						t.Fatalf("missing expected %s finding and nonzero summary", finding)
					}
				}
			})
		}
	}
}
