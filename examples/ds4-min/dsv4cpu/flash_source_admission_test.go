package dsv4cpu

import (
	"strings"
	"testing"
)

func TestOracleFunctionSourceBoundaries(t *testing.T) {
	body := "static float target(float x) { return x * 3; }"
	for _, source := range []string{body, "\n" + body, "static float target(float x);\nstatic int neighbour(void) { return 8; }\n" + body, "\nstatic float target(float x);\n" + body, "static float caller(float x) { if (target(x)) { return 9; } return 0; }\n" + body} {
		if got := oracleFunction(t, source, "target"); got != body {
			t.Fatalf("definition differs: %q", got)
		}
	}
}

func TestFlashLiteralSourceAdmission(t *testing.T) {
	source := string(pinnedCPUOracle(t))
	for _, tc := range []struct{ name, marker string }{
		{"cpu_directional_steering_enabled", "return dirs && scale != 0.0f;"},
		{"cpu_directional_steering_project_rows", "xr[i] -= coeff * dir[i];"},
		{"silu", "return x * sigmoid_stable(x);"},
		{"layer_shared_ffn_one", "matvec_q8_0_pair_prequant"},
	} {
		// The generic extractor must skip the declarations in the full blob.
		body := oracleFunction(t, source, tc.name)
		if !strings.Contains(body, tc.marker) {
			t.Fatalf("%s extracted neighbour or declaration", tc.name)
		}
		if oracleFunction(t, body, tc.name) != body || oracleFunction(t, "\n"+body, tc.name) != body {
			t.Fatal("scoped extraction differs", tc.name)
		}
	}
}
