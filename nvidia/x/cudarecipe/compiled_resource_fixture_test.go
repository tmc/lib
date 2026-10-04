package cudarecipe

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudalayer"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

const compiledMLPEntry = "compiled_mlp64"

// Caps are fixed before tools run; no adaptive unbounded search is permitted.
var compiledMLPCaps = []uint64{0, 64, 48, 32, 24}

func compiledMLPSource(t *testing.T) string {
	t.Helper()
	m, err := cudalayer.NewFusedMLP(64, 64, 64, 3)
	if err != nil {
		t.Fatal(err)
	}
	u, err := cudair.EmitTree(m, "compiled_mlp_eval")
	if err != nil {
		t.Fatal(err)
	}
	s, err := nvrtcx.SpliceTemplate(cudalayer.OuterKernelTemplate, map[string]string{"FORWARD_DEVICE_FUNCTION": u.Source, "LOSS_DEVICE_FUNCTION": "", "KERNEL_NAME": compiledMLPEntry, "FUNC_NAME": "compiled_mlp_eval", "IN_WIDTH": "64", "OUT_WIDTH": "64"})
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestCompiledResources_ExportMLPFixture(t *testing.T) {
	if dir := os.Getenv("COMPILED_RESOURCE_EXPORT_DIR"); dir != "" {
		if err := os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "mlp.cu"), []byte(compiledMLPSource(t)), 0600); err != nil {
			t.Fatal(err)
		}
		t.Logf("exported emitted MLP; predeclared caps=%v; offline compile only, no native claim", compiledMLPCaps)
	}
}
