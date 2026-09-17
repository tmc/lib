package cudakernel_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func TestArtifactKindRequiresLink(t *testing.T) {
	tests := []struct {
		kind         cudakernel.Kind
		requiresLink bool
	}{
		{cudakernel.KindPTX, false},
		{cudakernel.KindCUBIN, false},
		{cudakernel.KindLTOIR, true},
		{cudakernel.KindTileIR, false},
	}

	for _, tt := range tests {
		t.Run(tt.kind.String(), func(t *testing.T) {
			if got := tt.kind.RequiresLink(); got != tt.requiresLink {
				t.Errorf("Kind(%v).RequiresLink() = %v, want %v", tt.kind, got, tt.requiresLink)
			}
		})
	}

	schema, _ := cudakernel.NewSchema(cudakernel.Param{Name: "p", Type: "u32"})
	ltoirData := []byte("ltoir-binary-intermediate-representation")

	ltoArtifact, err := cudakernel.NewArtifact(
		cudakernel.KindLTOIR,
		ltoirData,
		"lto_entry",
		schema,
		"sm_90",
	)
	if err != nil {
		t.Fatalf("NewArtifact LTOIR: %v", err)
	}

	if !ltoArtifact.RequiresLink() {
		t.Error("LTOIR artifact must report RequiresLink() == true")
	}

	// Negative control: Attempt to load an unlinked artifact directly into CUDA driver
	// Must be rejected immediately before any driver interaction.
	var dev *cuda.Device // Even with nil device or dummy device, unlinked check precedes device use!
	_, err = ltoArtifact.Load(dev)
	if err == nil {
		t.Fatal("negative control failed: attempted load of unlinked LTOIR artifact succeeded")
	}
	if !strings.Contains(err.Error(), "requires linking before loading") {
		t.Fatalf("expected 'requires linking before loading' error, got: %v", err)
	}
}
