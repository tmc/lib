package cuptipcsampling_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

func ExamplePhaseConfig() {
	modules := cudacapture.NewModuleTracker()
	modules.Load("known image", 0x1000, "caller-verified-image-digest", cudacapture.CodeFiles{})
	cfg := cuptipcsampling.PhaseConfig{
		Modules:        modules,
		MetadataLimits: cudacapture.ReannouncementLimits{MaxEntries: 16, MaxBytes: 8192},
	}
	// Supply the sampling Config, phase Limits, context and workload when
	// calling CapturePhase. Metadata limits do not bound sampled PC counts.
	fmt.Println(cfg.MetadataLimits.MaxEntries)
	// Output: 16
}
