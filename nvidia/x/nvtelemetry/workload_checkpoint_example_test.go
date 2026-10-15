package nvtelemetry_test

import (
	"bytes"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleTrainingCheckpoint() {
	p := nvtelemetry.TrainingCheckpoint{CheckpointID: "synthetic", State: "requested", SourceAttempt: "first", SavedStep: 0, WorldSize: 1, Evidence: "application:request"}
	fmt.Println(p.State, p.SavedStep)
	// Output: requested 0
}

func ExampleTrainingCheckpointContext() {
	c := nvtelemetry.TrainingCheckpointContext{Version: 1, RunID: "synthetic", Checkpoints: []nvtelemetry.TrainingCheckpoint{}}
	fmt.Println(c.Version, len(c.Checkpoints))
	// Output: 1 0
}

func ExampleTrainingRecovery() {
	var r nvtelemetry.TrainingRecovery
	fmt.Println(r.RecoveryAvailable, r.ObservedAcceptedLoss == nil, r.UpdatePopulationComplete)
	// Output: false true false
}

func ExampleTrainingRecoverySummary() {
	var s nvtelemetry.TrainingRecoverySummary
	fmt.Println(len(s.Recoveries))
	// Output: 0
}

func ExampleWriteTrainingCheckpointContext() {
	c := nvtelemetry.TrainingCheckpointContext{Version: 1, RunID: "synthetic", Checkpoints: []nvtelemetry.TrainingCheckpoint{}}
	var b bytes.Buffer
	err := nvtelemetry.WriteTrainingCheckpointContext(&b, c, nvtelemetry.WorkloadLimits{})
	fmt.Println(b.String(), err)
	// Output: {"version":1,"run_id":"synthetic","checkpoints":[]} <nil>
}

func ExampleReadTrainingCheckpointContext() {
	raw := `{"version":1,"run_id":"synthetic","checkpoints":[]}`
	e, c, err := nvtelemetry.ReadTrainingCheckpointContext(strings.NewReader(raw), nvtelemetry.WorkloadSource{Provider: "application", Version: "synthetic-v1", RunID: "synthetic"}, nvtelemetry.WorkloadLimits{})
	fmt.Println(c.RunID, len(e.RawSHA256), err)
	// Output: synthetic 64 <nil>
}

func ExampleSummarizeTrainingRecovery() {
	// Synthetic application assertions; these do not certify durable storage.
	source := nvtelemetry.WorkloadSource{Provider: "application", Version: "synthetic-v1", RunID: "synthetic"}
	training := nvtelemetry.TrainingContext{Version: 1, RunID: source.RunID, Updates: []nvtelemetry.TrainingUpdate{
		{Attempt: "a", Step: 11, WorldSize: 1, Outcome: "updated", Quality: "accepted", QualityPolicy: "application:oracle", Evidence: "application:update"},
	}}
	generation := uint64(1)
	publication := nvtelemetry.TrainingCheckpoint{CheckpointID: "h", State: "published", Digest: strings.Repeat("a", 64), Storage: "application:storage", SourceAttempt: "a", SavedStep: 10, WorldSize: 1, Evidence: "application:publication"}
	restore := publication
	restore.State, restore.RestoredAttempt, restore.ResumeGeneration = "restored", "b", &generation
	restore.Evidence = "application:restore"
	checkpoints := nvtelemetry.TrainingCheckpointContext{Version: 1, RunID: source.RunID, Checkpoints: []nvtelemetry.TrainingCheckpoint{publication, restore}}
	var t, c bytes.Buffer
	if err := nvtelemetry.WriteTrainingContext(&t, training, nvtelemetry.WorkloadLimits{}); err != nil {
		panic(err)
	}
	if err := nvtelemetry.WriteTrainingCheckpointContext(&c, checkpoints, nvtelemetry.WorkloadLimits{}); err != nil {
		panic(err)
	}
	te, _, err := nvtelemetry.ReadTrainingContext(&t, source, nvtelemetry.WorkloadLimits{})
	if err != nil {
		panic(err)
	}
	ce, _, err := nvtelemetry.ReadTrainingCheckpointContext(&c, source, nvtelemetry.WorkloadLimits{})
	if err != nil {
		panic(err)
	}
	summary, err := nvtelemetry.SummarizeTrainingRecovery(te, ce, nvtelemetry.WorkloadLimits{})
	if err != nil {
		panic(err)
	}
	r := summary.Recoveries[0]
	fmt.Println(r.RecoveryAvailable, *r.ObservedAcceptedLoss, r.UpdatePopulationComplete)
	// Output: true 1 false
}
