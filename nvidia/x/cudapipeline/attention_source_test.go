package cudapipeline

import (
	"slices"
	"strings"
	"testing"
)

func TestAttentionSourceContract(t *testing.T) {
	for _, layout := range []AttentionLayout{
		{Keys: 1, HeadDim: 1, TileSize: 1},
		{Keys: 5, HeadDim: 3, TileSize: 2, Excluded: []bool{false, true, false, true, true}},
		{Keys: 64, HeadDim: 64, TileSize: 64},
		{Keys: 3, HeadDim: 1, TileSize: 2, Excluded: []bool{true, true, true}, FullyMaskedPolicy: "canonical-nan-exact-fp32"},
	} {
		payload, err := layout.PayloadBytes()
		if err != nil || payload%16 != 0 {
			t.Fatalf("payload = %d, %v", payload, err)
		}
		schedule, err := BuildWarpSpecializedSchedule(2, payload, (layout.Keys+layout.TileSize-1)/layout.TileSize)
		if err != nil {
			t.Fatal(err)
		}
		source, schema, err := schedule.AttentionSource("attention", layout)
		if err != nil || !strings.Contains(source, "u[d]/l") || strings.Contains(source, "cp.async") {
			t.Fatalf("source contract: %v", err)
		}
		if len(schema.Params) != 2 || schema.Params[0].Offset != 0 || schema.Params[1].Offset != 8 || schema.TotalParamBytes != 16 {
			t.Fatalf("pointer parameter bank: %+v", schema)
		}
	}
}

func TestAttentionSourceLayoutRefusals(t *testing.T) {
	for _, layout := range []AttentionLayout{
		{}, {Keys: 65, HeadDim: 1, TileSize: 1}, {Keys: 1, HeadDim: 65, TileSize: 1},
		{Keys: 1, HeadDim: 1, TileSize: 2}, {Keys: 1, HeadDim: 1, TileSize: -1},
		{Keys: 2, HeadDim: 1, TileSize: 1, Excluded: []bool{true}},
		{Keys: 1, HeadDim: 1, TileSize: 1, Excluded: []bool{true}},
		{Keys: 1, HeadDim: 1, TileSize: 1, FullyMaskedPolicy: "zero"},
	} {
		if _, err := layout.PayloadBytes(); err == nil {
			t.Fatalf("invalid layout admitted: %+v", layout)
		}
	}
	layout := AttentionLayout{Keys: 64, HeadDim: 64, TileSize: 64}
	payload, _ := layout.PayloadBytes()
	schedule, _ := BuildWarpSpecializedSchedule(3, payload, 1)
	if _, _, err := schedule.AttentionSource("attention", layout); err == nil {
		t.Fatal("unadmitted shared storage accepted")
	}
	layout = AttentionLayout{Keys: 1, HeadDim: 1, TileSize: 1}
	payload, _ = layout.PayloadBytes()
	schedule, _ = BuildWarpSpecializedSchedule(2, payload, 1)
	for _, name := range []string{"", "0bad", "a;b", strings.Repeat("a", 129)} {
		if _, _, err := schedule.AttentionSource(name, layout); err == nil {
			t.Fatalf("invalid name %q admitted", name)
		}
	}
}

func TestAttentionSourceProtocolControls(t *testing.T) {
	layout := AttentionLayout{Keys: 3, HeadDim: 1, TileSize: 1}
	payload, _ := layout.PayloadBytes()
	for _, control := range []struct {
		name  string
		alter func(*PipelineSchedule)
	}{
		{"omit_init", func(s *PipelineSchedule) { s.Steps = s.Steps[1:] }},
		{"under_count", func(s *PipelineSchedule) {
			for i := range s.Steps {
				if s.Steps[i].Kind == StepArriveExpectTx {
					s.Steps[i].Bytes--
					break
				}
			}
		}},
		{"over_count", func(s *PipelineSchedule) {
			for i := range s.Steps {
				if s.Steps[i].Kind == StepIssueTMA {
					s.Steps[i].Bytes++
					break
				}
			}
		}},
		{"omit_wait", func(s *PipelineSchedule) {
			for i := range s.Steps {
				if s.Steps[i].Kind == StepWaitParity {
					s.Steps = slices.Delete(s.Steps, i, i+1)
					break
				}
			}
		}},
		{"stale_epoch", func(s *PipelineSchedule) {
			for i := range s.Steps {
				if s.Steps[i].Epoch == 1 {
					s.Steps[i].Epoch = 0
				}
			}
		}},
		{"overwrite", func(s *PipelineSchedule) {
			for i := range s.Steps {
				if s.Steps[i].Kind == StepReleaseStage {
					s.Steps[i].Kind = StepAcquireStage
					s.Steps[i].Role = RoleProducer
					break
				}
			}
		}},
		{"step_identity", func(s *PipelineSchedule) { s.Steps[0].StepIndex = 999 }},
	} {
		t.Run(control.name, func(t *testing.T) {
			schedule, _ := BuildWarpSpecializedSchedule(2, payload, 3)
			control.alter(schedule)
			if _, _, err := schedule.AttentionSource("attention", layout); err == nil {
				t.Fatal("protocol mutation admitted")
			}
		})
	}
	// Independent stages can be consumed in another order under the generic
	// protocol, but an online row's recurrence must retain its tile order.
	layout.Keys = 2
	schedule, _ := BuildWarpSpecializedSchedule(3, payload, 2)
	var first, second int
	for i, step := range schedule.Steps {
		if step.Kind == StepWaitParity {
			if step.Stage == 0 {
				first = i
			} else {
				second = i
			}
		}
	}
	a := slices.Clone(schedule.Steps[first : first+3])
	copy(schedule.Steps[first:first+3], schedule.Steps[second:second+3])
	copy(schedule.Steps[second:second+3], a)
	for i := range schedule.Steps {
		schedule.Steps[i].StepIndex = i
	}
	if failures := schedule.Validate(); len(failures) != 0 {
		t.Fatalf("control must pass generic stage validation: %v", failures)
	}
	if _, _, err := schedule.AttentionSource("attention", layout); err == nil {
		t.Fatal("out-of-order row recurrence admitted")
	}
	// An otherwise valid schedule cannot leave a bound row tile unconsumed.
	schedule, _ = BuildWarpSpecializedSchedule(2, payload, 1)
	if _, _, err := schedule.AttentionSource("attention", layout); err == nil {
		t.Fatal("incomplete row admitted")
	}
}
