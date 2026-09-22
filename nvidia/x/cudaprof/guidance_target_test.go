package cudaprof_test

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

func profileTarget(s *cudaprof.ProfileSnapshot, workload, tokens, device string) cudaprof.ProfileTarget {
	return cudaprof.ProfileTarget{CodeDigest: s.CodeDigest, Region: s.Region, CollectorMode: s.CollectorMode, Workload: workload, TokensHash: tokens, Device: device}
}

func TestGuidanceFullTargetIdentity(t *testing.T) {
	for _, field := range []string{"code", "region", "collector"} {
		for _, value := range []string{"", "other"} {
			t.Run(field+"/"+value, func(t *testing.T) {
				s := validProfileSnapshot()
				target := profileTarget(&s, "work", "", "device")
				switch field {
				case "code":
					target.CodeDigest = value
				case "region":
					target.Region = value
				case "collector":
					target.CollectorMode = value
				}
				g, err := s.Guidance(target)
				if g != nil || !errors.Is(err, cudaprof.ErrProfileTargetMismatch) {
					t.Fatalf("mismatched target accepted: %+v %v", g, err)
				}
			})
		}
	}
}

func ExampleProfileSnapshot_Guidance() {
	digest := strings.Repeat("a", 64) // Synthetic fixture identity.
	snapshot, err := cudaprof.NewProfileSnapshot(cudaprof.ProfileSnapshot{CodeDigest: digest, Region: "kernel", Workload: "fixture", Device: "test-device", CollectorMode: "synthetic", Coverage: 1, MetricUnits: map[string]string{"samples": "samples"}, Metrics: map[string]int64{"samples": 5}})
	if err != nil {
		panic(err)
	}
	guidance, err := snapshot.Guidance(cudaprof.ProfileTarget{CodeDigest: digest, Region: "kernel", Workload: "fixture", Device: "test-device", CollectorMode: "synthetic"})
	fmt.Println(guidance.StallCounts["samples"], err)
	// Output: 5 <nil>
}
