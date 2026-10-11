package nvtelemetry_test

import (
	"fmt"
	n "github.com/tmc/lib/nvidia/x/nvtelemetry"
	"math"
	"testing"
)

func TestSuccessorAdversarialIdentityAndProvenance(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*n.JobAttributionSnapshot)
	}{
		{"conflicting_namespace_coordinates", func(s *n.JobAttributionSnapshot) { s.Executions[0].Process.NamespacePIDs = []uint32{999} }},
		{"missing_mapping_clock_domain", func(s *n.JobAttributionSnapshot) { s.Mappings[0].ClockDomain = "" }},
		{"missing_mapping_clock_epoch", func(s *n.JobAttributionSnapshot) { s.Mappings[0].ClockEpoch = "" }},
		{"duplicate_allocation_epoch_conflict", func(s *n.JobAttributionSnapshot) {
			a := s.Allocations[0]
			a.Snapshot.ClockEpoch = "other"
			s.Allocations = append(s.Allocations, a)
		}},
		{"duplicate_mapping_sharing_conflict", func(s *n.JobAttributionSnapshot) {
			m := s.Mappings[0]
			m.Sharing = "time_slice"
			s.Mappings = append(s.Mappings, m)
		}},
		{"mig_parent_conflict", func(s *n.JobAttributionSnapshot) {
			s.Mappings[0].Sharing = "mig"
			s.Mappings[0].Device.MIGScope = n.MIGScopeMIGInstance
			s.Mappings[0].Device.MIGParentUUID = "parent-1"
			s.Mappings[0].MIGConfigurationEpoch = "mig-epoch"
			s.Executions[0].Sharing = "mig"
			s.Executions[0].Device = s.Mappings[0].Device
			s.Executions[0].Device.MIGParentUUID = "parent-2"
			s.Executions[0].MIGConfigurationEpoch = "mig-epoch"
		}},
		{"mapping_mps_role_conflict", func(s *n.JobAttributionSnapshot) {
			s.Mappings[0].Sharing = "mps"
			s.Mappings[0].MPSRole = "server"
			s.Executions[0].Sharing = "mps"
			s.Executions[0].MPSRole = "client"
		}},
	}
	for _, c := range tests {
		t.Run(c.name, func(t *testing.T) {
			s := independentSnapshot()
			c.mutate(&s)
			r := independentJoin(t, s, 200)
			if c.name == "duplicate_mapping_sharing_conflict" {
				for _, v := range r {
					if v.Assignment.State == "assigned" && v.Assignment.Sharing == "exclusive" {
						t.Fatalf("conflicting shared mapping still yields exclusive relationship: %+v", r)
					}
				}
			} else if independentAssigned(r) {
				t.Fatalf("contradiction or absent clock provenance still assigned: %+v", r)
			}
		})
	}
}
func TestSuccessorAdversarialGRESBounds(t *testing.T) {
	for _, input := range []string{fmt.Sprintf("gres/gpu:a100=%d,gres/gpu:h100=1", math.MaxInt), "gres/gpu:a100=1,gres/gpu:a100=1"} {
		if got, err := n.ParseSlurmGRES(input); err == nil {
			t.Errorf("invalid typed counts accepted: %q => %+v", input, got)
		}
	}
}
func TestSuccessorAdversarialNoInvalidSharingOutput(t *testing.T) {
	s := independentSnapshot()
	s.Executions[0].Sharing = "mps"
	s.Executions[0].MPSRole = "client"
	server := s.Executions[0].Process
	server.Process.PID = 80
	s.Executions[0].MPSServer = &server
	r := independentJoin(t, s, 200)
	for _, v := range r {
		if v.Assignment.State == "assigned" && v.Assignment.Sharing == "mps" && v.Assignment.MPSRole != "client" {
			t.Fatalf("observed MPS client role discarded: %+v", v)
		}
	}
}
