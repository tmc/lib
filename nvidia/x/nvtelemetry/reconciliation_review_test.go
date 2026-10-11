package nvtelemetry_test

import (
	n "github.com/tmc/lib/nvidia/x/nvtelemetry"
	"testing"
)

func TestFinalReconciliationDuplicateIdentity(t *testing.T) {
	t.Run("mig_parent", func(t *testing.T) {
		s := independentSnapshot()
		s.Mappings[0].Sharing = "mig"
		s.Mappings[0].Device.MIGScope = n.MIGScopeMIGInstance
		s.Mappings[0].Device.MIGParentUUID = "parent-1"
		s.Mappings[0].MIGConfigurationEpoch = "mig-epoch"
		s.Executions[0].Sharing = "mig"
		s.Executions[0].Device = s.Mappings[0].Device
		s.Executions[0].MIGConfigurationEpoch = "mig-epoch"
		other := s.Mappings[0]
		other.Device.MIGParentUUID = "parent-2"
		s.Mappings = append(s.Mappings, other)
		if independentAssigned(independentJoin(t, s, 200)) {
			t.Fatal("duplicate UUID mappings with conflicting known MIG parents collapsed into assigned")
		}
	})
	t.Run("mps_role", func(t *testing.T) {
		s := independentSnapshot()
		s.Mappings[0].Sharing = "mps"
		s.Mappings[0].MPSRole = "client"
		s.Executions[0].Sharing = "mps"
		s.Executions[0].MPSRole = "client"
		other := s.Mappings[0]
		other.MPSRole = "server"
		s.Mappings = append(s.Mappings, other)
		if independentAssigned(independentJoin(t, s, 200)) {
			t.Fatal("duplicate mapping known client/server contradiction discarded")
		}
	})
}

func TestCompleteDuplicateMPSIdentity(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*n.EntityProcess)
	}{
		{"birth_generation", func(p *n.EntityProcess) { p.Process.Generation++ }},
		{"boot", func(p *n.EntityProcess) { p.BootID = "other-boot" }},
		{"node", func(p *n.EntityProcess) { p.NodeID = "other-node" }},
		{"namespace", func(p *n.EntityProcess) { p.PIDNamespace = "other-namespace" }},
		{"exec_generation", func(p *n.EntityProcess) { p.ExecGeneration++ }},
		{"namespace_coordinates", func(p *n.EntityProcess) { p.NamespacePIDs = []uint32{999} }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := independentSnapshot()
			server := s.Processes[0].Process
			server.Process.PID = 80
			server.NamespacePIDs = []uint32{80}
			s.Mappings[0].Sharing = "mps"
			s.Mappings[0].MPSRole = "client"
			s.Mappings[0].MPSServer = &server
			s.Executions[0].Sharing = "mps"
			s.Executions[0].MPSRole = "client"
			s.Executions[0].MPSServer = &server
			other := s.Mappings[0]
			different := server
			c.mutate(&different)
			other.MPSServer = &different
			s.Mappings = append(s.Mappings, other)
			if independentAssigned(independentJoin(t, s, 200)) {
				t.Fatal("duplicate known MPS server process contradiction discarded")
			}
		})
	}
}

func TestCompleteDuplicateMIGComputeIdentity(t *testing.T) {
	s := independentSnapshot()
	s.Mappings[0].Sharing = "mig"
	s.Mappings[0].Device.MIGScope = n.MIGScopeMIGInstance
	s.Mappings[0].Device.MIGParentUUID = "parent"
	s.Mappings[0].Device.HasMIGInstanceID = true
	s.Mappings[0].Device.MIGInstanceID = 3
	s.Mappings[0].Device.HasMIGComputeInstanceID = true
	s.Mappings[0].Device.MIGComputeInstanceID = 1
	s.Mappings[0].MIGConfigurationEpoch = "mig-epoch"
	s.Executions[0].Sharing = "mig"
	s.Executions[0].Device = s.Mappings[0].Device
	s.Executions[0].MIGConfigurationEpoch = "mig-epoch"
	other := s.Mappings[0]
	other.Device.MIGComputeInstanceID = 2
	s.Mappings = append(s.Mappings, other)
	if independentAssigned(independentJoin(t, s, 200)) {
		t.Fatal("duplicate known MIG compute-instance contradiction discarded")
	}
}
