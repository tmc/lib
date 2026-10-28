package cudashared_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudashared"
)

// These are host-model checks. The target string does not qualify GPU execution.
func TestClusterSharedRacePublic(t *testing.T) {
	a := cudashared.MemoryAccess{
		Scope: cudashared.ScopeCluster, Space: cudashared.MemoryShared,
		AgentID: 1, StreamID: 1, ThreadID: 0, BlockID: 0, ClusterID: 1,
		Offset: 0x600, SizeBytes: 4, IsWrite: true, Stage: 0,
	}
	b := a
	b.ThreadID = 1
	b.IsWrite = false
	for _, tt := range []struct {
		name   string
		change func(a, b *cudashared.MemoryAccess)
		race   bool
	}{
		{"cluster scope", nil, true},
		{"CTA scope", func(a, b *cudashared.MemoryAccess) { a.Scope, b.Scope = cudashared.ScopeCTA, cudashared.ScopeCTA }, true},
		{"same thread", func(a, b *cudashared.MemoryAccess) { b.ThreadID = a.ThreadID }, false},
		{"different stage", func(a, b *cudashared.MemoryAccess) { b.Stage++ }, false},
		{"read read", func(a, b *cudashared.MemoryAccess) { a.IsWrite = false }, false},
		{"disjoint bytes", func(a, b *cudashared.MemoryAccess) { a.SizeBytes, b.SizeBytes = 1, 1; b.Offset++ }, false},
		{"same stream kernels", func(a, b *cudashared.MemoryAccess) {
			a.Scope, b.Scope = cudashared.ScopeDevice, cudashared.ScopeDevice
			a.Space, b.Space = cudashared.MemoryGlobal, cudashared.MemoryGlobal
			b.AgentID++
		}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			a, b := a, b
			if tt.change != nil {
				tt.change(&a, &b)
			}
			m := cudashared.NewHazardMatrix("sm_90")
			if err := m.AddAccess(a); err != nil {
				t.Fatal(err)
			}
			if err := m.AddAccess(b); err != nil {
				t.Fatal(err)
			}
			records, err := m.Evaluate()
			if err != nil {
				t.Fatal(err)
			}
			gotRace := false
			for _, record := range records {
				gotRace = gotRace || record.IsRace
			}
			if gotRace != tt.race {
				t.Fatalf("Evaluate race=%t, want %t: %+v", gotRace, tt.race, records)
			}
			got, err := m.HasRace()
			if err != nil || got != tt.race {
				t.Fatalf("HasRace=%t, %v; want %t", got, err, tt.race)
			}
		})
	}
}
