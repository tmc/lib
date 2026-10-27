//go:build !windows

package nvlshm

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/imex"
)

func TestNVL72PlacementAll72Ranks(t *testing.T) {
	if NVL72TotalGPUs != 72 {
		t.Fatalf("NVL72TotalGPUs = %d, want 72", NVL72TotalGPUs)
	}

	trayCounts := make(map[int]int)
	graceCounts := make(map[int]int)

	for r := 0; r < NVL72TotalGPUs; r++ {
		pl, err := PlacementForRank(r)
		if err != nil {
			t.Fatalf("PlacementForRank(%d) unexpected error: %v", r, err)
		}
		if pl.Rank != r {
			t.Errorf("rank %d: Rank = %d", r, pl.Rank)
		}
		if want := r / 4; pl.TrayID != want || pl.TrayID < 0 || pl.TrayID >= NVL72Trays {
			t.Errorf("rank %d: TrayID = %d, want %d", r, pl.TrayID, want)
		}
		if want := r % 4; pl.TrayLocalGPU != want || pl.TrayLocalGPU < 0 || pl.TrayLocalGPU >= NVL72GPUsPerTray {
			t.Errorf("rank %d: TrayLocalGPU = %d, want %d", r, pl.TrayLocalGPU, want)
		}
		if want := (r % 4) / 2; pl.GraceSocket != want || (pl.GraceSocket != 0 && pl.GraceSocket != 1) {
			t.Errorf("rank %d: GraceSocket = %d, want %d", r, pl.GraceSocket, want)
		}
		if want := r / 2; pl.GlobalGraceID != want || pl.GlobalGraceID < 0 || pl.GlobalGraceID >= NVL72Trays*NVL72GracesPerTray {
			t.Errorf("rank %d: GlobalGraceID = %d, want %d", r, pl.GlobalGraceID, want)
		}
		if pl.HostNUMANode != pl.GraceSocket {
			t.Errorf("rank %d: HostNUMANode = %d, want GraceSocket %d", r, pl.HostNUMANode, pl.GraceSocket)
		}
		trayCounts[pl.TrayID]++
		graceCounts[pl.GlobalGraceID]++
	}

	if len(trayCounts) != NVL72Trays {
		t.Fatalf("observed %d trays, want %d", len(trayCounts), NVL72Trays)
	}
	for tray := 0; tray < NVL72Trays; tray++ {
		if trayCounts[tray] != NVL72GPUsPerTray {
			t.Errorf("tray %d has %d GPUs, want %d", tray, trayCounts[tray], NVL72GPUsPerTray)
		}
	}

	wantGraces := NVL72Trays * NVL72GracesPerTray // 36
	if len(graceCounts) != wantGraces {
		t.Fatalf("observed %d Grace CPUs, want %d", len(graceCounts), wantGraces)
	}
	for g := 0; g < wantGraces; g++ {
		if graceCounts[g] != NVL72GPUsPerGrace {
			t.Errorf("global Grace %d has %d GPUs, want %d", g, graceCounts[g], NVL72GPUsPerGrace)
		}
	}

	for _, badRank := range []int{-1, -72, 72, 73, 1024} {
		if _, err := PlacementForRank(badRank); err == nil {
			t.Errorf("PlacementForRank(%d) succeeded, want error", badRank)
		}
	}
}

func TestNVL72SameComputeTrayAndSameGraceCPU(t *testing.T) {
	for a := 0; a < NVL72TotalGPUs; a++ {
		trayPeers := 0
		gracePeers := 0
		for b := 0; b < NVL72TotalGPUs; b++ {
			sameTray := SameComputeTray(a, b)
			sameGrace := SameGraceCPU(a, b)
			if sameTray != (a/4 == b/4) {
				t.Fatalf("SameComputeTray(%d, %d) = %v, want %v", a, b, sameTray, a/4 == b/4)
			}
			if sameGrace != (a/2 == b/2) {
				t.Fatalf("SameGraceCPU(%d, %d) = %v, want %v", a, b, sameGrace, a/2 == b/2)
			}
			if sameGrace && !sameTray {
				t.Fatalf("ranks %d and %d share Grace CPU but not compute tray", a, b)
			}
			if sameTray {
				trayPeers++
			}
			if sameGrace {
				gracePeers++
			}
		}
		if trayPeers != NVL72GPUsPerTray {
			t.Errorf("rank %d has %d tray peers, want %d", a, trayPeers, NVL72GPUsPerTray)
		}
		if gracePeers != NVL72GPUsPerGrace {
			t.Errorf("rank %d has %d Grace CPU peers, want %d", a, gracePeers, NVL72GPUsPerGrace)
		}
	}

	// Out-of-bounds ranks never match.
	for _, pair := range [][2]int{{-1, 0}, {0, -1}, {-1, -1}, {72, 72}, {71, 72}, {72, 0}} {
		if SameComputeTray(pair[0], pair[1]) {
			t.Errorf("SameComputeTray(%d, %d) = true, want false", pair[0], pair[1])
		}
		if SameGraceCPU(pair[0], pair[1]) {
			t.Errorf("SameGraceCPU(%d, %d) = true, want false", pair[0], pair[1])
		}
	}
}

// buildIMEXSnapshotJSON builds a valid nvidia-imex-ctl -N -j JSON snapshot for
// numNodes compute trays and parses it through imex.Read.
func buildIMEXSnapshotJSON(
	t *testing.T,
	domainStatus string,
	numNodes int,
	nodeStatusOverride map[int]string,
	linkStatusOverride map[[2]int]string,
	omitLink map[[2]int]bool,
) imex.Snapshot {
	t.Helper()
	var sb strings.Builder
	sb.WriteString(`{"nodes":{`)
	for i := 0; i < numNodes; i++ {
		if i > 0 {
			sb.WriteByte(',')
		}
		nStatus := "READY"
		if st, ok := nodeStatusOverride[i]; ok {
			nStatus = st
		}
		host := fmt.Sprintf("10.0.0.%d", i+1)
		hostname := fmt.Sprintf("gb200-tray-%02d", i)
		fmt.Fprintf(&sb, `"%d":{"status":%q,"host":%q,"changed":false,"version":"580.30","hostName":%q,"connections":{`,
			i, nStatus, host, hostname)
		firstConn := true
		for j := 0; j < numNodes; j++ {
			if omitLink != nil && omitLink[[2]int{i, j}] {
				continue
			}
			if !firstConn {
				sb.WriteByte(',')
			}
			firstConn = false
			cStatus := "CONNECTED"
			if st, ok := linkStatusOverride[[2]int{i, j}]; ok {
				cStatus = st
			}
			peerHost := fmt.Sprintf("10.0.0.%d", j+1)
			fmt.Fprintf(&sb, `"%d":{"host":%q,"status":%q,"changed":false}`, j, peerHost, cStatus)
		}
		sb.WriteString(`}}`)
	}
	fmt.Fprintf(&sb, `},"timestamp":"10/26/2026 12:00:00.000","status":%q}`, domainStatus)

	snap, err := imex.Read(strings.NewReader(sb.String()), imex.DCGMSpec58030, imex.Limits{})
	if err != nil {
		t.Fatalf("imex.Read synthetic snapshot failed: %v\njson: %s", err, sb.String())
	}
	return snap
}

func TestNVL72VerifyIMEXDomainHealthy18Trays(t *testing.T) {
	snap := buildIMEXSnapshotJSON(t, "UP", NVL72Trays, nil, nil, nil)
	report, err := VerifyNVL72IMEXDomain(&snap, NVL72Trays)
	if err != nil {
		t.Fatalf("VerifyNVL72IMEXDomain(18 trays) unexpected error: %v", err)
	}
	if !report.FabricAdmitted {
		t.Fatal("FabricAdmitted = false, want true")
	}
	if report.ExpectedNodes != NVL72Trays || report.ObservedNodes != NVL72Trays {
		t.Fatalf("nodes = %d/%d, want %d", report.ObservedNodes, report.ExpectedNodes, NVL72Trays)
	}
	if report.TotalGPUs != NVL72TotalGPUs {
		t.Fatalf("TotalGPUs = %d, want %d", report.TotalGPUs, NVL72TotalGPUs)
	}
	if len(report.HealthyTrays) != NVL72Trays {
		t.Fatalf("len(HealthyTrays) = %d, want %d", len(report.HealthyTrays), NVL72Trays)
	}
	if len(report.UnhealthyTrays) != 0 {
		t.Fatalf("UnhealthyTrays = %v, want empty", report.UnhealthyTrays)
	}
	if len(report.UnhealthyLinks) != 0 {
		t.Fatalf("UnhealthyLinks = %+v, want empty", report.UnhealthyLinks)
	}

	// Verify a 4-tray (16-GPU) subset with READY status also passes when expectedNodes=4.
	snap4 := buildIMEXSnapshotJSON(t, "READY", 4, nil, nil, nil)
	rep4, err := VerifyNVL72IMEXDomain(&snap4, 4)
	if err != nil || !rep4.FabricAdmitted || rep4.TotalGPUs != 16 {
		t.Fatalf("4-tray check failed: report=%+v, err=%v", rep4, err)
	}
}

func TestNVL72VerifyIMEXDomainDegradedAndDisconnected(t *testing.T) {
	t.Run("nil snapshot and invalid expectedNodes", func(t *testing.T) {
		if _, err := VerifyNVL72IMEXDomain(nil, NVL72Trays); err == nil {
			t.Fatal("nil snapshot: want error")
		}
		snap := buildIMEXSnapshotJSON(t, "UP", 2, nil, nil, nil)
		if _, err := VerifyNVL72IMEXDomain(&snap, 0); err == nil {
			t.Fatal("zero expectedNodes: want error")
		}
	})

	t.Run("degraded domain status", func(t *testing.T) {
		snap := buildIMEXSnapshotJSON(t, "DEGRADED", NVL72Trays, nil, nil, nil)
		report, err := VerifyNVL72IMEXDomain(&snap, NVL72Trays)
		if err == nil {
			t.Fatal("DEGRADED domain status: want error")
		}
		if report == nil || report.FabricAdmitted {
			t.Fatalf("expected FabricAdmitted=false, got %+v", report)
		}
	})

	t.Run("node count mismatch", func(t *testing.T) {
		snap := buildIMEXSnapshotJSON(t, "UP", 17, nil, nil, nil)
		report, err := VerifyNVL72IMEXDomain(&snap, NVL72Trays)
		if err == nil {
			t.Fatal("17 of 18 trays present: want error")
		}
		if report == nil || report.FabricAdmitted || report.ObservedNodes != 17 {
			t.Fatalf("unexpected report: %+v", report)
		}
	})

	t.Run("unhealthy compute tray", func(t *testing.T) {
		snap := buildIMEXSnapshotJSON(t, "UP", NVL72Trays, map[int]string{5: "DOWN"}, nil, nil)
		report, err := VerifyNVL72IMEXDomain(&snap, NVL72Trays)
		if err == nil {
			t.Fatal("tray 5 DOWN: want error")
		}
		if report == nil || report.FabricAdmitted || len(report.UnhealthyTrays) != 1 || report.UnhealthyTrays[0] != "5" {
			t.Fatalf("unexpected UnhealthyTrays: %+v", report)
		}
	})

	t.Run("disconnected directional link", func(t *testing.T) {
		snap := buildIMEXSnapshotJSON(t, "UP", NVL72Trays, nil, map[[2]int]string{{2, 14}: "DISCONNECTED"}, nil)
		report, err := VerifyNVL72IMEXDomain(&snap, NVL72Trays)
		if err == nil {
			t.Fatal("disconnected link 2->14: want error")
		}
		if report == nil || report.FabricAdmitted || len(report.UnhealthyLinks) != 1 {
			t.Fatalf("expected 1 unhealthy link, got %+v", report)
		}
		issue := report.UnhealthyLinks[0]
		if issue.FromNode != "2" || issue.ToNode != "14" || issue.Status != "DISCONNECTED" {
			t.Fatalf("unexpected link issue: %+v", issue)
		}
	})

	t.Run("missing directional peer link", func(t *testing.T) {
		snap := buildIMEXSnapshotJSON(t, "UP", 4, nil, nil, map[[2]int]bool{{1, 3}: true})
		report, err := VerifyNVL72IMEXDomain(&snap, 4)
		if err == nil {
			t.Fatal("missing link 1->3: want error")
		}
		if report == nil || report.FabricAdmitted || len(report.UnhealthyLinks) != 1 {
			t.Fatalf("expected 1 missing link issue, got %+v", report)
		}
		issue := report.UnhealthyLinks[0]
		if issue.FromNode != "1" || issue.ToNode != "3" || issue.Status != "MISSING" {
			t.Fatalf("unexpected missing link issue: %+v", issue)
		}
	})

	t.Run("dcgm spec degraded fixture with unknown peers", func(t *testing.T) {
		const dcgmFixture = `{"nodes":{
			"3": {"status":"READY","host":"10.76.186.255",
				"connections":{"1":{"host":"10.76.183.30","status":"CONNECTED","changed":true},
					"6":{"host":"10.76.190.23","status":"CONNECTED","changed":true}
				},
				"changed":true,
				"version":"580.30",
				"hostName":""},
			"1":{"status":"READY","host":"10.76.183.30",
				"connections":{"0":{"host":"10.76.179.115","status":"CONNECTED","changed":true},
				"1":{"host":"10.76.183.30","status":"CONNECTED","changed":true}},
				"changed":true,"version":"580.30","hostName":""}},
			"timestamp":"5/22/2025 19:46:40.891",
			"status":"DEGRADED"}`
		snap, err := imex.Read(strings.NewReader(dcgmFixture), imex.DCGMSpec58030, imex.Limits{})
		if err != nil {
			t.Fatalf("imex.Read: %v", err)
		}
		report, err := VerifyNVL72IMEXDomain(&snap, 2)
		if err == nil {
			t.Fatal("DCGM degraded fixture: want error")
		}
		if report == nil || report.FabricAdmitted || len(report.UnhealthyLinks) == 0 {
			t.Fatalf("expected unhealthy links on DCGM degraded fixture, got %+v", report)
		}
	})
}
