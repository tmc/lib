//go:build !windows

package nvlshm

import (
	"errors"
	"fmt"
	"sort"

	"github.com/tmc/lib/nvidia/x/imex"
)

const (
	// NVL72Trays is the number of compute trays in a full GB200 NVL72 rack.
	NVL72Trays = 18

	// NVL72GPUsPerTray is the number of Blackwell GPUs per GB200 compute tray.
	NVL72GPUsPerTray = 4

	// NVL72GracesPerTray is the number of Grace CPUs per GB200 compute tray.
	NVL72GracesPerTray = 2

	// NVL72GPUsPerGrace is the number of Blackwell GPUs connected via NVLink-C2C
	// to each Grace CPU socket.
	NVL72GPUsPerGrace = 2

	// NVL72TotalGPUs is the total number of GPUs in a GB200 NVL72 NVLink domain.
	NVL72TotalGPUs = NVL72Trays * NVL72GPUsPerTray // 72
)

// NVL72Placement describes a rank's physical coordinates inside a GB200 NVL72 rack.
type NVL72Placement struct {
	Rank          int // 0..71
	TrayID        int // 0..17 (rank / 4)
	TrayLocalGPU  int // 0..3  (rank % 4)
	GraceSocket   int // 0..1  ((rank % 4) / 2)
	GlobalGraceID int // 0..35 (rank / 2)
	HostNUMANode  int // NUMA node for coherent Grace LPDDR5X via NVLink-C2C
}

// PlacementForRank returns the physical compute tray, local GPU index, Grace
// CPU socket, global Grace ID, and host LPDDR5X NUMA node for a rank in [0, 72).
func PlacementForRank(rank int) (NVL72Placement, error) {
	if rank < 0 || rank >= NVL72TotalGPUs {
		return NVL72Placement{}, fmt.Errorf("nvlshm: rank %d out of NVL72 range [0,%d)", rank, NVL72TotalGPUs)
	}
	trayLocalGPU := rank % NVL72GPUsPerTray
	graceSocket := trayLocalGPU / NVL72GPUsPerGrace
	return NVL72Placement{
		Rank:          rank,
		TrayID:        rank / NVL72GPUsPerTray,
		TrayLocalGPU:  trayLocalGPU,
		GraceSocket:   graceSocket,
		GlobalGraceID: rank / NVL72GPUsPerGrace,
		HostNUMANode:  graceSocket,
	}, nil
}

// SameComputeTray reports whether rankA and rankB reside on the same GB200
// compute tray (and therefore share a single Linux OS kernel instance where
// POSIX file descriptors can be exchanged). Out-of-range ranks return false.
func SameComputeTray(rankA, rankB int) bool {
	if rankA < 0 || rankA >= NVL72TotalGPUs || rankB < 0 || rankB >= NVL72TotalGPUs {
		return false
	}
	return rankA/NVL72GPUsPerTray == rankB/NVL72GPUsPerTray
}

// SameGraceCPU reports whether rankA and rankB are attached to the same Grace
// CPU socket over NVLink-C2C and share its coherent LPDDR5X NUMA node.
// Out-of-range ranks return false.
func SameGraceCPU(rankA, rankB int) bool {
	if rankA < 0 || rankA >= NVL72TotalGPUs || rankB < 0 || rankB >= NVL72TotalGPUs {
		return false
	}
	return rankA/NVL72GPUsPerGrace == rankB/NVL72GPUsPerGrace
}

// NVL72LinkIssue describes an unhealthy or missing directional IMEX connection
// from one compute tray node to another.
type NVL72LinkIssue struct {
	FromNode string
	ToNode   string
	Host     string
	Status   string
}

// NVL72DomainReport summarizes the pre-flight readiness of an IMEX domain
// for multi-node CU_MEM_HANDLE_TYPE_FABRIC allocations across GB200 compute trays.
type NVL72DomainReport struct {
	DomainStatus   string
	ExpectedNodes  int
	ObservedNodes  int
	TotalGPUs      int
	HealthyTrays   []string
	UnhealthyTrays []string
	UnhealthyLinks []NVL72LinkIssue
	FabricAdmitted bool
}

func isHealthyDomainOrNodeStatus(status string) bool {
	return status == "UP" || status == "READY"
}

func isHealthyConnectionStatus(status string) bool {
	return status == "CONNECTED" || status == "UP" || status == "READY"
}

// VerifyNVL72IMEXDomain inspects a parsed nvidia-imex-ctl snapshot and verifies
// whether the multi-node IMEX domain is ready to admit CU_MEM_HANDLE_TYPE_FABRIC
// allocations across expectedNodes compute trays:
//  1. Domain status is "UP" or "READY";
//  2. Number of participating compute trays/nodes matches expectedNodes;
//  3. Every participating compute tray has status "UP" or "READY";
//  4. Every directional node-to-node IMEX connection is present and healthy
//     ("CONNECTED", "UP", or "READY") with zero degraded or disconnected peers.
func VerifyNVL72IMEXDomain(snap *imex.Snapshot, expectedNodes int) (*NVL72DomainReport, error) {
	if snap == nil {
		return nil, errors.New("nvlshm: nil imex snapshot")
	}
	if expectedNodes <= 0 {
		return nil, fmt.Errorf("nvlshm: invalid expected node count %d", expectedNodes)
	}

	nodeIDs := make([]string, 0, len(snap.Nodes))
	for id := range snap.Nodes {
		nodeIDs = append(nodeIDs, id)
	}
	sort.Strings(nodeIDs)

	report := &NVL72DomainReport{
		DomainStatus:  snap.Status,
		ExpectedNodes: expectedNodes,
		ObservedNodes: len(snap.Nodes),
		TotalGPUs:     expectedNodes * NVL72GPUsPerTray,
	}

	for _, id := range nodeIDs {
		node := snap.Nodes[id]
		if isHealthyDomainOrNodeStatus(node.Status) {
			report.HealthyTrays = append(report.HealthyTrays, id)
		} else {
			report.UnhealthyTrays = append(report.UnhealthyTrays, id)
		}

		// Collect all target peer IDs from reported connections cleanly in sorted order.
		connTargets := make([]string, 0, len(node.Connections))
		for toID := range node.Connections {
			connTargets = append(connTargets, toID)
		}
		sort.Strings(connTargets)

		for _, toID := range connTargets {
			conn := node.Connections[toID]
			if _, knownPeer := snap.Nodes[toID]; !knownPeer {
				report.UnhealthyLinks = append(report.UnhealthyLinks, NVL72LinkIssue{
					FromNode: id,
					ToNode:   toID,
					Host:     conn.Host,
					Status:   "UNKNOWN_PEER:" + conn.Status,
				})
				continue
			}
			if !isHealthyConnectionStatus(conn.Status) {
				report.UnhealthyLinks = append(report.UnhealthyLinks, NVL72LinkIssue{
					FromNode: id,
					ToNode:   toID,
					Host:     conn.Host,
					Status:   conn.Status,
				})
			}
		}

		// Ensure every other participating node in the snapshot has a directional link from id.
		for _, peerID := range nodeIDs {
			if peerID == id {
				continue
			}
			if _, ok := node.Connections[peerID]; !ok {
				peer := snap.Nodes[peerID]
				report.UnhealthyLinks = append(report.UnhealthyLinks, NVL72LinkIssue{
					FromNode: id,
					ToNode:   peerID,
					Host:     peer.Host,
					Status:   "MISSING",
				})
			}
		}
	}

	var errs []error
	if !isHealthyDomainOrNodeStatus(snap.Status) {
		errs = append(errs, fmt.Errorf("domain status is %q (want UP or READY)", snap.Status))
	}
	if len(snap.Nodes) != expectedNodes {
		errs = append(errs, fmt.Errorf("observed %d IMEX nodes, want %d", len(snap.Nodes), expectedNodes))
	}
	if len(report.UnhealthyTrays) > 0 {
		errs = append(errs, fmt.Errorf("%d unhealthy compute trays: %v", len(report.UnhealthyTrays), report.UnhealthyTrays))
	}
	if len(report.UnhealthyLinks) > 0 {
		first := report.UnhealthyLinks[0]
		errs = append(errs, fmt.Errorf("%d unhealthy directional IMEX links (first: %s->%s status=%s)",
			len(report.UnhealthyLinks), first.FromNode, first.ToNode, first.Status))
	}
	if len(errs) > 0 {
		return report, fmt.Errorf("nvlshm: IMEX domain readiness check failed: %w", errors.Join(errs...))
	}

	report.FabricAdmitted = true
	return report, nil
}
