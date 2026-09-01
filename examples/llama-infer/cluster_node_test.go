package main

import (
	"context"
	"testing"
	"time"
)

func TestIrohClusterNodeRoleInference(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cfg := Config{
		Name:      "Test-Llama-7B",
		Dim:       4096,
		NumLayers: 32,
	}

	node, err := NewIrohClusterNode(ctx, nil, cfg, 0)
	if err != nil {
		t.Fatalf("failed initializing IrohClusterNode: %v", err)
	}
	defer node.Close()

	if node.NodeID() == "" {
		t.Fatal("expected non-empty Ed25519 NodeID")
	}
	if len(node.SelfTelemetry().EndpointAddrs) == 0 {
		t.Fatal("expected self telemetry to advertise Iroh endpoint addresses")
	}

	// Register a simulated Linux DGX peer with high compute (150 TFLOPs) and moderate bandwidth (300 GB/s)
	dgxPeer := NodeTelemetry{
		NodeID:        "ed25519_linux_dgx_001",
		OS:            "linux",
		Arch:          "amd64",
		ComputeTFLOPs: 150.0,
		BandwidthGBS:  300.0,
		FreeVRAMMB:    80000,
	}
	node.RegisterPeer(dgxPeer)

	// Scenario 1: Heavy Prompt Prefill (Prompt=8192 tokens, Gen=32 tokens)
	// The E-Graph solver should dynamically route prefill to the compute-dense Linux node
	assignment := node.InferRoles(8192, 32)
	if assignment.PrefillNodeID != dgxPeer.NodeID {
		t.Errorf("expected prefill on DGX peer (%s), got: %s", dgxPeer.NodeID, assignment.PrefillNodeID)
	}
	t.Logf("Heavy Prompt Assignment: Prefill=%s, Decode=%s, Latency=%.2f ms (%s)",
		assignment.PrefillNodeID, assignment.DecodeNodeID, assignment.EstimatedLatencyMs, assignment.Reason)

	// Scenario 2: Single Node Fallback when no peers exist
	singleNode, err := NewIrohClusterNode(ctx, nil, cfg, 0)
	if err != nil {
		t.Fatalf("failed initializing singleNode: %v", err)
	}
	defer singleNode.Close()

	singleAssignment := singleNode.InferRoles(128, 128)
	if singleAssignment.PrefillNodeID != singleNode.NodeID() || singleAssignment.DecodeNodeID != singleNode.NodeID() {
		t.Errorf("expected standalone node to handle both roles, got: %+v", singleAssignment)
	}

	if err := node.ConnectIroh(ctx, singleNode.Addr()); err != nil {
		t.Fatalf("connect inference peer: %v", err)
	}
	if _, err := node.PeerAddr(singleNode.NodeID()); err != nil {
		t.Fatalf("resolve advertised peer address: %v", err)
	}
}
