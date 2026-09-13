//go:build !tpu

package main

import (
	"context"
	"fmt"
	"net/netip"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmc/go-iroh/netaddr"
)

func testEndpointAddr(node *IrohClusterNode) netaddr.EndpointAddr {
	addr := node.ep.LocalAddr()
	loopback := netip.AddrFrom4([4]byte{127, 0, 0, 1})
	return netaddr.NewEndpointAddr(node.ep.ID()).WithIP(netip.AddrPortFrom(loopback, addr.Port()))
}

func TestNodeSecretKeyPersistent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state", "iroh.key")
	first, err := nodeSecretKey(path)
	if err != nil {
		t.Fatalf("create node key: %v", err)
	}
	second, err := nodeSecretKey(path)
	if err != nil {
		t.Fatalf("reload node key: %v", err)
	}
	if first.Public() != second.Public() {
		t.Fatalf("node identity changed: %s != %s", first.Public(), second.Public())
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat node key: %v", err)
	}
	if info.Mode().Perm() != 0600 {
		t.Errorf("node key permissions = %o, want 0600", info.Mode().Perm())
	}
}

func TestIrohClusterNodePersistentIdentity(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cfg := Config{NodeKeyPath: filepath.Join(t.TempDir(), "iroh.key")}

	first, err := NewIrohClusterNode(ctx, nil, cfg, 0)
	if err != nil {
		t.Fatalf("create first node: %v", err)
	}
	firstID := first.NodeID()
	if err := first.Close(); err != nil {
		t.Fatalf("close first node: %v", err)
	}

	second, err := NewIrohClusterNode(ctx, nil, cfg, 0)
	if err != nil {
		t.Fatalf("create second node: %v", err)
	}
	defer second.Close()
	if second.NodeID() != firstID {
		t.Errorf("node ID changed after restart: %s != %s", second.NodeID(), firstID)
	}
}

func TestIrohClusterNodePersistentPeers(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	dir := t.TempDir()
	cfg := Config{
		NodeKeyPath:   filepath.Join(dir, "iroh.key"),
		PeerStatePath: filepath.Join(dir, "peers.json"),
	}
	first, err := NewIrohClusterNode(ctx, nil, cfg, 0)
	if err != nil {
		t.Fatalf("create first node: %v", err)
	}
	peer, err := NewIrohClusterNode(ctx, nil, Config{}, 0)
	if err != nil {
		_ = first.Close()
		t.Fatalf("create peer node: %v", err)
	}
	defer peer.Close()
	if err := first.RegisterPeer(peer.SelfTelemetry()); err != nil {
		_ = first.Close()
		t.Fatalf("register peer: %v", err)
	}
	if err := first.Close(); err != nil {
		t.Fatalf("close first node: %v", err)
	}

	second, err := NewIrohClusterNode(ctx, nil, cfg, 0)
	if err != nil {
		t.Fatalf("restart node: %v", err)
	}
	defer second.Close()
	if _, err := second.PeerAddr(peer.NodeID()); err != nil {
		t.Fatalf("restore peer address: %v", err)
	}
	info, err := os.Stat(cfg.PeerStatePath)
	if err != nil {
		t.Fatalf("stat peer state: %v", err)
	}
	if info.Mode().Perm() != 0600 {
		t.Errorf("peer state permissions = %o, want 0600", info.Mode().Perm())
	}
}

func TestEndpointAddrFromTelemetryRelay(t *testing.T) {
	_, err := endpointAddrFromTelemetry(NodeTelemetry{
		NodeID:        "63c567da88a8c72d3b1714a6fddbdb0139a6bc7166c48b7997e9bdad0d1f651b",
		EndpointAddrs: []string{"relay:https://aps1-1.relay.n0.iroh-canary.iroh.link./"},
	})
	if err != nil {
		t.Fatal(err)
	}
}

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

	if err := node.ConnectIroh(ctx, testEndpointAddr(singleNode)); err != nil {
		t.Fatalf("connect inference peer: %v", err)
	}
	if _, err := node.PeerAddr(singleNode.NodeID()); err != nil {
		t.Fatalf("resolve advertised peer address: %v", err)
	}
}

func TestIrohClusterNodeSpeculate(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	localEngine := &ModelEngine{metrics: NewRuntimeMetrics(4)}
	remoteEngine := &ModelEngine{metrics: NewRuntimeMetrics(4)}
	local, err := NewIrohClusterNode(ctx, localEngine, Config{}, 0)
	if err != nil {
		t.Fatalf("create local node: %v", err)
	}
	defer local.Close()
	remote, err := NewIrohClusterNode(ctx, remoteEngine, Config{}, 0)
	if err != nil {
		t.Fatalf("create remote node: %v", err)
	}
	defer remote.Close()

	if err := local.ConnectIroh(ctx, testEndpointAddr(remote)); err != nil {
		t.Fatalf("connect nodes: %v", err)
	}
	trace, err := local.Speculate(ctx, remote.NodeID(), "paired execution", 2)
	if err != nil {
		t.Fatalf("speculate: %v", err)
	}
	if len(trace.Draft) != 2 || len(trace.Target) != 2 || trace.Accepted != 2 {
		t.Errorf("trace = %+v, want two accepted paired tokens", trace)
	}
	if got := localEngine.metrics.Speculation(); len(got) != 1 || got[0].TargetNodeID != remote.NodeID() {
		t.Errorf("recorded traces = %+v, want remote target trace", got)
	}
}

func TestIrohClusterNodeGenerateRemote(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	local, err := NewIrohClusterNode(ctx, &ModelEngine{metrics: NewRuntimeMetrics(4)}, Config{}, 0)
	if err != nil {
		t.Fatalf("create local node: %v", err)
	}
	defer local.Close()
	remote, err := NewIrohClusterNode(ctx, &ModelEngine{metrics: NewRuntimeMetrics(4)}, Config{}, 0)
	if err != nil {
		t.Fatalf("create remote node: %v", err)
	}
	defer remote.Close()

	if err := local.ConnectIroh(ctx, testEndpointAddr(remote)); err != nil {
		t.Fatalf("connect nodes: %v", err)
	}
	result, err := local.GenerateRemote(ctx, remote.NodeID(), "selected node", 2)
	if err != nil {
		t.Fatalf("generate remotely: %v", err)
	}
	if result.NodeID != remote.NodeID() || len(result.Tokens) != 2 || result.Text == "" {
		t.Errorf("result = %+v, want two tokens from selected remote node", result)
	}
}

func TestMeshExecutionPlan_StandaloneFallback(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cfg := Config{
		Dim:       2048,
		NumLayers: 16,
	}
	node, err := NewIrohClusterNode(ctx, nil, cfg, 0)
	if err != nil {
		t.Fatalf("create standalone node: %v", err)
	}
	defer node.Close()

	plan := node.InferMeshPlan(512, 64, 1)

	if plan.PrefillNodeID != node.NodeID() {
		t.Errorf("prefill node = %s, want %s", plan.PrefillNodeID, node.NodeID())
	}
	if plan.DecodeNodeID != node.NodeID() {
		t.Errorf("decode node = %s, want %s", plan.DecodeNodeID, node.NodeID())
	}
	if plan.AttentionNodeID != node.NodeID() {
		t.Errorf("attention node = %s, want %s", plan.AttentionNodeID, node.NodeID())
	}
	if plan.MLPNodeID != node.NodeID() {
		t.Errorf("mlp node = %s, want %s", plan.MLPNodeID, node.NodeID())
	}
	if plan.TransferBytes != 0 {
		t.Errorf("transfer bytes = %d, want 0", plan.TransferBytes)
	}
	if plan.NetworkLatencyMs != 0 {
		t.Errorf("network latency = %.2f ms, want 0", plan.NetworkLatencyMs)
	}
	if plan.EstimatedLatencyMs <= 0 {
		t.Errorf("estimated latency = %.2f ms, want > 0", plan.EstimatedLatencyMs)
	}
	if len(plan.ComponentPlacements) != 4 {
		t.Errorf("component placements count = %d, want 4", len(plan.ComponentPlacements))
	}
}

func TestMeshExecutionPlan_HeterogeneousAssignment(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cfg := Config{
		Dim:       4096,
		NumLayers: 32,
	}
	node, err := NewIrohClusterNode(ctx, nil, cfg, 0)
	if err != nil {
		t.Fatalf("create node: %v", err)
	}
	defer node.Close()

	// High-compute Linux DGX peer (e.g. Blackwell/H100 NVLink cluster)
	dgxPeer := NodeTelemetry{
		NodeID:        "ed25519_peer_dgx_h100",
		OS:            "linux",
		Arch:          "amd64",
		ComputeTFLOPs: 500.0,
		BandwidthGBS:  200.0,
		FreeVRAMMB:    80000,
	}
	// High-bandwidth Apple Silicon Mac Studio peer (unified memory ~1600 GB/s)
	applePeer := NodeTelemetry{
		NodeID:        "ed25519_peer_mac_m3max",
		OS:            "darwin",
		Arch:          "arm64",
		ComputeTFLOPs: 40.0,
		BandwidthGBS:  1600.0,
		FreeVRAMMB:    128000,
	}

	_ = node.RegisterPeer(dgxPeer)
	_ = node.RegisterPeer(applePeer)

	// High-speed link between peers (e.g. 50 GB/s, 0.05ms RTT)
	fastLink := PeerLinkMetrics{
		RTT:          50 * time.Microsecond,
		BandwidthGBS: 50.0,
	}
	node.SetPeerLinkMetrics(dgxPeer.NodeID, fastLink)
	node.SetPeerLinkMetrics(applePeer.NodeID, fastLink)

	tests := []struct {
		name        string
		promptLen   int
		genTokens   int
		batchSize   int
		wantPrefill string
		wantDecode  string
		wantAttn    string
		wantMLP     string
	}{
		{
			name:        "ComputeDensePrefillAndMemoryDenseDecode",
			promptLen:   4096,
			genTokens:   128,
			batchSize:   1,
			wantPrefill: dgxPeer.NodeID,
			wantDecode:  applePeer.NodeID,
			wantAttn:    applePeer.NodeID,
			wantMLP:     dgxPeer.NodeID,
		},
		{
			name:        "LongSequenceBatchedPrefill",
			promptLen:   8192,
			genTokens:   64,
			batchSize:   2,
			wantPrefill: dgxPeer.NodeID,
			wantDecode:  applePeer.NodeID,
			wantAttn:    applePeer.NodeID,
			wantMLP:     dgxPeer.NodeID,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plan := node.InferMeshPlan(tt.promptLen, tt.genTokens, tt.batchSize)
			if plan.PrefillNodeID != tt.wantPrefill {
				t.Errorf("prefill node = %s, want %s", plan.PrefillNodeID, tt.wantPrefill)
			}
			if plan.DecodeNodeID != tt.wantDecode {
				t.Errorf("decode node = %s, want %s", plan.DecodeNodeID, tt.wantDecode)
			}
			if plan.AttentionNodeID != tt.wantAttn {
				t.Errorf("attention node = %s, want %s", plan.AttentionNodeID, tt.wantAttn)
			}
			if plan.MLPNodeID != tt.wantMLP {
				t.Errorf("mlp node = %s, want %s", plan.MLPNodeID, tt.wantMLP)
			}
			if plan.TransferBytes <= 0 {
				t.Errorf("expected non-zero transfer bytes for disaggregated plan, got %d", plan.TransferBytes)
			}
			if plan.NetworkLatencyMs <= 0 {
				t.Errorf("expected non-zero network latency for disaggregated plan, got %.2f", plan.NetworkLatencyMs)
			}
		})
	}
}

func TestMeshExecutionPlan_NetworkLinkSensitivity(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cfg := Config{
		Dim:       4096,
		NumLayers: 32,
	}
	node, err := NewIrohClusterNode(ctx, nil, cfg, 0)
	if err != nil {
		t.Fatalf("create node: %v", err)
	}
	defer node.Close()

	dgxPeer := NodeTelemetry{
		NodeID:        "ed25519_peer_dgx",
		OS:            "linux",
		Arch:          "amd64",
		ComputeTFLOPs: 250.0,
		BandwidthGBS:  300.0,
		FreeVRAMMB:    80000,
	}
	macPeer := NodeTelemetry{
		NodeID:        "ed25519_peer_mac",
		OS:            "darwin",
		Arch:          "arm64",
		ComputeTFLOPs: 35.0,
		BandwidthGBS:  900.0,
		FreeVRAMMB:    64000,
	}

	_ = node.RegisterPeer(dgxPeer)
	_ = node.RegisterPeer(macPeer)

	tests := []struct {
		name              string
		linkMetrics       PeerLinkMetrics
		promptLen         int
		genTokens         int
		wantDisaggregated bool
	}{
		{
			name: "FastInterconnectEnablesDisaggregation",
			linkMetrics: PeerLinkMetrics{
				RTT:          50 * time.Microsecond,
				BandwidthGBS: 50.0, // 400 Gbps
			},
			promptLen:         4096,
			genTokens:         256,
			wantDisaggregated: true,
		},
		{
			name: "SlowWANForcesCoLocation",
			linkMetrics: PeerLinkMetrics{
				RTT:          100 * time.Millisecond,
				BandwidthGBS: 0.01, // 80 Mbps
			},
			promptLen:         4096,
			genTokens:         256,
			wantDisaggregated: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node.SetPeerLinkMetrics(dgxPeer.NodeID, tt.linkMetrics)
			node.SetPeerLinkMetrics(macPeer.NodeID, tt.linkMetrics)

			plan := node.InferMeshPlan(tt.promptLen, tt.genTokens, 1)
			isDisaggregated := plan.PrefillNodeID != plan.DecodeNodeID

			if isDisaggregated != tt.wantDisaggregated {
				t.Errorf("disaggregated = %v, want %v (prefill=%s, decode=%s, reason=%s)",
					isDisaggregated, tt.wantDisaggregated, plan.PrefillNodeID, plan.DecodeNodeID, plan.Reason)
			}
			if !tt.wantDisaggregated && plan.TransferBytes != 0 {
				t.Errorf("co-located transfer bytes = %d, want 0", plan.TransferBytes)
			}
		})
	}
}

func TestMeshExecutionPlan_TensorTransferAccounting(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cfg := Config{
		Dim:        2048,
		NumLayers:  16,
		NumHeads:   16,
		NumKVHeads: 4, // 4:1 GQA -> kvDim = 512
		Precision:  "f16",
	}
	node, err := NewIrohClusterNode(ctx, nil, cfg, 0)
	if err != nil {
		t.Fatalf("create node: %v", err)
	}
	defer node.Close()

	peer := NodeTelemetry{
		NodeID:        "ed25519_remote_worker",
		OS:            "linux",
		Arch:          "amd64",
		ComputeTFLOPs: 200.0,
		BandwidthGBS:  400.0,
	}
	_ = node.RegisterPeer(peer)
	node.SetPeerLinkMetrics(peer.NodeID, PeerLinkMetrics{
		RTT:          10 * time.Microsecond,
		BandwidthGBS: 100.0,
	})

	batch := 2
	seqLen := 1024
	plan := node.InferMeshPlan(seqLen, 32, batch)

	// activationBytes: batch * seqLen * dim * sizeof(dtype) = 2 * 1024 * 2048 * 2 = 8,388,608 bytes
	wantActivationBytes := int64(2 * 1024 * 2048 * 2)
	// kvCacheBytes: 2 * layers * batch * seqLen * kvDim * sizeof(dtype) = 2 * 16 * 2 * 1024 * 512 * 2 = 67,108,864 bytes
	wantKVCacheBytes := int64(2 * 16 * 2 * 1024 * 512 * 2)
	wantTotalTransfer := wantActivationBytes + wantKVCacheBytes

	if plan.PrefillNodeID != plan.DecodeNodeID {
		if plan.TransferBytes != wantTotalTransfer {
			t.Errorf("transfer bytes = %d, want %d (act=%d, kv=%d)",
				plan.TransferBytes, wantTotalTransfer, wantActivationBytes, wantKVCacheBytes)
		}
	}
}

func ExampleIrohClusterNode_InferMeshPlan() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	cfg := Config{
		Dim:       2048,
		NumLayers: 16,
	}
	node, err := NewIrohClusterNode(ctx, nil, cfg, 0)
	if err != nil {
		return
	}
	defer node.Close()

	plan := node.InferMeshPlan(512, 32, 1)
	fmt.Printf("Standalone prefill equal decode: %t\n", plan.PrefillNodeID == plan.DecodeNodeID)
	fmt.Printf("Standalone transfer bytes: %d\n", plan.TransferBytes)
	fmt.Printf("Standalone component count: %d\n", len(plan.ComponentPlacements))

	// Output:
	// Standalone prefill equal decode: true
	// Standalone transfer bytes: 0
	// Standalone component count: 4
}
