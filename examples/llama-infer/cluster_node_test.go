package main

import (
	"context"
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
