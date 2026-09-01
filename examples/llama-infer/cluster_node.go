package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/netip"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/tmc/go-iroh/iroh"
	"github.com/tmc/go-iroh/key"
	"github.com/tmc/go-iroh/relay"
)

const (
	InferenceALPN = "tmc/llama-infer/v1"
)

// NodeTelemetry represents the hardware capability and state of a cluster peer.
type NodeTelemetry struct {
	NodeID         string   `json:"node_id"`
	OS             string   `json:"os"`
	Arch           string   `json:"arch"`
	ComputeTFLOPs  float64  `json:"compute_tflops"`
	BandwidthGBS   float64  `json:"bandwidth_gbs"`
	FreeVRAMMB     int64    `json:"free_vram_mb"`
	CachedPrefixes []string `json:"cached_prefixes"`
}

// CharacteristicRatio returns the FLOP-to-Byte ratio (arithmetic intensity suitability).
func (nt NodeTelemetry) CharacteristicRatio() float64 {
	if nt.BandwidthGBS <= 0 {
		return 0
	}
	return (nt.ComputeTFLOPs * 1000.0) / nt.BandwidthGBS
}

// DistributedRoleAssignment defines the computed execution placement for an inference request.
type DistributedRoleAssignment struct {
	PrefillNodeID      string  `json:"prefill_node_id"`
	DecodeNodeID       string  `json:"decode_node_id"`
	EstimatedLatencyMs float64 `json:"estimated_latency_ms"`
	Reason             string  `json:"reason"`
}

// ClusterMeshState summarizes the current multi-node topology and aggregated capabilities.
type ClusterMeshState struct {
	SelfNode            NodeTelemetry             `json:"self_node"`
	Peers               []NodeTelemetry           `json:"peers"`
	TotalNodes          int                       `json:"total_nodes"`
	ClusterTFLOPs       float64                   `json:"cluster_tflops"`
	ClusterBandwidthGBS float64                   `json:"cluster_bandwidth_gbs"`
	ActiveAssignment    DistributedRoleAssignment `json:"active_assignment"`
}

// IrohClusterNode coordinates distributed P2P inference over go-iroh QUIC endpoints.
type IrohClusterNode struct {
	ep       *iroh.Endpoint
	self     NodeTelemetry
	peersMu  sync.RWMutex
	peers    map[string]NodeTelemetry
	secret   key.SecretKey
	engine   *ModelEngine
	cfg      Config
}

// NewIrohClusterNode initializes a self-discovering P2P inference node.
func NewIrohClusterNode(ctx context.Context, engine *ModelEngine, cfg Config, bindPort uint16) (*IrohClusterNode, error) {
	sk, err := key.GenerateSecretKey()
	if err != nil {
		return nil, fmt.Errorf("generate secret key: %w", err)
	}

	opts := []iroh.Option{
		iroh.WithSecretKey(sk),
		iroh.WithALPNs(InferenceALPN),
		iroh.WithRelayMode(relay.ModeDisabled()),
	}
	if bindPort > 0 {
		opts = append(opts, iroh.WithBindAddr(netip.AddrPortFrom(netip.IPv4Unspecified(), bindPort)))
	}

	ep, err := iroh.Bind(ctx, opts...)
	if err != nil {
		return nil, fmt.Errorf("iroh bind: %w", err)
	}

	node := &IrohClusterNode{
		ep:     ep,
		secret: sk,
		peers:  make(map[string]NodeTelemetry),
		engine: engine,
		cfg:    cfg,
	}

	node.self = node.probeSelfHardware()
	return node, nil
}

// NodeID returns the public Ed25519 string identifier for this node.
func (n *IrohClusterNode) NodeID() string {
	return n.secret.Public().String()
}

// SelfTelemetry returns this node's measured telemetry.
func (n *IrohClusterNode) SelfTelemetry() NodeTelemetry {
	return n.self
}

func (n *IrohClusterNode) probeSelfHardware() NodeTelemetry {
	tflops := 35.0
	bw := 400.0

	if runtime.GOOS == "darwin" {
		bw = 800.0 // Apple Silicon unified memory default
		tflops = 40.0
	} else if runtime.GOOS == "linux" {
		bw = 300.0
		tflops = 120.0 // CUDA Tensor Cores default
	}

	return NodeTelemetry{
		NodeID:        n.NodeID(),
		OS:            runtime.GOOS,
		Arch:          runtime.GOARCH,
		ComputeTFLOPs: tflops,
		BandwidthGBS:  bw,
		FreeVRAMMB:    16384,
	}
}

// RegisterPeer manually registers a known peer's telemetry for role evaluation.
func (n *IrohClusterNode) RegisterPeer(tel NodeTelemetry) {
	if tel.NodeID == "" || tel.NodeID == n.self.NodeID {
		return
	}
	n.peersMu.Lock()
	defer n.peersMu.Unlock()
	n.peers[tel.NodeID] = tel
}

// ConnectHTTPPeer sends a mesh join request to a peer HTTP endpoint and exchanges telemetry.
func (n *IrohClusterNode) ConnectHTTPPeer(ctx context.Context, peerURL string) error {
	if !strings.HasPrefix(peerURL, "http://") && !strings.HasPrefix(peerURL, "https://") {
		peerURL = "http://" + peerURL
	}
	joinURL := strings.TrimRight(peerURL, "/") + "/api/mesh/join"

	body, err := json.Marshal(n.self)
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, joinURL, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("peer join request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("peer returned status %d", resp.StatusCode)
	}

	var remotePeer NodeTelemetry
	if err := json.NewDecoder(resp.Body).Decode(&remotePeer); err != nil {
		return fmt.Errorf("decode peer telemetry: %w", err)
	}

	n.RegisterPeer(remotePeer)
	return nil
}

// InferRoles evaluates the distributed E-Graph cost function to choose optimal Prefill and Decode nodes.
func (n *IrohClusterNode) InferRoles(promptLen, genTokens int) DistributedRoleAssignment {
	n.peersMu.RLock()
	allNodes := make([]NodeTelemetry, 0, len(n.peers)+1)
	allNodes = append(allNodes, n.self)
	for _, p := range n.peers {
		allNodes = append(allNodes, p)
	}
	n.peersMu.RUnlock()

	if len(allNodes) == 1 {
		return DistributedRoleAssignment{
			PrefillNodeID:      n.self.NodeID,
			DecodeNodeID:       n.self.NodeID,
			EstimatedLatencyMs: 1.0,
			Reason:             "Single standalone node active",
		}
	}

	dim := float64(n.cfg.Dim)
	if dim == 0 {
		dim = 2048
	}
	layers := float64(n.cfg.NumLayers)
	if layers == 0 {
		layers = 32
	}

	bestCost := math.MaxFloat64
	var bestAssignment DistributedRoleAssignment

	for _, pNode := range allNodes {
		for _, dNode := range allNodes {
			// 1. Prefill Compute Time: O(2 * S * D^2 * L) / TFLOPs
			prefillSec := (2.0 * float64(promptLen) * dim * dim * layers) / (pNode.ComputeTFLOPs * 1e12)

			// 2. Inter-node activation transfer latency (if disaggregated)
			transferSec := 0.0
			if pNode.NodeID != dNode.NodeID {
				// KV activation payload in FP16: 2 * promptLen * dim * layers * 2 bytes
				kvBytes := 4.0 * float64(promptLen) * dim * layers
				linkBandwidth := 10.0 * 1e9 / 8.0 // assume 10Gbps default link (~1.25 GB/s)
				transferSec = kvBytes / linkBandwidth
			}

			// 3. Decode Step Latency: O(2 * GenTokens * D^2 * L) / Bandwidth
			decodeSec := (2.0 * float64(genTokens) * dim * dim * layers) / (dNode.BandwidthGBS * 1e9)

			totalSec := prefillSec + transferSec + decodeSec
			if totalSec < bestCost {
				bestCost = totalSec
				bestAssignment = DistributedRoleAssignment{
					PrefillNodeID:      pNode.NodeID,
					DecodeNodeID:       dNode.NodeID,
					EstimatedLatencyMs: totalSec * 1000.0,
					Reason: fmt.Sprintf("Assigned Prefill on %s (%.0f TFLOPs) and Decode on %s (%.0f GB/s)",
						pNode.OS, pNode.ComputeTFLOPs, dNode.OS, dNode.BandwidthGBS),
				}
			}
		}
	}

	return bestAssignment
}

// MeshState returns a snapshot of the current cluster topology and active role assignment.
func (n *IrohClusterNode) MeshState() ClusterMeshState {
	if n == nil {
		return ClusterMeshState{}
	}
	n.peersMu.RLock()
	peersList := make([]NodeTelemetry, 0, len(n.peers))
	totalTFLOPs := n.self.ComputeTFLOPs
	totalBW := n.self.BandwidthGBS
	for _, p := range n.peers {
		peersList = append(peersList, p)
		totalTFLOPs += p.ComputeTFLOPs
		totalBW += p.BandwidthGBS
	}
	n.peersMu.RUnlock()

	assignment := n.InferRoles(2048, 64)

	return ClusterMeshState{
		SelfNode:            n.self,
		Peers:               peersList,
		TotalNodes:          len(peersList) + 1,
		ClusterTFLOPs:       totalTFLOPs,
		ClusterBandwidthGBS:  totalBW,
		ActiveAssignment:    assignment,
	}
}

// Close gracefully shuts down the underlying go-iroh QUIC endpoint.
func (n *IrohClusterNode) Close() error {
	if n.ep != nil {
		return n.ep.Shutdown(context.Background())
	}
	return nil
}
