//go:build !tpu

package main

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/netip"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/tmc/go-iroh/iroh"
	"github.com/tmc/go-iroh/key"
	"github.com/tmc/go-iroh/netaddr"
	"github.com/tmc/go-iroh/relay"
	"github.com/tmc/lib/hfsync"
)

const (
	InferenceALPN = "tmc/llama-infer/v1"
)

const (
	draftTargetKind    = "draft-target"
	remoteGenerateKind = "generate"
)

// inferenceRequest is a message sent on an Iroh inference stream. Empty Kind
// preserves the original telemetry handshake.
type inferenceRequest struct {
	Kind      string        `json:"kind,omitempty"`
	Telemetry NodeTelemetry `json:"telemetry,omitempty"`
	Prompt    string        `json:"prompt,omitempty"`
	MaxTokens int           `json:"max_tokens,omitempty"`
}

type inferenceResponse struct {
	Telemetry NodeTelemetry `json:"telemetry,omitempty"`
	Tokens    []int32       `json:"tokens,omitempty"`
	Text      string        `json:"text,omitempty"`
	Error     string        `json:"error,omitempty"`
}

// RemoteGeneration is an inference result produced on a selected mesh node.
// Text and token IDs are both returned by that node's engine.
type RemoteGeneration struct {
	NodeID string  `json:"node_id"`
	Tokens []int32 `json:"tokens"`
	Text   string  `json:"text"`
}

// NodeTelemetry represents the hardware capability and state of a cluster peer.
type NodeTelemetry struct {
	NodeID          string                `json:"node_id"`
	EndpointAddrs   []string              `json:"endpoint_addrs,omitempty"`
	OS              string                `json:"os"`
	Arch            string                `json:"arch"`
	ComputeTFLOPs   float64               `json:"compute_tflops"`
	BandwidthGBS    float64               `json:"bandwidth_gbs"`
	FreeVRAMMB      int64                 `json:"free_vram_mb"`
	CachedPrefixes  []string              `json:"cached_prefixes"`
	AvailableModels []hfsync.ModelSummary `json:"available_models"`
}

// CharacteristicRatio returns the FLOP-to-Byte ratio (arithmetic intensity suitability).
func (nt NodeTelemetry) CharacteristicRatio() float64 {
	if nt.BandwidthGBS <= 0 {
		return 0
	}
	return (nt.ComputeTFLOPs * 1000.0) / nt.BandwidthGBS
}

// PeerLinkMetrics contains measured network latency and bandwidth for a peer link.
type PeerLinkMetrics struct {
	RTT          time.Duration `json:"rtt"`
	BandwidthGBS float64       `json:"bandwidth_gbs"`
}

// MeshComponent identifies a transformer graph component.
type MeshComponent string

const (
	ComponentPrefill   MeshComponent = "prefill"
	ComponentDecode    MeshComponent = "decode"
	ComponentAttention MeshComponent = "attention_gqa"
	ComponentMLP       MeshComponent = "mlp_gemm"
)

// ComponentPlacement records the execution node and estimated latency for a graph component.
type ComponentPlacement struct {
	Component MeshComponent `json:"component"`
	NodeID    string        `json:"node_id"`
	LatencyMs float64       `json:"latency_ms"`
}

// MeshExecutionPlan represents the partitioned execution schedule across cluster nodes.
type MeshExecutionPlan struct {
	PrefillNodeID       string               `json:"prefill_node_id"`
	DecodeNodeID        string               `json:"decode_node_id"`
	AttentionNodeID     string               `json:"attention_node_id,omitempty"`
	MLPNodeID           string               `json:"mlp_node_id,omitempty"`
	ComponentPlacements []ComponentPlacement `json:"component_placements,omitempty"`

	EstimatedLatencyMs float64 `json:"estimated_latency_ms"`
	ComputeLatencyMs   float64 `json:"compute_latency_ms"`
	NetworkLatencyMs   float64 `json:"network_latency_ms"`
	TransferBytes      int64   `json:"transfer_bytes"`
	Reason             string  `json:"reason"`
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
	ActivePlan          MeshExecutionPlan         `json:"active_plan,omitempty"`
}

// IrohClusterNode coordinates distributed P2P inference over go-iroh QUIC endpoints.
type IrohClusterNode struct {
	ep        *iroh.Endpoint
	router    *iroh.Router
	self      NodeTelemetry
	peersMu   sync.RWMutex
	peers     map[string]NodeTelemetry
	peerAddrs map[string]netaddr.EndpointAddr
	peerLinks map[string]PeerLinkMetrics
	stateMu   sync.Mutex
	secret    key.SecretKey
	engine    *ModelEngine
	cfg       Config
	peerState string
	hfStore   *hfsync.Store
	hfServer  *hfsync.Server
	hfClient  *hfsync.Client
}

// NewIrohClusterNode initializes a self-discovering P2P inference node.
func NewIrohClusterNode(ctx context.Context, engine *ModelEngine, cfg Config, bindPort uint16) (*IrohClusterNode, error) {
	sk, err := nodeSecretKey(cfg.NodeKeyPath)
	if err != nil {
		return nil, fmt.Errorf("load node secret key: %w", err)
	}

	opts := []iroh.Option{
		iroh.WithSecretKey(sk),
		iroh.WithRelayMode(relay.ModeDefault()),
	}
	if bindPort > 0 {
		opts = append(opts, iroh.WithBindAddr(netip.AddrPortFrom(netip.IPv4Unspecified(), bindPort)))
	}

	ep, err := iroh.Bind(ctx, opts...)
	if err != nil {
		return nil, fmt.Errorf("iroh bind: %w", err)
	}

	hfStore := hfsync.NewStore("")
	hfServer := hfsync.NewServer(ep, hfStore)
	hfClient := hfsync.NewClient(ep, hfStore)

	node := &IrohClusterNode{
		ep:        ep,
		secret:    sk,
		peers:     make(map[string]NodeTelemetry),
		peerAddrs: make(map[string]netaddr.EndpointAddr),
		peerLinks: make(map[string]PeerLinkMetrics),
		engine:    engine,
		cfg:       cfg,
		peerState: cfg.PeerStatePath,
		hfStore:   hfStore,
		hfServer:  hfServer,
		hfClient:  hfClient,
	}

	node.self = node.probeSelfHardware()
	if err := node.loadPeers(); err != nil {
		_ = ep.Shutdown(context.Background())
		return nil, fmt.Errorf("load mesh peers: %w", err)
	}
	router, err := iroh.NewRouter(ep, map[string]iroh.ProtocolHandler{
		InferenceALPN: iroh.ProtocolHandlerFunc(node.handleInferenceConnection),
		hfsync.ALPN:   iroh.ProtocolHandlerFunc(hfServer.HandleConnection),
	}, nil)
	if err != nil {
		_ = ep.Shutdown(context.Background())
		return nil, fmt.Errorf("start iroh protocol router: %w", err)
	}
	node.router = router
	return node, nil
}

func defaultNodeKeyPath() string {
	cacheDir, err := os.UserCacheDir()
	if err != nil {
		return ""
	}
	return filepath.Join(cacheDir, "tmc", "llama-infer", "iroh.key")
}

func defaultPeerStatePath() string {
	cacheDir, err := os.UserCacheDir()
	if err != nil {
		return ""
	}
	return filepath.Join(cacheDir, "tmc", "llama-infer", "peers.json")
}

func (n *IrohClusterNode) loadPeers() error {
	if n.peerState == "" {
		return nil
	}
	data, err := os.ReadFile(n.peerState)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read %q: %w", n.peerState, err)
	}
	var peers []NodeTelemetry
	if err := json.Unmarshal(data, &peers); err != nil {
		return fmt.Errorf("decode %q: %w", n.peerState, err)
	}
	for _, peer := range peers {
		if err := n.registerPeer(peer, false); err != nil {
			return fmt.Errorf("restore peer %q: %w", peer.NodeID, err)
		}
	}
	return nil
}

func (n *IrohClusterNode) savePeers(peers []NodeTelemetry) error {
	if n.peerState == "" {
		return nil
	}
	data, err := json.Marshal(peers)
	if err != nil {
		return fmt.Errorf("encode mesh peers: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(n.peerState), 0700); err != nil {
		return fmt.Errorf("create peer state directory: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(n.peerState), ".peers-*")
	if err != nil {
		return fmt.Errorf("create peer state file: %w", err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if err := tmp.Chmod(0600); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("set peer state permissions: %w", err)
	}
	if _, err := tmp.Write(append(data, '\n')); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("write peer state: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close peer state: %w", err)
	}
	if err := os.Rename(tmpName, n.peerState); err != nil {
		return fmt.Errorf("replace peer state: %w", err)
	}
	return nil
}

func nodeSecretKey(path string) (key.SecretKey, error) {
	if path == "" {
		return key.GenerateSecretKey()
	}

	data, err := os.ReadFile(path)
	if err == nil {
		sk, err := key.ParseSecretKey(strings.TrimSpace(string(data)))
		if err != nil {
			return key.SecretKey{}, fmt.Errorf("parse %q: %w", path, err)
		}
		return sk, nil
	}
	if !errors.Is(err, os.ErrNotExist) {
		return key.SecretKey{}, fmt.Errorf("read %q: %w", path, err)
	}

	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return key.SecretKey{}, fmt.Errorf("create key directory: %w", err)
	}
	sk, err := key.GenerateSecretKey()
	if err != nil {
		return key.SecretKey{}, err
	}
	seed := sk.Bytes()
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if errors.Is(err, os.ErrExist) {
		return nodeSecretKey(path)
	}
	if err != nil {
		return key.SecretKey{}, fmt.Errorf("create %q: %w", path, err)
	}
	_, writeErr := f.WriteString(hex.EncodeToString(seed[:]) + "\n")
	closeErr := f.Close()
	if writeErr != nil {
		return key.SecretKey{}, fmt.Errorf("write %q: %w", path, writeErr)
	}
	if closeErr != nil {
		return key.SecretKey{}, fmt.Errorf("close %q: %w", path, closeErr)
	}
	return sk, nil
}

// Close gracefully shuts down the Iroh endpoint and protocol router.
func (n *IrohClusterNode) Close() error {
	if n.router != nil {
		_ = n.router.Shutdown(context.Background())
	}
	if n.ep != nil {
		return n.ep.Shutdown(context.Background())
	}
	return nil
}

func (n *IrohClusterNode) handleInferenceConnection(ctx context.Context, conn *iroh.Conn) error {
	stream, err := conn.AcceptStream(ctx)
	if err != nil {
		return err
	}
	defer stream.Close()

	var raw json.RawMessage
	if err := json.NewDecoder(stream).Decode(&raw); err != nil {
		return fmt.Errorf("decode peer telemetry: %w", err)
	}
	var request inferenceRequest
	if err := json.Unmarshal(raw, &request); err != nil {
		return fmt.Errorf("decode inference request: %w", err)
	}
	if request.Kind == draftTargetKind {
		if err := n.RegisterPeer(request.Telemetry); err != nil {
			return fmt.Errorf("register peer: %w", err)
		}
		if n.engine == nil {
			return json.NewEncoder(stream).Encode(inferenceResponse{Error: "target engine unavailable"})
		}
		tokens, err := n.engine.GenerateTokenIDs(ctx, n.cfg, request.Prompt, request.MaxTokens)
		if err != nil {
			return json.NewEncoder(stream).Encode(inferenceResponse{Error: err.Error()})
		}
		return json.NewEncoder(stream).Encode(inferenceResponse{Tokens: tokens})
	}
	if request.Kind == remoteGenerateKind {
		if err := n.RegisterPeer(request.Telemetry); err != nil {
			return fmt.Errorf("register peer: %w", err)
		}
		if n.engine == nil {
			return json.NewEncoder(stream).Encode(inferenceResponse{Error: "target engine unavailable"})
		}
		var text strings.Builder
		var tokens []int32
		err := n.engine.Generate(ctx, n.cfg, request.Prompt, request.MaxTokens, func(tokenID int32, tokenText string) bool {
			tokens = append(tokens, tokenID)
			text.WriteString(tokenText)
			return true
		})
		if err != nil {
			return json.NewEncoder(stream).Encode(inferenceResponse{Error: err.Error()})
		}
		return json.NewEncoder(stream).Encode(inferenceResponse{Tokens: tokens, Text: text.String()})
	}

	var peerTel NodeTelemetry
	if err := json.Unmarshal(raw, &peerTel); err != nil {
		return fmt.Errorf("decode peer telemetry: %w", err)
	}
	if err := n.RegisterPeer(peerTel); err != nil {
		return fmt.Errorf("register peer: %w", err)
	}
	if err := json.NewEncoder(stream).Encode(n.SelfTelemetry()); err != nil {
		return fmt.Errorf("send self telemetry: %w", err)
	}
	return nil
}

// RequestDraftTarget asks a registered peer to generate exact token IDs for a
// prompt. The peer runs its own configured engine; this method never derives
// target tokens from local model metadata.
func (n *IrohClusterNode) RequestDraftTarget(ctx context.Context, nodeID, prompt string, maxTokens int) ([]int32, error) {
	if n == nil || n.ep == nil {
		return nil, errors.New("iroh node unavailable")
	}
	if nodeID == "" || nodeID == n.NodeID() {
		return nil, errors.New("draft target must be a remote node")
	}
	if maxTokens <= 0 {
		return nil, errors.New("draft token count must be positive")
	}
	addr, err := n.PeerAddr(nodeID)
	if err != nil {
		return nil, fmt.Errorf("resolve draft target: %w", err)
	}
	conn, err := n.connectInference(ctx, addr)
	if err != nil {
		return nil, fmt.Errorf("connect draft target: %w", err)
	}
	stream, err := conn.OpenStreamSync(ctx)
	if err != nil {
		return nil, fmt.Errorf("open draft target stream: %w", err)
	}
	defer stream.Close()
	request := inferenceRequest{
		Kind:      draftTargetKind,
		Telemetry: n.SelfTelemetry(),
		Prompt:    prompt,
		MaxTokens: maxTokens,
	}
	if err := json.NewEncoder(stream).Encode(request); err != nil {
		return nil, fmt.Errorf("send draft target request: %w", err)
	}
	var response inferenceResponse
	if err := json.NewDecoder(stream).Decode(&response); err != nil {
		return nil, fmt.Errorf("receive draft target response: %w", err)
	}
	if response.Error != "" {
		return nil, fmt.Errorf("draft target: %s", response.Error)
	}
	return response.Tokens, nil
}

// GenerateRemote runs a complete generation on a selected registered peer.
// Streaming is intentionally not multiplexed through this request-response
// protocol; callers receive a completed result from the selected node.
func (n *IrohClusterNode) GenerateRemote(ctx context.Context, nodeID, prompt string, maxTokens int) (RemoteGeneration, error) {
	if n == nil || n.ep == nil {
		return RemoteGeneration{}, errors.New("iroh node unavailable")
	}
	if nodeID == "" || nodeID == n.NodeID() {
		return RemoteGeneration{}, errors.New("remote execution node required")
	}
	if maxTokens <= 0 {
		return RemoteGeneration{}, errors.New("maximum token count must be positive")
	}
	addr, err := n.PeerAddr(nodeID)
	if err != nil {
		return RemoteGeneration{}, fmt.Errorf("resolve execution node: %w", err)
	}
	conn, err := n.connectInference(ctx, addr)
	if err != nil {
		return RemoteGeneration{}, fmt.Errorf("connect execution node: %w", err)
	}
	stream, err := conn.OpenStreamSync(ctx)
	if err != nil {
		return RemoteGeneration{}, fmt.Errorf("open execution stream: %w", err)
	}
	defer stream.Close()
	request := inferenceRequest{
		Kind:      remoteGenerateKind,
		Telemetry: n.SelfTelemetry(),
		Prompt:    prompt,
		MaxTokens: maxTokens,
	}
	if err := json.NewEncoder(stream).Encode(request); err != nil {
		return RemoteGeneration{}, fmt.Errorf("send generation request: %w", err)
	}
	var response inferenceResponse
	if err := json.NewDecoder(stream).Decode(&response); err != nil {
		return RemoteGeneration{}, fmt.Errorf("receive generation response: %w", err)
	}
	if response.Error != "" {
		return RemoteGeneration{}, fmt.Errorf("remote generation: %s", response.Error)
	}
	return RemoteGeneration{NodeID: nodeID, Tokens: response.Tokens, Text: response.Text}, nil
}

// Speculate compares local draft tokens with exact tokens generated by a
// registered remote target. The result is recorded only after both engines
// completed their work.
func (n *IrohClusterNode) Speculate(ctx context.Context, targetNodeID, prompt string, maxTokens int) (SpeculationTrace, error) {
	if n == nil || n.engine == nil {
		return SpeculationTrace{}, errors.New("draft engine unavailable")
	}
	draft, err := n.engine.GenerateTokenIDs(ctx, n.cfg, prompt, maxTokens)
	if err != nil {
		return SpeculationTrace{}, fmt.Errorf("generate draft tokens: %w", err)
	}
	target, err := n.RequestDraftTarget(ctx, targetNodeID, prompt, maxTokens)
	if err != nil {
		return SpeculationTrace{}, err
	}
	trace := SpeculationTrace{
		When:         time.Now(),
		DraftNodeID:  n.NodeID(),
		TargetNodeID: targetNodeID,
		Draft:        draft,
		Target:       target,
	}
	for i, token := range draft {
		parent := ""
		if i > 0 {
			parent = fmt.Sprintf("d%d", i-1)
		}
		state := "rejected"
		if i < len(target) && token == target[i] {
			state = "accepted"
			trace.Accepted++
		}
		engine := "draft"
		if state == "accepted" {
			engine = "both"
		}
		trace.Nodes = append(trace.Nodes, SpeculationNode{
			ID: fmt.Sprintf("d%d", i), Parent: parent, Token: token, Engine: engine, State: state,
		})
		if state == "rejected" {
			if i < len(target) {
				trace.Nodes = append(trace.Nodes, SpeculationNode{
					ID: fmt.Sprintf("t%d", i), Parent: parent, Token: target[i], Engine: "target", State: "target",
				})
			}
			break
		}
	}
	n.engine.metrics.RecordSpeculation(trace)
	return trace, nil
}

// Addr returns the complete netaddr.EndpointAddr including DERP relay URLs and direct IP endpoints.
func (n *IrohClusterNode) Addr() netaddr.EndpointAddr {
	if n.ep == nil {
		return netaddr.EndpointAddr{}
	}
	return n.ep.Addr()
}

func (n *IrohClusterNode) connectInference(ctx context.Context, addr netaddr.EndpointAddr) (*iroh.Conn, error) {
	if n == nil || n.ep == nil {
		return nil, errors.New("iroh node unavailable")
	}
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		conn, err := n.ep.Connect(ctx, addr, InferenceALPN)
		if err == nil {
			return conn, nil
		}
		lastErr = err
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(100 * time.Millisecond):
		}
	}
	return nil, fmt.Errorf("iroh connect after retries: %w", lastErr)
}

// ConnectIroh connects to a remote peer over pure Iroh using its EndpointAddr.
func (n *IrohClusterNode) ConnectIroh(ctx context.Context, addr netaddr.EndpointAddr) error {
	start := time.Now()
	conn, err := n.connectInference(ctx, addr)
	if err != nil {
		return fmt.Errorf("iroh connect: %w", err)
	}
	handshakeRTT := time.Since(start)

	stream, err := conn.OpenStreamSync(ctx)
	if err != nil {
		return fmt.Errorf("iroh open stream: %w", err)
	}
	defer stream.Close()

	if err := json.NewEncoder(stream).Encode(n.self); err != nil {
		return fmt.Errorf("send telemetry: %w", err)
	}

	var peerTel NodeTelemetry
	if err := json.NewDecoder(stream).Decode(&peerTel); err != nil {
		return fmt.Errorf("receive peer telemetry: %w", err)
	}

	if err := n.RegisterPeer(peerTel); err != nil {
		return fmt.Errorf("register peer: %w", err)
	}
	// Keep the address that just completed a handshake. A peer's advertised
	// telemetry can contain only a relay address even when this connection used
	// a working direct route.
	n.peersMu.Lock()
	n.peerAddrs[peerTel.NodeID] = addr
	if n.peerLinks == nil {
		n.peerLinks = make(map[string]PeerLinkMetrics)
	}
	if existing, ok := n.peerLinks[peerTel.NodeID]; !ok || existing.RTT == 0 {
		n.peerLinks[peerTel.NodeID] = PeerLinkMetrics{
			RTT:          handshakeRTT,
			BandwidthGBS: 1.25,
		}
	}
	n.peersMu.Unlock()
	return nil
}

// NodeID returns the public Ed25519 string identifier for this node.
func (n *IrohClusterNode) NodeID() string {
	return n.secret.Public().String()
}

// SelfTelemetry returns this node's measured telemetry.
func (n *IrohClusterNode) SelfTelemetry() NodeTelemetry {
	n.peersMu.RLock()
	defer n.peersMu.RUnlock()
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

	var models []hfsync.ModelSummary
	if n.hfStore != nil {
		models, _ = n.hfStore.ListModels()
	}

	return NodeTelemetry{
		NodeID:          n.NodeID(),
		EndpointAddrs:   endpointAddrStrings(n.Addr()),
		OS:              runtime.GOOS,
		Arch:            runtime.GOARCH,
		ComputeTFLOPs:   tflops,
		BandwidthGBS:    bw,
		FreeVRAMMB:      16384,
		AvailableModels: models,
	}
}

// SyncModel synchronizes a model snapshot from a peer into the local Hugging Face store with progress tracking.
func (n *IrohClusterNode) SyncModel(ctx context.Context, peerAddr netaddr.EndpointAddr, modelID string, onProgress hfsync.ProgressFunc) (*hfsync.SyncStats, error) {
	if n.hfClient == nil {
		return nil, fmt.Errorf("hfsync client unavailable")
	}
	stats, err := n.hfClient.SyncWithProgress(ctx, peerAddr, modelID, onProgress)
	if err == nil {
		n.peersMu.Lock()
		n.self.AvailableModels, _ = n.hfStore.ListModels()
		n.peersMu.Unlock()
	}
	return stats, err
}

// SyncFromPeer synchronizes a model from a mesh peer selected by node ID.
func (n *IrohClusterNode) SyncFromPeer(ctx context.Context, nodeID, modelID string, onProgress hfsync.ProgressFunc) (*hfsync.SyncStats, error) {
	addr, err := n.PeerAddr(nodeID)
	if err != nil {
		return nil, err
	}
	return n.SyncModel(ctx, addr, modelID, onProgress)
}

// RegisterPeer manually registers a known peer's telemetry for role evaluation
// and saves it when peer-state persistence is configured.
func (n *IrohClusterNode) RegisterPeer(tel NodeTelemetry) error {
	return n.registerPeer(tel, true)
}

func (n *IrohClusterNode) registerPeer(tel NodeTelemetry, persist bool) error {
	if tel.NodeID == "" {
		return errors.New("peer node ID is empty")
	}
	if tel.NodeID == n.self.NodeID {
		return nil
	}
	if persist {
		n.stateMu.Lock()
		defer n.stateMu.Unlock()
	}
	n.peersMu.Lock()
	n.peers[tel.NodeID] = tel
	if addr, err := endpointAddrFromTelemetry(tel); err == nil {
		n.peerAddrs[tel.NodeID] = addr
	} else {
		delete(n.peerAddrs, tel.NodeID)
	}
	peers := make([]NodeTelemetry, 0, len(n.peers))
	for _, peer := range n.peers {
		peers = append(peers, peer)
	}
	n.peersMu.Unlock()
	if !persist {
		return nil
	}
	return n.savePeers(peers)
}

// PeerAddr returns a mesh peer's advertised Iroh endpoint address.
func (n *IrohClusterNode) PeerAddr(nodeID string) (netaddr.EndpointAddr, error) {
	n.peersMu.RLock()
	defer n.peersMu.RUnlock()
	addr, ok := n.peerAddrs[nodeID]
	if !ok || addr.IsEmpty() {
		return netaddr.EndpointAddr{}, fmt.Errorf("peer %q has no Iroh endpoint address", nodeID)
	}
	return addr, nil
}

func endpointAddrStrings(addr netaddr.EndpointAddr) []string {
	addrs := addr.Addrs()
	out := make([]string, 0, len(addrs))
	for _, addr := range addrs {
		out = append(out, addr.String())
	}
	return out
}

func endpointAddrFromTelemetry(tel NodeTelemetry) (netaddr.EndpointAddr, error) {
	var id key.EndpointID
	if err := id.UnmarshalText([]byte(tel.NodeID)); err != nil {
		return netaddr.EndpointAddr{}, fmt.Errorf("parse peer node ID: %w", err)
	}
	transportAddrs := make([]netaddr.TransportAddr, 0, len(tel.EndpointAddrs))
	for _, text := range tel.EndpointAddrs {
		addr, err := netaddr.ParseTransportAddr(text)
		if err != nil {
			return netaddr.EndpointAddr{}, fmt.Errorf("parse peer endpoint address: %w", err)
		}
		transportAddrs = append(transportAddrs, addr)
	}
	return netaddr.NewEndpointAddr(id, transportAddrs...), nil
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

	if err := n.RegisterPeer(remotePeer); err != nil {
		return fmt.Errorf("register peer: %w", err)
	}
	return nil
}

// SetPeerLinkMetrics records measured network telemetry for a peer.
func (n *IrohClusterNode) SetPeerLinkMetrics(nodeID string, metrics PeerLinkMetrics) {
	n.peersMu.Lock()
	defer n.peersMu.Unlock()
	if n.peerLinks == nil {
		n.peerLinks = make(map[string]PeerLinkMetrics)
	}
	n.peerLinks[nodeID] = metrics
}

// PeerLink returns measured or fallback link metrics for the specified node ID.
func (n *IrohClusterNode) PeerLink(nodeID string) PeerLinkMetrics {
	n.peersMu.RLock()
	defer n.peersMu.RUnlock()
	return n.peerLinkLocked(nodeID)
}

func (n *IrohClusterNode) peerLinkLocked(nodeID string) PeerLinkMetrics {
	if nodeID == n.self.NodeID {
		bw := n.self.BandwidthGBS
		if bw <= 0 {
			bw = 400.0
		}
		return PeerLinkMetrics{
			RTT:          0,
			BandwidthGBS: bw,
		}
	}
	if n.peerLinks != nil {
		if m, ok := n.peerLinks[nodeID]; ok && (m.RTT > 0 || m.BandwidthGBS > 0) {
			bw := m.BandwidthGBS
			if bw <= 0 {
				bw = 1.25 // default 10 Gbps (~1.25 GB/s)
			}
			return PeerLinkMetrics{
				RTT:          m.RTT,
				BandwidthGBS: bw,
			}
		}
	}
	// Fallback default: 1ms RTT, 10 Gbps (1.25 GB/s)
	return PeerLinkMetrics{
		RTT:          time.Millisecond,
		BandwidthGBS: 1.25,
	}
}

func (n *IrohClusterNode) linkBetween(uID, vID string) PeerLinkMetrics {
	if uID == vID {
		return n.peerLinkLocked(uID)
	}
	if uID == n.self.NodeID {
		return n.peerLinkLocked(vID)
	}
	if vID == n.self.NodeID {
		return n.peerLinkLocked(uID)
	}
	mU := n.peerLinkLocked(uID)
	mV := n.peerLinkLocked(vID)
	rtt := mU.RTT + mV.RTT
	bw := mU.BandwidthGBS
	if mV.BandwidthGBS > 0 && mV.BandwidthGBS < bw {
		bw = mV.BandwidthGBS
	}
	if bw <= 0 {
		bw = 1.25
	}
	return PeerLinkMetrics{
		RTT:          rtt,
		BandwidthGBS: bw,
	}
}

// InferMeshPlan evaluates the distributed E-Graph cost function to determine the optimal
// partitioned execution schedule across cluster nodes, taking into account peer link metrics
// (RTT and throughput), arithmetic intensity, and tensor transfer overheads.
func (n *IrohClusterNode) InferMeshPlan(promptLen, genTokens, batchSize int) MeshExecutionPlan {
	if n == nil {
		return MeshExecutionPlan{}
	}

	n.peersMu.RLock()
	allNodes := make([]NodeTelemetry, 0, len(n.peers)+1)
	allNodes = append(allNodes, n.self)
	for _, p := range n.peers {
		allNodes = append(allNodes, p)
	}
	n.peersMu.RUnlock()

	dim := float64(n.cfg.Dim)
	if dim <= 0 {
		dim = 2048
	}
	layers := float64(n.cfg.NumLayers)
	if layers <= 0 {
		layers = 32
	}
	hiddenDim := float64(n.cfg.HiddenDim)
	if hiddenDim <= 0 {
		hiddenDim = math.Floor(8.0 / 3.0 * dim)
	}
	kvDim := dim
	if n.cfg.NumHeads > 0 && n.cfg.NumKVHeads > 0 {
		kvDim = dim * float64(n.cfg.NumKVHeads) / float64(n.cfg.NumHeads)
	} else {
		kvDim = dim / 4.0
	}
	dtypeSize := 2.0
	if strings.EqualFold(n.cfg.Precision, "f32") {
		dtypeSize = 4.0
	}
	batch := float64(batchSize)
	if batch <= 0 {
		batch = 1.0
	}
	sPrompt := float64(promptLen)
	if sPrompt <= 0 {
		sPrompt = 1.0
	}
	sGen := float64(genTokens)
	if sGen <= 0 {
		sGen = 1.0
	}

	// 1. Prefill Attention Compute FLOPs (QKV + Score + Out projections across L layers)
	attnPrefillFLOPs := layers * 2.0 * batch * sPrompt * (2.0*dim*dim + 2.0*dim*kvDim + sPrompt*dim)
	// Prefill Dense MLP / MoE GEMM FLOPs (Gate + Up + Down projections)
	mlpPrefillFLOPs := layers * 6.0 * batch * sPrompt * dim * hiddenDim
	totalPrefillFLOPs := attnPrefillFLOPs + mlpPrefillFLOPs

	// 2. Decode Memory Bytes and Compute FLOPs across S_g autoregressive steps
	attnWeightBytes := layers * (2.0*dim*dim + 2.0*dim*kvDim) * dtypeSize
	mlpWeightBytes := layers * 3.0 * dim * hiddenDim * dtypeSize
	avgKVCacheBytes := 2.0 * layers * batch * (sPrompt + sGen/2.0) * kvDim * dtypeSize

	attnDecodeBytes := sGen * (attnWeightBytes + avgKVCacheBytes)
	mlpDecodeBytes := sGen * mlpWeightBytes
	totalDecodeBytes := attnDecodeBytes + mlpDecodeBytes

	attnDecodeFLOPs := sGen * layers * 2.0 * batch * (2.0*dim*dim + 2.0*dim*kvDim + (sPrompt+sGen/2.0)*dim)
	mlpDecodeFLOPs := sGen * layers * 6.0 * batch * dim * hiddenDim
	totalDecodeFLOPs := attnDecodeFLOPs + mlpDecodeFLOPs

	if len(allNodes) == 1 {
		cOps := math.Max(n.self.ComputeTFLOPs*1e12, 1e9)
		bBytes := math.Max(n.self.BandwidthGBS*1e9, 1e9)
		prefillSec := totalPrefillFLOPs / cOps
		decodeSec := math.Max(totalDecodeBytes/bBytes, totalDecodeFLOPs/cOps)
		totalSec := prefillSec + decodeSec
		return MeshExecutionPlan{
			PrefillNodeID:   n.self.NodeID,
			DecodeNodeID:    n.self.NodeID,
			AttentionNodeID: n.self.NodeID,
			MLPNodeID:       n.self.NodeID,
			ComponentPlacements: []ComponentPlacement{
				{Component: ComponentPrefill, NodeID: n.self.NodeID, LatencyMs: prefillSec * 1000.0},
				{Component: ComponentDecode, NodeID: n.self.NodeID, LatencyMs: decodeSec * 1000.0},
				{Component: ComponentAttention, NodeID: n.self.NodeID, LatencyMs: (attnDecodeBytes / bBytes) * 1000.0},
				{Component: ComponentMLP, NodeID: n.self.NodeID, LatencyMs: (mlpPrefillFLOPs / cOps) * 1000.0},
			},
			EstimatedLatencyMs: totalSec * 1000.0,
			ComputeLatencyMs:   totalSec * 1000.0,
			NetworkLatencyMs:   0,
			TransferBytes:      0,
			Reason:             "Single standalone node active",
		}
	}

	bestCost := math.MaxFloat64
	var bestP, bestD NodeTelemetry
	var bestPrefillSec, bestDecodeSec, bestNetSec float64
	var bestTransferBytes int64

	for _, pNode := range allNodes {
		pCompute := math.Max(pNode.ComputeTFLOPs*1e12, 1e9)
		pBandwidth := math.Max(pNode.BandwidthGBS*1e9, 1e9)
		pTime := math.Max(totalPrefillFLOPs/pCompute, (layers*(2.0*dim*dim+2.0*dim*kvDim+3.0*dim*hiddenDim)*dtypeSize)/pBandwidth)

		for _, dNode := range allNodes {
			dCompute := math.Max(dNode.ComputeTFLOPs*1e12, 1e9)
			dBandwidth := math.Max(dNode.BandwidthGBS*1e9, 1e9)
			dTime := math.Max(totalDecodeBytes/dBandwidth, totalDecodeFLOPs/dCompute)

			var transferSec float64
			var transBytes int64
			if pNode.NodeID != dNode.NodeID {
				activationBytes := int64(batch * sPrompt * dim * dtypeSize)
				kvCacheBytes := int64(2.0 * layers * batch * sPrompt * kvDim * dtypeSize)
				transBytes = activationBytes + kvCacheBytes

				link := n.linkBetween(pNode.NodeID, dNode.NodeID)
				linkBWBytes := math.Max(link.BandwidthGBS*1e9, 1e6)
				transferSec = link.RTT.Seconds() + float64(transBytes)/linkBWBytes
			}

			totalSec := pTime + dTime + transferSec
			if totalSec < bestCost {
				bestCost = totalSec
				bestP = pNode
				bestD = dNode
				bestPrefillSec = pTime
				bestDecodeSec = dTime
				bestNetSec = transferSec
				bestTransferBytes = transBytes
			}
		}
	}

	var bestAttnNode, bestMLPNode NodeTelemetry
	for i, node := range allNodes {
		if i == 0 || node.BandwidthGBS > bestAttnNode.BandwidthGBS {
			bestAttnNode = node
		}
		if i == 0 || node.ComputeTFLOPs > bestMLPNode.ComputeTFLOPs {
			bestMLPNode = node
		}
	}

	attnNodeID := bestD.NodeID
	if bestD.BandwidthGBS < bestAttnNode.BandwidthGBS && bestAttnNode.BandwidthGBS > 0 {
		attnNodeID = bestAttnNode.NodeID
	}
	mlpNodeID := bestP.NodeID
	if bestP.ComputeTFLOPs < bestMLPNode.ComputeTFLOPs && bestMLPNode.ComputeTFLOPs > 0 {
		mlpNodeID = bestMLPNode.NodeID
	}

	attnLatMs := (attnDecodeBytes / math.Max(bestD.BandwidthGBS*1e9, 1e9)) * 1000.0
	mlpLatMs := (mlpPrefillFLOPs / math.Max(bestP.ComputeTFLOPs*1e12, 1e9)) * 1000.0

	var reason string
	if bestP.NodeID == bestD.NodeID {
		reason = fmt.Sprintf("Co-located on %s (%.0f TFLOPs, %.0f GB/s) avoiding network transfer overhead",
			bestP.OS, bestP.ComputeTFLOPs, bestP.BandwidthGBS)
	} else {
		link := n.linkBetween(bestP.NodeID, bestD.NodeID)
		reason = fmt.Sprintf("Disaggregated Prefill on %s (%.0f TFLOPs) and Decode on %s (%.0f GB/s) over link (RTT: %.2fms, BW: %.1f GB/s, Transfer: %.1f MB)",
			bestP.OS, bestP.ComputeTFLOPs, bestD.OS, bestD.BandwidthGBS,
			link.RTT.Seconds()*1000.0, link.BandwidthGBS, float64(bestTransferBytes)/(1024*1024))
	}

	return MeshExecutionPlan{
		PrefillNodeID:   bestP.NodeID,
		DecodeNodeID:    bestD.NodeID,
		AttentionNodeID: attnNodeID,
		MLPNodeID:       mlpNodeID,
		ComponentPlacements: []ComponentPlacement{
			{Component: ComponentPrefill, NodeID: bestP.NodeID, LatencyMs: bestPrefillSec * 1000.0},
			{Component: ComponentDecode, NodeID: bestD.NodeID, LatencyMs: bestDecodeSec * 1000.0},
			{Component: ComponentAttention, NodeID: attnNodeID, LatencyMs: attnLatMs},
			{Component: ComponentMLP, NodeID: mlpNodeID, LatencyMs: mlpLatMs},
		},
		EstimatedLatencyMs: bestCost * 1000.0,
		ComputeLatencyMs:   (bestPrefillSec + bestDecodeSec) * 1000.0,
		NetworkLatencyMs:   bestNetSec * 1000.0,
		TransferBytes:      bestTransferBytes,
		Reason:             reason,
	}
}

// InferRoles evaluates the distributed E-Graph cost function to choose optimal Prefill and Decode nodes.
func (n *IrohClusterNode) InferRoles(promptLen, genTokens int) DistributedRoleAssignment {
	plan := n.InferMeshPlan(promptLen, genTokens, 1)
	return DistributedRoleAssignment{
		PrefillNodeID:      plan.PrefillNodeID,
		DecodeNodeID:       plan.DecodeNodeID,
		EstimatedLatencyMs: plan.EstimatedLatencyMs,
		Reason:             plan.Reason,
	}
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

	plan := n.InferMeshPlan(2048, 64, 1)

	return ClusterMeshState{
		SelfNode:            n.self,
		Peers:               peersList,
		TotalNodes:          len(peersList) + 1,
		ClusterTFLOPs:       totalTFLOPs,
		ClusterBandwidthGBS: totalBW,
		ActiveAssignment: DistributedRoleAssignment{
			PrefillNodeID:      plan.PrefillNodeID,
			DecodeNodeID:       plan.DecodeNodeID,
			EstimatedLatencyMs: plan.EstimatedLatencyMs,
			Reason:             plan.Reason,
		},
		ActivePlan: plan,
	}
}
