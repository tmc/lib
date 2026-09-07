//go:build !tpu
package main

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"html"
	"net/http"
	"net/http/pprof"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/tmc/lib/apple/x/metal/egraph"
	"github.com/tmc/lib/apple/x/metal/metalprof"
	"github.com/tmc/lib/examples/internal/continuousbatch"
)

// embeddedDashboardJS is the dashboard script used when no development copy is available.
//
//go:embed dashboard.js
var embeddedDashboardJS []byte

func dashboardJS() []byte {
	paths := []string{"dashboard.js", "examples/llama-infer/dashboard.js"}
	if _, file, _, ok := runtime.Caller(0); ok {
		paths = append([]string{filepath.Join(filepath.Dir(file), "dashboard.js")}, paths...)
	}
	return loadDashboardJS(os.ReadFile, paths...)
}

func loadDashboardJS(readFile func(string) ([]byte, error), paths ...string) []byte {
	if len(paths) == 0 {
		paths = []string{"dashboard.js", "examples/llama-infer/dashboard.js"}
	}
	for _, name := range paths {
		if js, err := readFile(name); err == nil {
			return js
		}
	}
	return embeddedDashboardJS
}

// SyncJobStatus tracks real-time progress and bandwidth of an active model synchronization.
type SyncJobStatus struct {
	ID              string    `json:"id"`
	ModelID         string    `json:"model_id"`
	SourceNodeID    string    `json:"source_node_id"`
	SourcePlatform  string    `json:"source_platform"`
	CurrentFile     string    `json:"current_file"`
	DownloadedBytes int64     `json:"downloaded_bytes"`
	TotalBytes      int64     `json:"total_bytes"`
	Percent         float64   `json:"percent"`
	ThroughputMBs   float64   `json:"throughput_mbs"`
	State           string    `json:"state"` // "running", "completed", "error"
	Error           string    `json:"error,omitempty"`
	StartTime       time.Time `json:"start_time"`
	DurationMs      float64   `json:"duration_ms"`
}

// GenerationRecord captures latency, token throughput, and node routing attribution for a completion.
type GenerationRecord struct {
	ID              string    `json:"id"`
	Timestamp       time.Time `json:"timestamp"`
	Prompt          string    `json:"prompt"`
	Completion      string    `json:"completion"`
	TokensGenerated int       `json:"tokens_generated"`
	DurationMs      float64   `json:"duration_ms"`
	TokPerSec       float64   `json:"tok_per_sec"`
	PrefillNode     string    `json:"prefill_node"`
	PrefillPlatform string    `json:"prefill_platform"`
	DecodeNode      string    `json:"decode_node"`
	DecodePlatform  string    `json:"decode_platform"`
	Reason          string    `json:"reason"`
}

// PGOResponse is the telemetry payload served at /api/optimizer/pgo.
// Data is absent until an egraph.PGOTracker reports an observed tuning event.
type PGOResponse struct {
	State                    string                   `json:"state"`
	Reason                   string                   `json:"reason,omitempty"`
	FlightRecorderDispatches int                      `json:"flight_recorder_dispatches"`
	Data                     *egraph.PGOTelemetryData `json:"data,omitempty"`
}

// RequestSlot records one request's actual admission state in the inference
// scheduler. User is supplied by the OpenAI request and is not authenticated.
type RequestSlot struct {
	ID                string     `json:"id"`
	User              string     `json:"user,omitempty"`
	Kind              string     `json:"kind"`
	State             string     `json:"state"`
	Stream            bool       `json:"stream"`
	QueuePosition     int        `json:"queue_position,omitempty"`
	EnqueuedAt        time.Time  `json:"enqueued_at"`
	StartedAt         *time.Time `json:"started_at,omitempty"`
	FinishedAt        *time.Time `json:"finished_at,omitempty"`
	QueueDurationMS   *float64   `json:"queue_duration_ms,omitempty"`
	RuntimeDurationMS *float64   `json:"runtime_duration_ms,omitempty"`
}

// RequestQueueTelemetry is a snapshot of the service scheduler, not model
// batching telemetry. The queue exists to protect the shared engine buffers.
type RequestQueueTelemetry struct {
	State    string        `json:"state"`
	Capacity int           `json:"capacity"`
	Running  int           `json:"running"`
	Queued   int           `json:"queued"`
	Slots    []RequestSlot `json:"slots"`
	Recent   []RequestSlot `json:"recent"`
	Reason   string        `json:"reason"`
}

// egraphSaturationTelemetry reports compile-time planning evidence, not GPU work.
func egraphSaturationTelemetry(eng *ModelEngine) (*int, string) {
	if eng == nil || !eng.egraphKnown {
		return nil, "unmeasured"
	}
	count := eng.egraphSats
	return &count, "compile-time rewrite merges"
}

func formatEGraphSaturations(count *int) string {
	if count == nil {
		return "Rewrite merges unmeasured"
	}
	return fmt.Sprintf("%d compile-time rewrite merges", *count)
}

// RuntimeSchedulingTelemetry supplies live service scheduling state and the
// boundary of speculative runtime evidence.
type RuntimeSchedulingTelemetry struct {
	ServingMode  string                `json:"serving_mode"`
	ServingError string                `json:"serving_error,omitempty"`
	RequestQueue RequestQueueTelemetry `json:"request_queue"`
	Speculation  SpeculationTelemetry  `json:"speculation"`
}

// SpeculationTelemetry reports only comparisons completed by a paired draft
// and remote target engine. It does not describe OpenAI request acceleration.
type SpeculationTelemetry struct {
	State  string            `json:"state"`
	Reason string            `json:"reason"`
	Trace  *SpeculationTrace `json:"trace,omitempty"`
}

// RuntimeSpeculationRequest starts one explicit local-draft/remote-target
// comparison. The target is never selected implicitly.
type RuntimeSpeculationRequest struct {
	TargetNodeID string `json:"target_node_id"`
	Prompt       string `json:"prompt"`
	MaxTokens    int    `json:"max_tokens"`
}

// MemoryTelemetryResponse is the evidence-bounded memory capability payload
// served at /api/memory/kvcache.
type MemoryTelemetryResponse struct {
	KVCache     KVCacheTelemetry    `json:"kv_cache"`
	PrefixCache CapabilityTelemetry `json:"prefix_cache"`
	Precision   PrecisionTelemetry  `json:"precision"`
	Speculation CapabilityTelemetry `json:"speculation"`
	Batching    CapabilityTelemetry `json:"batching"`
}

// KVCacheTelemetry reports configured and observed KV-cache storage.
type KVCacheTelemetry struct {
	State             string `json:"state"`
	Reason            string `json:"reason,omitempty"`
	MaxSequenceTokens int    `json:"max_sequence_tokens,omitempty"`
	Layers            int    `json:"layers,omitempty"`
	KVHeads           int    `json:"kv_heads,omitempty"`
	HeadDim           int    `json:"head_dim,omitempty"`
	AllocatedBytes    int64  `json:"allocated_bytes,omitempty"`
	ActiveTokens      int    `json:"active_tokens,omitempty"`
}

// TensorPrecision is an observed safetensors metadata aggregate.
type TensorPrecision struct {
	DType       string `json:"dtype"`
	TensorCount int    `json:"tensor_count"`
	Bytes       uint64 `json:"bytes"`
}

// PrecisionTelemetry reports safetensors dtype metadata when the active engine
// has opened model weights.
type PrecisionTelemetry struct {
	State  string            `json:"state"`
	Reason string            `json:"reason,omitempty"`
	DTypes []TensorPrecision `json:"dtypes,omitempty"`
}

// CapabilityTelemetry makes an unsupported runtime measurement explicit.
type CapabilityTelemetry struct {
	State       string `json:"state"`
	Reason      string `json:"reason"`
	Entries     int    `json:"entries,omitempty"`
	Bytes       int64  `json:"bytes,omitempty"`
	Hits        int64  `json:"hits,omitempty"`
	Misses      int64  `json:"misses,omitempty"`
	SavedTokens int64  `json:"saved_tokens,omitempty"`
}

// RooflineTelemetry describes nominal hardware ceilings and workload points.
// Workload points remain unmeasured until runtime FLOP and byte counters exist.
type RooflineTelemetry struct {
	Ceilings []RooflineCeiling `json:"ceilings"`
	Points   []RooflinePoint   `json:"points"`
}

// RooflineCeiling is a node capability estimate, not observed throughput.
type RooflineCeiling struct {
	NodeID             string   `json:"node_id"`
	Platform           string   `json:"platform"`
	ComputeTFLOPs      *float64 `json:"compute_tflops,omitempty"`
	MemoryBandwidthGBS *float64 `json:"memory_bandwidth_gbs,omitempty"`
	Source             string   `json:"source"`
}

// RooflinePoint is one workload phase on the active assignment. The numeric
// fields are absent until the runtime measures both FLOPs and bytes moved.
type RooflinePoint struct {
	Phase               string   `json:"phase"`
	NodeID              string   `json:"node_id,omitempty"`
	State               string   `json:"state"`
	Reason              string   `json:"reason,omitempty"`
	FLOPs               *uint64  `json:"flops,omitempty"`
	BytesMoved          *uint64  `json:"bytes_moved,omitempty"`
	ArithmeticIntensity *float64 `json:"arithmetic_intensity,omitempty"`
	AttainedTFLOPs      *float64 `json:"attained_tflops,omitempty"`
}

// TelemetryResponse represents the JSON payload served at /api/telemetry.
type TelemetryResponse struct {
	ModelName         string                    `json:"model_name"`
	HiddenDim         int                       `json:"hidden_dim"`
	NumLayers         int                       `json:"num_layers"`
	EGraphSats        *int                      `json:"egraph_saturations"`
	EGraphSatsSource  string                    `json:"egraph_saturations_source"`
	ActivePlan        string                    `json:"active_plan"`
	ExecutionPlan     *ExecutionPlan            `json:"execution_plan,omitempty"`
	TotalDispatches   int                       `json:"total_dispatches"`
	MeanLatencyUs     float64                   `json:"mean_latency_us"`
	P50LatencyUs      float64                   `json:"p50_latency_us"`
	P95LatencyUs      float64                   `json:"p95_latency_us"`
	EffectiveTokS     float64                   `json:"effective_tok_s"`
	ClusterMesh       ClusterMeshState          `json:"cluster_mesh"`
	RecentGenerations []GenerationRecord        `json:"recent_generations"`
	ActiveSyncs       []*SyncJobStatus          `json:"active_syncs"`
	RecentEvents      []metalprof.KernelEvent   `json:"recent_events"`
	RecentTimeline    []metalprof.TimelineEvent `json:"recent_timeline"`
	Roofline          RooflineTelemetry         `json:"roofline"`
}

// TelemetryServer serves a live diagnostic dashboard for E-Graph and FlightRecorder telemetry.
type TelemetryServer struct {
	engine      *ModelEngine
	clusterNode *IrohClusterNode
	cfg         Config
	mux         *http.ServeMux

	historyMu sync.RWMutex
	history   []GenerationRecord

	syncJobsMu sync.RWMutex
	syncJobs   map[string]*SyncJobStatus

	pgoMu      sync.RWMutex
	pgoTracker *egraph.PGOTracker

	requestScheduler  *requestSlotScheduler
	continuous        *continuousbatch.Scheduler
	continuousEngine  servingBackend
	continuousInitErr error
}

// RecordGeneration appends a completed generation event with node routing attribution.
func (s *TelemetryServer) RecordGeneration(rec GenerationRecord) {
	s.historyMu.Lock()
	defer s.historyMu.Unlock()
	s.history = append(s.history, rec)
	if len(s.history) > 20 {
		s.history = s.history[len(s.history)-20:]
	}
}

// NewTelemetryServer initializes a new diagnostic HTTP server.
func NewTelemetryServer(engine *ModelEngine, cfg Config) *TelemetryServer {
	s := &TelemetryServer{
		engine:           engine,
		cfg:              cfg,
		mux:              http.NewServeMux(),
		syncJobs:         make(map[string]*SyncJobStatus),
		requestScheduler: newRequestSlotScheduler(1),
	}
	if engine != nil {
		backend, err := newPlatformServingBackend(engine, cfg, 8)
		if err != nil {
			s.continuousInitErr = fmt.Errorf("initialize continuous serving backend: %w", err)
		} else {
			scheduler, err := newServingScheduler(backend, continuousbatch.Config{MaxActive: 8, MaxQueue: 64, MaxBatch: 8, BatchWait: 10 * time.Millisecond, TokenBuffer: 16, PrefillWorkers: 4})
			if err != nil {
				s.continuousInitErr = fmt.Errorf("initialize continuous serving scheduler: %w", err)
				if backend != nil {
					backend.Close()
				}
			} else {
				s.continuousEngine = backend
				s.continuous = scheduler
			}
		}
	}
	s.routes()
	return s
}

// SetClusterNode attaches the active go-iroh P2P cluster node to the telemetry server.
func (s *TelemetryServer) SetClusterNode(node *IrohClusterNode) {
	s.clusterNode = node
}

// SetPGOTracker attaches a tracker that supplies observed PGO telemetry.
// Passing nil disables PGO telemetry.
func (s *TelemetryServer) SetPGOTracker(tracker *egraph.PGOTracker) {
	s.pgoMu.Lock()
	defer s.pgoMu.Unlock()
	s.pgoTracker = tracker
}

type requestSlotWaiter struct {
	slot    RequestSlot
	ready   chan struct{}
	running bool
}

type requestSlotScheduler struct {
	mu       sync.Mutex
	capacity int
	running  int
	waiting  []*requestSlotWaiter
	active   map[string]RequestSlot
	recent   []RequestSlot
}

func newRequestSlotScheduler(capacity int) *requestSlotScheduler {
	if capacity < 1 {
		capacity = 1
	}
	return &requestSlotScheduler{
		capacity: capacity,
		active:   make(map[string]RequestSlot),
	}
}

func (s *TelemetryServer) acquireRequestSlot(ctx context.Context, id, user, kind string, stream bool) (func(), error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if s.requestScheduler == nil {
		return func() {}, nil
	}

	waiter := &requestSlotWaiter{
		slot: RequestSlot{
			ID:         id,
			User:       user,
			Kind:       kind,
			State:      "queued",
			Stream:     stream,
			EnqueuedAt: time.Now(),
		},
		ready: make(chan struct{}),
	}
	scheduler := s.requestScheduler
	scheduler.mu.Lock()
	scheduler.waiting = append(scheduler.waiting, waiter)
	scheduler.dispatchLocked()
	scheduler.mu.Unlock()

	select {
	case <-waiter.ready:
		return s.releaseRequestSlot(waiter), nil
	case <-ctx.Done():
		scheduler.mu.Lock()
		if waiter.running {
			scheduler.mu.Unlock()
			return s.releaseRequestSlot(waiter), nil
		}
		scheduler.cancelLocked(waiter)
		scheduler.dispatchLocked()
		scheduler.mu.Unlock()
		return nil, ctx.Err()
	}
}

func (s *TelemetryServer) releaseRequestSlot(waiter *requestSlotWaiter) func() {
	var once sync.Once
	return func() {
		once.Do(func() {
			s.requestScheduler.release(waiter)
		})
	}
}

func (scheduler *requestSlotScheduler) dispatchLocked() {
	for scheduler.running < scheduler.capacity && len(scheduler.waiting) > 0 {
		waiter := scheduler.waiting[0]
		scheduler.waiting = scheduler.waiting[1:]
		now := time.Now()
		waiter.running = true
		waiter.slot.State = "running"
		waiter.slot.StartedAt = &now
		waiter.slot.QueueDurationMS = durationMilliseconds(now.Sub(waiter.slot.EnqueuedAt))
		scheduler.active[waiter.slot.ID] = waiter.slot
		scheduler.running++
		close(waiter.ready)
	}
}

func (scheduler *requestSlotScheduler) cancelLocked(waiter *requestSlotWaiter) {
	for i, candidate := range scheduler.waiting {
		if candidate != waiter {
			continue
		}
		scheduler.waiting = append(scheduler.waiting[:i], scheduler.waiting[i+1:]...)
		waiter.slot.State = "canceled"
		now := time.Now()
		waiter.slot.FinishedAt = &now
		waiter.slot.QueueDurationMS = durationMilliseconds(now.Sub(waiter.slot.EnqueuedAt))
		scheduler.appendRecentLocked(waiter.slot)
		return
	}
}

func (scheduler *requestSlotScheduler) release(waiter *requestSlotWaiter) {
	scheduler.mu.Lock()
	defer scheduler.mu.Unlock()
	slot, ok := scheduler.active[waiter.slot.ID]
	if !ok {
		return
	}
	delete(scheduler.active, waiter.slot.ID)
	scheduler.running--
	slot.State = "finished"
	now := time.Now()
	slot.FinishedAt = &now
	if slot.StartedAt != nil {
		slot.RuntimeDurationMS = durationMilliseconds(now.Sub(*slot.StartedAt))
	}
	scheduler.appendRecentLocked(slot)
	scheduler.dispatchLocked()
}

func (scheduler *requestSlotScheduler) appendRecentLocked(slot RequestSlot) {
	scheduler.recent = append(scheduler.recent, slot)
	if len(scheduler.recent) > 20 {
		scheduler.recent = scheduler.recent[len(scheduler.recent)-20:]
	}
}

func durationMilliseconds(d time.Duration) *float64 {
	milliseconds := float64(d) / float64(time.Millisecond)
	return &milliseconds
}

func (scheduler *requestSlotScheduler) snapshot() RequestQueueTelemetry {
	scheduler.mu.Lock()
	defer scheduler.mu.Unlock()

	slots := make([]RequestSlot, 0, len(scheduler.active)+len(scheduler.waiting))
	for _, slot := range scheduler.active {
		slots = append(slots, slot)
	}
	sort.Slice(slots, func(i, j int) bool {
		return slots[i].StartedAt.Before(*slots[j].StartedAt)
	})
	for i, waiter := range scheduler.waiting {
		slot := waiter.slot
		slot.QueuePosition = i + 1
		slots = append(slots, slot)
	}
	recent := append([]RequestSlot{}, scheduler.recent...)
	return RequestQueueTelemetry{
		State:    "observed",
		Capacity: scheduler.capacity,
		Running:  scheduler.running,
		Queued:   len(scheduler.waiting),
		Slots:    slots,
		Recent:   recent,
		Reason:   "one-slot FIFO admission protects the shared inference engine; user labels are client-provided and unauthenticated",
	}
}

func (s *TelemetryServer) routes() {
	s.mux.HandleFunc("/", s.handleDashboard)
	s.mux.HandleFunc("/dashboard.js", s.handleDashboardJS)
	s.mux.HandleFunc("/api/telemetry", s.handleAPITelemetry)
	s.mux.HandleFunc("/api/graph/plan", s.handleGraphPlan)
	s.mux.HandleFunc("/api/optimizer/pgo", s.handlePGO)
	s.mux.HandleFunc("/api/memory/kvcache", s.handleKVCache)
	s.mux.HandleFunc("/api/runtime/scheduling", s.handleRuntimeScheduling)
	s.mux.HandleFunc("/api/runtime/speculation", s.handleRuntimeSpeculation)
	s.mux.HandleFunc("/api/runtime/execution-nodes", s.handleExecutionNodes)
	s.mux.HandleFunc("/api/mesh/join", s.handleMeshJoin)
	s.mux.HandleFunc("/api/mesh/models", s.handleMeshModels)
	s.mux.HandleFunc("/api/mesh/sync", s.handleMeshSync)
	s.mux.HandleFunc("/api/mesh/sync/status", s.handleMeshSyncStatus)

	// OpenAI Compatible API Endpoints
	s.mux.HandleFunc("/v1/models", s.handleV1Models)
	s.mux.HandleFunc("/v1/chat/completions", s.handleV1ChatCompletions)
	s.mux.HandleFunc("/v1/completions", s.handleV1ChatCompletions)

	// Standard Go runtime pprof endpoints
	s.mux.HandleFunc("/debug/pprof/", pprof.Index)
	s.mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
	s.mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
	s.mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	s.mux.HandleFunc("/debug/pprof/trace", pprof.Trace)

	// GPU hardware FlightRecorder profile endpoint
	s.mux.HandleFunc("/debug/pprof/gpu", s.handlePprof)
}

// GraphPlan describes the small E-Graph expression used to select the active
// RMSNorm and GEMV execution plan.
type GraphPlan struct {
	Fused             bool          `json:"fused"`
	Saturations       *int          `json:"saturations"`
	SaturationsSource string        `json:"saturations_source"`
	Before            *PlanNode     `json:"before"`
	After             *PlanNode     `json:"after"`
	Rewrites          []PlanRewrite `json:"rewrites"`
}

// PlanNode is one expression in a graph plan.
type PlanNode struct {
	Op       string      `json:"op"`
	Target   string      `json:"target"`
	Children []*PlanNode `json:"children,omitempty"`
}

// PlanRewrite records whether a configured E-Graph rewrite matched.
type PlanRewrite struct {
	Name    string `json:"name"`
	Matched bool   `json:"matched"`
}

func (s *TelemetryServer) handleGraphPlan(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	plan, err := s.graphPlan()
	if err != nil {
		http.Error(w, fmt.Sprintf("build graph plan: %v", err), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(plan)
}

func (s *TelemetryServer) graphPlan() (GraphPlan, error) {
	eg := egraph.New()
	x := eg.Add(egraph.NewLeaf("x"))
	wNorm := eg.Add(egraph.NewLeaf("w_norm"))
	wq := eg.Add(egraph.NewLeaf("wq"))
	norm := eg.Add(egraph.NewNode("rmsnorm", x, wNorm))
	root := eg.Add(egraph.NewNode("gemv", norm, wq))

	before, _, err := egraph.ExtractBest(eg, root, egraph.AstSizeCost{})
	if err != nil {
		return GraphPlan{}, fmt.Errorf("extract unfused plan: %w", err)
	}
	rules := []*egraph.Rewrite{
		egraph.MustRewrite("fuse_rmsnorm_gemv", "(gemv (rmsnorm ?x ?w) ?wq)", "(fused_rmsnorm_gemv ?x ?w ?wq)"),
		egraph.MustRewrite("fuse_swiglu_gemv", "(gemv (* (silu (gemv ?x ?wg)) (gemv ?x ?wu)) ?wd)", "(fused_swiglu_gemv ?x ?wg ?wu ?wd)"),
	}
	matched := make([]bool, len(rules))
	if s.cfg.FuseEGraph {
		for i, rule := range rules {
			matched[i] = rule.Apply(eg) > 0
		}
		eg.Rebuild()
	}
	after, _, err := egraph.ExtractBest(eg, root, egraph.FusedMegakernelCost{})
	if err != nil {
		return GraphPlan{}, fmt.Errorf("extract fused plan: %w", err)
	}

	saturations, source := egraphSaturationTelemetry(s.engine)
	return GraphPlan{
		Fused:             s.engine.fusionActive(),
		Saturations:       saturations,
		SaturationsSource: source,
		Before:            planNode(before, nil),
		After:             planNode(after, egraph.HeterogeneousPartition(after)),
		Rewrites: []PlanRewrite{
			{Name: rules[0].Name, Matched: matched[0]},
			{Name: rules[1].Name, Matched: matched[1]},
		},
	}, nil
}

func planNode(expr *egraph.Expr, partition *egraph.PartitionedGraph) *PlanNode {
	if expr == nil {
		return nil
	}
	target := "host_cpu"
	if partition != nil {
		for _, candidate := range partition.ANEExprs {
			if candidate == expr {
				target = "apple_ane"
			}
		}
		for _, candidate := range partition.GPUExprs {
			if candidate == expr {
				target = "metal_gpu"
			}
		}
	}
	node := &PlanNode{Op: expr.Op, Target: target}
	for _, child := range expr.Children {
		node.Children = append(node.Children, planNode(child, partition))
	}
	return node
}

// ClusterModelInfo represents a model aggregated across the mesh.
type ClusterModelInfo struct {
	ModelID    string        `json:"model_id"`
	TotalBytes int64         `json:"total_bytes"`
	FileCount  int           `json:"file_count"`
	Nodes      []NodeSummary `json:"nodes"`
}

// NodeSummary summarizes a node hosting a model snapshot.
type NodeSummary struct {
	NodeID    string  `json:"node_id"`
	OS        string  `json:"os"`
	Arch      string  `json:"arch"`
	TFLOPs    float64 `json:"tflops"`
	Bandwidth float64 `json:"bandwidth"`
}

func (s *TelemetryServer) handleMeshModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	modelsMap := make(map[string]*ClusterModelInfo)
	if s.clusterNode != nil {
		allNodes := append([]NodeTelemetry{s.clusterNode.SelfTelemetry()}, s.clusterNode.MeshState().Peers...)
		for _, n := range allNodes {
			for _, m := range n.AvailableModels {
				info, ok := modelsMap[m.ModelID]
				if !ok {
					info = &ClusterModelInfo{
						ModelID:    m.ModelID,
						TotalBytes: m.TotalBytes,
						FileCount:  m.FileCount,
					}
					modelsMap[m.ModelID] = info
				}
				info.Nodes = append(info.Nodes, NodeSummary{
					NodeID:    n.NodeID,
					OS:        n.OS,
					Arch:      n.Arch,
					TFLOPs:    n.ComputeTFLOPs,
					Bandwidth: n.BandwidthGBS,
				})
			}
		}
	}

	var res []ClusterModelInfo
	for _, v := range modelsMap {
		res = append(res, *v)
	}
	sort.Slice(res, func(i, j int) bool {
		return res[i].ModelID < res[j].ModelID
	})
	_ = json.NewEncoder(w).Encode(res)
}

// SyncRequest specifies a model and peer to sync from.
type SyncRequest struct {
	ModelID      string `json:"model_id"`
	SourceNodeID string `json:"source_node_id"`
}

func (s *TelemetryServer) handleMeshSync(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req SyncRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if req.ModelID == "" {
		http.Error(w, "model_id required", http.StatusBadRequest)
		return
	}
	if s.clusterNode == nil {
		http.Error(w, "cluster node not running", http.StatusServiceUnavailable)
		return
	}
	if req.SourceNodeID == "" {
		http.Error(w, "source_node_id required", http.StatusBadRequest)
		return
	}
	if _, err := s.clusterNode.PeerAddr(req.SourceNodeID); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	jobID := fmt.Sprintf("sync-%d", time.Now().UnixNano())
	job := &SyncJobStatus{
		ID:           jobID,
		ModelID:      req.ModelID,
		SourceNodeID: req.SourceNodeID,
		State:        "running",
		StartTime:    time.Now(),
	}

	for _, p := range s.clusterNode.MeshState().Peers {
		if p.NodeID == req.SourceNodeID {
			job.SourcePlatform = fmt.Sprintf("%s/%s", p.OS, p.Arch)
			break
		}
	}

	s.syncJobsMu.Lock()
	s.syncJobs[jobID] = job
	s.syncJobsMu.Unlock()

	go func() {
		stats, err := s.clusterNode.SyncFromPeer(context.Background(), req.SourceNodeID, req.ModelID, func(file string, dl, total int64, speed float64) {
			s.syncJobsMu.Lock()
			job.CurrentFile = file
			job.DownloadedBytes = dl
			job.TotalBytes = total
			if total > 0 {
				job.Percent = (float64(dl) / float64(total)) * 100.0
			}
			job.ThroughputMBs = speed
			s.syncJobsMu.Unlock()
		})

		s.syncJobsMu.Lock()
		defer s.syncJobsMu.Unlock()
		job.DurationMs = float64(time.Since(job.StartTime).Milliseconds())
		if err != nil {
			job.State = "error"
			job.Error = err.Error()
		} else {
			job.State = "completed"
			job.Percent = 100.0
			if stats != nil {
				job.ThroughputMBs = stats.ThroughputMBs
			}
		}
	}()

	_ = json.NewEncoder(w).Encode(job)
}

func (s *TelemetryServer) handleMeshSyncStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	_ = json.NewEncoder(w).Encode(s.syncJobSnapshots())
}

func (s *TelemetryServer) handleMeshJoin(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var peer NodeTelemetry
	if err := json.NewDecoder(r.Body).Decode(&peer); err != nil {
		http.Error(w, fmt.Sprintf("invalid JSON payload: %v", err), http.StatusBadRequest)
		return
	}

	if s.clusterNode != nil {
		if err := s.clusterNode.RegisterPeer(peer); err != nil {
			http.Error(w, fmt.Sprintf("save peer: %v", err), http.StatusInternalServerError)
			return
		}
	}

	w.Header().Set("Content-Type", "application/json")
	if s.clusterNode != nil {
		_ = json.NewEncoder(w).Encode(s.clusterNode.SelfTelemetry())
	} else {
		_ = json.NewEncoder(w).Encode(s.getStats().ClusterMesh.SelfNode)
	}
}

// ServeHTTP implements http.Handler.
func (s *TelemetryServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
}

func (s *TelemetryServer) getStats() TelemetryResponse {
	var events []metalprof.KernelEvent
	var timeline []metalprof.TimelineEvent
	if s.engine != nil && s.engine.recorder != nil {
		events = s.engine.recorder.Events()
		timeline = s.engine.recorder.Timeline()
	}

	var totalDur time.Duration
	var latencies []float64
	for _, ev := range events {
		totalDur += ev.Duration
		latencies = append(latencies, float64(ev.Duration.Microseconds()))
	}

	meanUs := 0.0
	p50Us := 0.0
	p95Us := 0.0
	tokS := 0.0

	if len(latencies) > 0 {
		sort.Float64s(latencies)
		meanUs = float64(totalDur.Microseconds()) / float64(len(latencies))
		p50Us = latencies[len(latencies)*50/100]
		p95Us = latencies[len(latencies)*95/100]
		if meanUs > 0 {
			tokS = 1000000.0 / meanUs
		}
	}

	activePlan := "Fused Megakernel (RMSNorm + GEMV + SwiGLU)"
	if !s.engine.fusionActive() {
		activePlan = "Discrete Multi-Pass Pipeline"
	}

	// Truncate to most recent 50 events for UI view
	recent := events
	if len(recent) > 50 {
		recent = recent[len(recent)-50:]
	}
	recentTimeline := timeline
	if len(recentTimeline) > 50 {
		recentTimeline = recentTimeline[len(recentTimeline)-50:]
	}

	mesh := ClusterMeshState{}
	if s.clusterNode != nil {
		mesh = s.clusterNode.MeshState()
	}
	if mesh.TotalNodes == 0 {
		mesh = ClusterMeshState{
			TotalNodes: 1,
			SelfNode: NodeTelemetry{
				NodeID:        "local-node",
				OS:            s.cfg.Name,
				Arch:          "gpu",
				ComputeTFLOPs: 40.0,
				BandwidthGBS:  800.0,
			},
			ClusterTFLOPs:       40.0,
			ClusterBandwidthGBS: 800.0,
			ActiveAssignment: DistributedRoleAssignment{
				PrefillNodeID: "local-node",
				DecodeNodeID:  "local-node",
				Reason:        "Standalone local execution",
			},
		}
	}
	roofline := s.rooflineTelemetry(mesh)

	saturations, source := egraphSaturationTelemetry(s.engine)
	return TelemetryResponse{
		ModelName:         s.cfg.Name,
		HiddenDim:         s.cfg.Dim,
		NumLayers:         s.cfg.NumLayers,
		EGraphSats:        saturations,
		EGraphSatsSource:  source,
		ActivePlan:        activePlan,
		ExecutionPlan:     s.engine.Plan,
		TotalDispatches:   len(events),
		MeanLatencyUs:     meanUs,
		P50LatencyUs:      p50Us,
		P95LatencyUs:      p95Us,
		EffectiveTokS:     tokS,
		ClusterMesh:       mesh,
		RecentGenerations: s.getRecentGenerations(),
		ActiveSyncs:       s.getActiveSyncs(),
		RecentEvents:      recent,
		RecentTimeline:    recentTimeline,
		Roofline:          roofline,
	}
}

func (s *TelemetryServer) rooflineTelemetry(mesh ClusterMeshState) RooflineTelemetry {
	reason := "runtime FLOP and byte counters are unavailable"
	if s.engine != nil && s.engine.metrics != nil {
		forwards, _, _, _, _ := s.engine.metrics.Snapshot()
		for _, measurement := range forwards {
			if measurement.GPUTime > 0 {
				reason = "GPU timing is measured, but runtime FLOP and byte counters are unavailable"
				break
			}
		}
	}
	nodes := append([]NodeTelemetry{mesh.SelfNode}, mesh.Peers...)
	ceilings := make([]RooflineCeiling, 0, len(nodes))
	for _, node := range nodes {
		ceilings = append(ceilings, RooflineCeiling{
			NodeID:             node.NodeID,
			Platform:           node.OS + "/" + node.Arch,
			ComputeTFLOPs:      positiveFloat(node.ComputeTFLOPs),
			MemoryBandwidthGBS: positiveFloat(node.BandwidthGBS),
			Source:             "heuristic capability estimate",
		})
	}
	sort.Slice(ceilings, func(i, j int) bool {
		return ceilings[i].NodeID < ceilings[j].NodeID
	})
	return RooflineTelemetry{
		Ceilings: ceilings,
		Points: []RooflinePoint{
			{
				Phase:  "prefill",
				NodeID: mesh.ActiveAssignment.PrefillNodeID,
				State:  "unmeasured",
				Reason: reason,
			},
			{
				Phase:  "decode",
				NodeID: mesh.ActiveAssignment.DecodeNodeID,
				State:  "unmeasured",
				Reason: reason,
			},
		},
	}
}

func positiveFloat(v float64) *float64 {
	if v <= 0 {
		return nil
	}
	return &v
}

func (s *TelemetryServer) getActiveSyncs() []*SyncJobStatus {
	return s.syncJobSnapshots()
}

func (s *TelemetryServer) syncJobSnapshots() []*SyncJobStatus {
	s.syncJobsMu.RLock()
	defer s.syncJobsMu.RUnlock()
	list := make([]*SyncJobStatus, 0, len(s.syncJobs))
	for _, j := range s.syncJobs {
		copy := *j
		list = append(list, &copy)
	}
	sort.Slice(list, func(i, j int) bool { return list[i].StartTime.Before(list[j].StartTime) })
	return list
}

func (s *TelemetryServer) getRecentGenerations() []GenerationRecord {
	s.historyMu.RLock()
	defer s.historyMu.RUnlock()
	res := make([]GenerationRecord, len(s.history))
	copy(res, s.history)
	return res
}

func (s *TelemetryServer) handleAPITelemetry(w http.ResponseWriter, r *http.Request) {
	stats := s.getStats()
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	_ = json.NewEncoder(w).Encode(stats)
}

func (s *TelemetryServer) handlePGO(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	_ = json.NewEncoder(w).Encode(s.pgoTelemetry())
}

func (s *TelemetryServer) handleKVCache(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	_ = json.NewEncoder(w).Encode(s.memoryTelemetry())
}

func (s *TelemetryServer) handleRuntimeScheduling(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	_ = json.NewEncoder(w).Encode(s.runtimeSchedulingTelemetry())
}

func (s *TelemetryServer) handleRuntimeSpeculation(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var request RuntimeSpeculationRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "invalid JSON payload", http.StatusBadRequest)
		return
	}
	request.TargetNodeID = strings.TrimSpace(request.TargetNodeID)
	if request.TargetNodeID == "" {
		http.Error(w, "target_node_id required", http.StatusBadRequest)
		return
	}
	if request.MaxTokens <= 0 {
		http.Error(w, "max_tokens must be positive", http.StatusBadRequest)
		return
	}
	if s.clusterNode == nil {
		http.Error(w, "paired draft/target execution unavailable", http.StatusServiceUnavailable)
		return
	}
	if request.TargetNodeID == s.clusterNode.NodeID() {
		http.Error(w, "target_node_id must identify a remote node", http.StatusBadRequest)
		return
	}
	releaseSlot, err := s.acquireRequestSlot(r.Context(), fmt.Sprintf("speculation-%d", time.Now().UnixNano()), "", "paired_speculation", false)
	if err != nil {
		http.Error(w, "request canceled while waiting for inference slot", http.StatusRequestTimeout)
		return
	}
	defer releaseSlot()
	trace, err := s.clusterNode.Speculate(r.Context(), request.TargetNodeID, request.Prompt, request.MaxTokens)
	if err != nil {
		http.Error(w, fmt.Sprintf("paired speculation: %v", err), http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(trace)
}

func (s *TelemetryServer) runtimeSchedulingTelemetry() RuntimeSchedulingTelemetry {
	queue := RequestQueueTelemetry{
		State:    "unavailable",
		Reason:   "request scheduler is unavailable",
		Capacity: 0,
	}
	if s.requestScheduler != nil {
		queue = s.requestScheduler.snapshot()
	}
	speculation := SpeculationTelemetry{
		State:  "unavailable",
		Reason: "no active paired draft and target execution reports proposal or acceptance events",
	}
	if s.engine != nil && s.engine.metrics != nil {
		traces := s.engine.metrics.Speculation()
		if len(traces) > 0 {
			latest := traces[len(traces)-1]
			speculation.State = "observed"
			speculation.Reason = "latest paired draft and remote target comparison; OpenAI generation is not accelerated by this trace"
			speculation.Trace = &latest
		}
	}
	mode := "serial"
	if s.continuous != nil {
		mode = "continuous-batch"
	} else if s.engine == nil {
		mode = "unavailable"
	}
	var initError string
	if s.continuousInitErr != nil {
		initError = s.continuousInitErr.Error()
	}
	return RuntimeSchedulingTelemetry{RequestQueue: queue, Speculation: speculation, ServingMode: mode, ServingError: initError}
}

func (s *TelemetryServer) memoryTelemetry() MemoryTelemetryResponse {
	response := MemoryTelemetryResponse{
		KVCache: KVCacheTelemetry{
			State:             "configured",
			Reason:            "configuration only; runtime allocation, occupancy, fragmentation, and block counters are unavailable",
			MaxSequenceTokens: s.cfg.MaxSeqLen,
			Layers:            s.cfg.NumLayers,
			KVHeads:           s.cfg.NumKVHeads,
			HeadDim:           s.cfg.HeadDim,
		},
		PrefixCache: CapabilityTelemetry{
			State:  "unavailable",
			Reason: "active runtime does not expose prefix-cache hit or block telemetry",
		},
		Precision: PrecisionTelemetry{
			State:  "unavailable",
			Reason: "active engine has no observed safetensors dtype metadata",
		},
		Speculation: CapabilityTelemetry{
			State:  "unavailable",
			Reason: "active runtime does not expose draft-tree or acceptance telemetry",
		},
		Batching: CapabilityTelemetry{
			State:  "unavailable",
			Reason: "active runtime does not expose batch-slot telemetry",
		},
	}
	if s.cfg.MaxSeqLen <= 0 {
		response.KVCache.State = "unavailable"
		response.KVCache.Reason = "KV-cache capacity is not configured"
	}
	if s.engine == nil {
		response.Precision.Reason = "no active engine is attached"
		return response
	}
	if s.engine.metrics != nil {
		_, bytes, allocated, tokens, maxTokens := s.engine.metrics.Snapshot()
		if maxTokens > 0 {
			response.KVCache.MaxSequenceTokens = maxTokens
		}
		if allocated {
			response.KVCache.State = "observed"
			response.KVCache.Reason = "physical KV buffers allocated by the active engine; fragmentation and block counters are unavailable"
			response.KVCache.AllocatedBytes = bytes
			response.KVCache.ActiveTokens = tokens
		}
		if traces := s.engine.metrics.Speculation(); len(traces) > 0 {
			response.Speculation.State = "observed"
			response.Speculation.Reason = "latest paired draft and remote target comparison is available from runtime scheduling telemetry"
		}
	}
	prefix := s.engine.prefixCacheMetrics()
	prefixAvailable := s.engine.prefixCacheAvailable()
	if source, ok := s.continuousEngine.(servingPrefixCache); ok {
		if servingPrefix, available := source.prefixCacheMetrics(); available {
			prefix = servingPrefix
			prefixAvailable = true
		}
	}
	if prefixAvailable {
		response.PrefixCache = CapabilityTelemetry{
			State:       "observed",
			Reason:      "bounded snapshots are captured from and restored into active KV buffers",
			Entries:     prefix.Entries,
			Bytes:       prefix.Bytes,
			Hits:        prefix.Hits,
			Misses:      prefix.Misses,
			SavedTokens: prefix.SavedTokens,
		}
	}
	if dtypes := s.engine.precisionTelemetry(); len(dtypes) > 0 {
		response.Precision.State = "observed"
		response.Precision.Reason = "observed from active safetensors metadata"
		response.Precision.DTypes = dtypes
	}
	return response
}

func (s *TelemetryServer) pgoTelemetry() PGOResponse {
	response := PGOResponse{
		State:                    "inactive",
		FlightRecorderDispatches: s.flightRecorderDispatches(),
	}
	s.pgoMu.RLock()
	tracker := s.pgoTracker
	s.pgoMu.RUnlock()
	if tracker == nil {
		response.Reason = "no PGO tracker is attached"
		if response.FlightRecorderDispatches > 0 {
			response.Reason = "FlightRecorder dispatches have no PGO tuning attribution"
		}
		return response
	}

	data := tracker.Telemetry()
	if len(data.History) == 0 && len(data.AutoTuningTrials) == 0 && len(data.HotSwaps) == 0 && !data.Convergence.Locked {
		response.Reason = "PGO tracker has no tuning trials, hot swaps, or convergence evidence"
		return response
	}
	response.State = "observed"
	response.Data = &data
	return response
}

func (s *TelemetryServer) flightRecorderDispatches() int {
	if s.engine == nil || s.engine.recorder == nil {
		return 0
	}
	return len(s.engine.recorder.Events())
}

func (s *TelemetryServer) handlePprof(w http.ResponseWriter, r *http.Request) {
	if s.engine == nil || s.engine.recorder == nil {
		http.Error(w, "flight recorder unavailable", http.StatusServiceUnavailable)
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Disposition", "attachment; filename=\"gpu_profile.pb.gz\"")
	if err := s.engine.recorder.SnapshotPprof(w); err != nil {
		http.Error(w, fmt.Sprintf("failed writing pprof: %v", err), http.StatusInternalServerError)
	}
}

func formatBytes(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(b)/float64(div), "KMGTPE"[exp])
}

func (s *TelemetryServer) handleDashboard(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	stats := s.getStats()

	var meshRows strings.Builder
	allNodes := append([]NodeTelemetry{stats.ClusterMesh.SelfNode}, stats.ClusterMesh.Peers...)
	for _, n := range allNodes {
		roleBadge := `<span class="badge" style="background:#21262d; color:#8b949e;">STANDALONE WORKER</span>`
		if n.NodeID == stats.ClusterMesh.ActiveAssignment.PrefillNodeID && n.NodeID == stats.ClusterMesh.ActiveAssignment.DecodeNodeID {
			roleBadge = `<span class="badge" style="background:#238636; color:white;">PREFILL + DECODE</span>`
		} else if n.NodeID == stats.ClusterMesh.ActiveAssignment.PrefillNodeID {
			roleBadge = `<span class="badge" style="background:#1f6feb; color:white;">PREFILL (COMPUTE)</span>`
		} else if n.NodeID == stats.ClusterMesh.ActiveAssignment.DecodeNodeID {
			roleBadge = `<span class="badge" style="background:#8957e5; color:white;">DECODE (BANDWIDTH)</span>`
		}

		nodeModelsCount := len(n.AvailableModels)
		meshRows.WriteString(fmt.Sprintf(`
                    <tr style="cursor:pointer;" role="button" tabindex="0" onclick="openNodeModal('%s')" onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();openNodeModal('%s')}">
                        <td style="color:var(--accent); font-weight:600; font-family:monospace;">%s</td>
                        <td><span class="badge" style="background:#1f6feb;">%s</span></td>
                        <td style="color:#8b949e;">%s</td>
                        <td style="color:#ffffff; font-weight:600;">%.1f TFLOPs</td>
                        <td style="color:#ffffff; font-weight:600;">%.1f GB/s</td>
                        <td>%s</td>
                        <td><span class="badge" style="background:#30363d; color:white;">📦 %d Models</span> &nbsp; <span style="font-size:11px; color:var(--accent);">🔍 View</span></td>
                    </tr>`,
			n.NodeID,
			n.NodeID,
			n.NodeID,
			strings.ToUpper(n.OS),
			n.Arch,
			n.ComputeTFLOPs,
			n.BandwidthGBS,
			roleBadge,
			nodeModelsCount,
		))
	}

	var modelCatalogRows strings.Builder
	modelsMap := make(map[string]*ClusterModelInfo)
	for _, n := range allNodes {
		for _, m := range n.AvailableModels {
			info, ok := modelsMap[m.ModelID]
			if !ok {
				info = &ClusterModelInfo{
					ModelID:    m.ModelID,
					TotalBytes: m.TotalBytes,
					FileCount:  m.FileCount,
				}
				modelsMap[m.ModelID] = info
			}
			info.Nodes = append(info.Nodes, NodeSummary{
				NodeID:    n.NodeID,
				OS:        n.OS,
				Arch:      n.Arch,
				TFLOPs:    n.ComputeTFLOPs,
				Bandwidth: n.BandwidthGBS,
			})
		}
	}
	for _, m := range modelsMap {
		var nodeBadges strings.Builder
		for _, n := range m.Nodes {
			pShort := n.NodeID
			if len(pShort) > 8 {
				pShort = pShort[:8]
			}
			nodeBadges.WriteString(fmt.Sprintf(`<span class="badge" style="background:#21262d; border:1px solid #30363d; margin-right:6px;"><b style="color:var(--accent);">%s</b> <code>%s</code></span>`, strings.ToUpper(n.OS), pShort))
		}
		modelCatalogRows.WriteString(fmt.Sprintf(`
                    <tr>
                        <td style="color:var(--accent-purple); font-weight:600; font-family:monospace;">%s</td>
                        <td style="color:#ffffff; font-weight:600;">%s</td>
                        <td style="color:#8b949e;">%d files</td>
                        <td>%s</td>
                    </tr>`,
			m.ModelID,
			formatBytes(m.TotalBytes),
			m.FileCount,
			nodeBadges.String(),
		))
	}
	if modelCatalogRows.Len() == 0 {
		modelCatalogRows.WriteString(`<tr><td colspan="4" style="text-align:center; color:#8b949e; padding:18px;">No Hugging Face models discovered in local cache (~/.cache/huggingface/hub).</td></tr>`)
	}

	var genRows strings.Builder
	for i := len(stats.RecentGenerations) - 1; i >= 0; i-- {
		g := stats.RecentGenerations[i]
		prefillShort := g.PrefillNode
		if len(prefillShort) > 8 {
			prefillShort = prefillShort[:8]
		}
		decodeShort := g.DecodeNode
		if len(decodeShort) > 8 {
			decodeShort = decodeShort[:8]
		}
		genRows.WriteString(fmt.Sprintf(`
                    <tr>
                        <td style="color:#8b949e;">%s</td>
                        <td style="color:var(--text-bright); max-width:220px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="%s">%s</td>
                        <td><span class="badge" style="background:#1f6feb;">%s</span> <code>%s</code></td>
                        <td><span class="badge" style="background:#8957e5;">%s</span> <code>%s</code></td>
                        <td style="color:#3fb950; font-weight:600;">%d tok (%.1f tok/s)</td>
                        <td style="color:#8b949e;">%.1f ms</td>
                    </tr>`,
			g.Timestamp.Format("15:04:05"),
			html.EscapeString(g.Prompt),
			html.EscapeString(g.Prompt),
			g.PrefillPlatform, prefillShort,
			g.DecodePlatform, decodeShort,
			g.TokensGenerated, g.TokPerSec,
			g.DurationMs,
		))
	}
	if genRows.Len() == 0 {
		genRows.WriteString(`<tr><td colspan="6" style="text-align:center; color:#8b949e; padding:18px;">No generations recorded yet. Use the prompt box above to run inference.</td></tr>`)
	}

	var eventRows strings.Builder
	for i := len(stats.RecentEvents) - 1; i >= 0; i-- {
		ev := stats.RecentEvents[i]
		lbls := make([]string, 0, len(ev.Labels))
		for k, v := range ev.Labels {
			lbls = append(lbls, fmt.Sprintf("%s=%s", k, v))
		}
		eventRows.WriteString(fmt.Sprintf(`
                    <tr>
                        <td style="color:#8b949e;">%d</td>
                        <td style="color:var(--accent); font-weight:600;">%s</td>
                        <td>%.2f</td>
                        <td>[%d, %d, %d]</td>
                        <td>[%d, %d, %d]</td>
                        <td style="color:#8b949e;">%s</td>
                    </tr>`,
			ev.TimestampNs,
			ev.Name,
			float64(ev.Duration.Microseconds()),
			ev.GridSize[0], ev.GridSize[1], ev.GridSize[2],
			ev.TGSize[0], ev.TGSize[1], ev.TGSize[2],
			strings.Join(lbls, ", "),
		))
	}

	htmlOut := fmt.Sprintf(htmlTemplate,
		stats.ActivePlan, formatEGraphSaturations(stats.EGraphSats),
		stats.TotalDispatches,
		stats.P50LatencyUs,
		stats.P95LatencyUs,
		stats.EffectiveTokS,
		modelCatalogRows.String(),
		stats.ClusterMesh.TotalNodes, stats.ClusterMesh.ClusterTFLOPs, stats.ClusterMesh.ClusterBandwidthGBS,
		stats.ClusterMesh.ActiveAssignment.Reason,
		meshRows.String(),
		genRows.String(),
		eventRows.String(),
	)

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write([]byte(htmlOut))
}

func (s *TelemetryServer) handleDashboardJS(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/dashboard.js" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("Content-Type", "text/javascript; charset=utf-8")
	_, _ = w.Write(dashboardJS())
}

const htmlTemplate = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>tmc/lib Adaptive E-Graph Telemetry &amp; Serving Hub</title>
    <style>
        :root {
            --bg: #080c0d;
            --bg-subtle: #090d0e;
            --card-bg: #0d1314;
            --card-hover: #131b19;
            --border: #1d2828;
            --border-hover: #26302f;
            --text: #a9b2af;
            --text-bright: #ebefee;
            --accent: #c4f000;
            --accent-green: #3fb950;
            --accent-purple: #835dff;
            --accent-blue: #58a6ff;
            --accent-orange: #ff8e32;
        }
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 0;
            min-height: 100vh;
        }
        .topbar {
            height: 47px;
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 0 24px;
            border-bottom: 1px solid var(--border);
            background: #080c0d;
            position: sticky;
            top: 0;
            z-index: 20;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #ebefee;
            font-weight: 700;
            letter-spacing: -.03em;
            font-size: 15px;
            text-decoration: none;
            background: transparent;
            border: 0;
            cursor: pointer;
        }
        .brand-mark {
            background: var(--accent);
            color: #07100c;
            width: 27px;
            height: 17px;
            border-radius: 8px;
            display: grid;
            place-items: center;
            font-size: 14px;
            font-weight: 900;
        }
        .global-search {
            height: 28px;
            width: 240px;
            margin-left: 24px;
            display: flex;
            align-items: center;
            gap: 7px;
            padding: 0 10px;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: #101616;
            color: #7f8b89;
            font-size: 12px;
        }
        .global-search input {
            background: transparent;
            border: 0;
            outline: 0;
            color: #ebefee;
            font-size: 12px;
            width: 100%%;
        }
        .global-search kbd {
            margin-left: auto;
            color: #87908e;
            font-size: 10px;
        }
        .topbar nav {
            display: flex;
            align-items: stretch;
            height: 47px;
            margin-left: 16px;
        }
        .nav-link {
            padding: 0 14px;
            font-size: 13px;
            color: #889290;
            background: transparent;
            border: 0;
            cursor: pointer;
            display: flex;
            align-items: center;
            border-bottom: 2px solid transparent;
            transition: color 0.15s ease;
        }
        .nav-link:hover { color: #f0f2f1; }
        .nav-link.active {
            color: var(--accent);
            border-bottom-color: var(--accent);
            font-weight: 600;
        }
        .top-actions {
            margin-left: auto;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .btn-top {
            padding: 5px 10px;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: #0d1213;
            color: #aeb6b4;
            text-decoration: none;
            font-size: 12px;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            cursor: pointer;
        }
        .btn-top:hover {
            border-color: #567000;
            color: var(--accent);
            background: #151c11;
        }
        .btn-perfetto {
            background: #151c11;
            border-color: #567000;
            color: var(--accent);
            font-weight: 600;
        }

        .main-content {
            max-width: 1240px;
            margin: 0 auto;
            padding: 24px 28px 80px;
        }

        /* Metrics Strip */
        .metric-strip {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-bottom: 24px;
        }
        .metric-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px 16px;
        }
        .metric-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: .06em;
            color: #77817e;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 22px;
            font-weight: 700;
            color: var(--text-bright);
        }
        .metric-sub {
            font-size: 11px;
            color: #77817e;
            margin-top: 4px;
        }

        /* Section Container */
        .panel {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 18px 20px;
            margin-bottom: 20px;
        }
        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 14px;
        }
        .panel-title {
            font-size: 15px;
            font-weight: 600;
            color: var(--text-bright);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Modality & View Switches */
        .toolbar {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 14px;
            flex-wrap: wrap;
        }
        .modality-tabs {
            display: flex;
            align-items: center;
            gap: 16px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
            margin-bottom: 14px;
        }
        .modality-tabs button {
            background: transparent;
            border: 0;
            color: #77817e;
            font-size: 13px;
            padding: 4px 2px;
            cursor: pointer;
            position: relative;
        }
        .modality-tabs button.active {
            color: var(--accent);
            font-weight: 600;
        }
        .modality-tabs button.active::after {
            content: '';
            position: absolute;
            bottom: -9px;
            left: 0;
            right: 0;
            height: 2px;
            background: var(--accent);
        }
        .view-switch {
            display: flex;
            margin-left: auto;
            border: 1px solid var(--border);
            border-radius: 6px;
            overflow: hidden;
        }
        .view-switch button {
            background: #111718;
            border: 0;
            color: #77817e;
            padding: 5px 10px;
            font-size: 12px;
            cursor: pointer;
        }
        .view-switch button.active {
            background: #151d15;
            color: var(--accent);
        }

        /* Models Table & Cards */
        .model-table {
            border: 1px solid var(--border);
            border-radius: 6px;
            overflow: hidden;
            background: #0b1112;
            width: 100%%;
        }
        .model-row {
            display: grid;
            grid-template-columns: minmax(260px, 3fr) repeat(6, minmax(70px, 1fr));
            gap: 8px;
            align-items: center;
            min-height: 44px;
            padding: 0 14px;
            color: #a9b2af;
            border-bottom: 1px solid var(--border);
            font-size: 12px;
            cursor: pointer;
        }
        .model-row:hover {
            background: var(--card-hover);
            color: #ebefee;
        }
        .model-row.table-head {
            background: #0d1314;
            color: #77817e;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            cursor: default;
        }
        .model-name {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #d8dddb;
            font-weight: 600;
            font-family: ui-monospace, monospace;
        }
        .model-mark {
            width: 20px;
            height: 20px;
            border-radius: 5px;
            display: inline-grid;
            place-items: center;
            background: rgba(196, 240, 0, 0.15);
            color: var(--accent);
            font-size: 11px;
            font-weight: 800;
        }
        .model-cards {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
            gap: 12px;
        }
        .model-card {
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px;
            background: #0b1112;
            color: #d8dddb;
        }
        .model-card:hover {
            border-color: #567000;
            background: #141c16;
        }
        .model-card dl {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            margin: 12px 0 0;
            font-size: 12px;
        }
        dt { color: #77817e; font-size: 11px; }
        dd { margin: 2px 0 0; color: #ebefee; font-weight: 600; }

        /* Buttons & Badges */
        .badge {
            background: #131c16;
            color: var(--accent);
            border: 1px solid #567000;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
        }
        .button {
            padding: 6px 12px;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: #0d1213;
            color: #aeb6b4;
            font-size: 12px;
            cursor: pointer;
        }
        .button:hover {
            color: var(--accent);
            border-color: #567000;
            background: #151c11;
        }
        .primary {
            border: 1px solid #b6dc00;
            border-radius: 6px;
            padding: 7px 14px;
            color: #101405;
            background: var(--accent);
            font-weight: 700;
            font-size: 12px;
            cursor: pointer;
        }
        .primary:hover { background: #d5fa23; }

        /* Sync status bar */
        #sync-banner {
            display: none;
            margin-bottom: 20px;
            padding: 14px 18px;
            border-radius: 8px;
            background: #101713;
            border: 1px solid #567000;
        }
        .progress-bar-bg {
            width: 100%%;
            height: 6px;
            background: #1d2828;
            border-radius: 3px;
            overflow: hidden;
            margin-top: 8px;
        }
        .progress-bar-fill {
            height: 100%%;
            background: linear-gradient(90deg, #1f6feb, var(--accent));
            width: 0%%;
            transition: width 0.3s ease;
        }

        /* Playground Chat & Composer */
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 14px;
        }
        .chat-bubbles {
            min-height: 120px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .user-bubble {
            align-self: flex-end;
            background: #1e2b20;
            color: #ebefee;
            border-radius: 9px 9px 2px 9px;
            padding: 10px 14px;
            max-width: 75%%;
            font-size: 14px;
        }
        .assistant-card {
            background: #0b1112;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px 16px;
        }
        .assistant-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
            font-size: 13px;
            color: #ebefee;
        }
        .assistant-text {
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 13px;
            line-height: 1.55;
            color: #d8dddb;
            white-space: pre-wrap;
        }
        .composer-box {
            background: #101616;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
        }
        .composer-box textarea {
            width: 100%%;
            background: transparent;
            border: 0;
            outline: 0;
            color: #ebefee;
            font-size: 13px;
            resize: none;
            line-height: 1.45;
        }
        .composer-footer {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 8px;
        }
        .prompt-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 12px;
        }
        .prompt-chip {
            background: #111718;
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 4px 8px;
            color: #77817e;
            font-size: 11px;
            cursor: pointer;
        }
        .prompt-chip:hover {
            color: #ebefee;
            border-color: #567000;
        }

        /* Drawer / Modal */
        .drawer {
            display: none;
            position: fixed;
            top: 47px;
            right: 0;
            bottom: 0;
            width: min(440px, 100vw);
            z-index: 30;
            background: #0d1314;
            border-left: 1px solid var(--border);
            padding: 24px;
            box-shadow: -20px 0 40px rgba(0,0,0,0.6);
            overflow-y: auto;
        }
        .modal-overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.8);
            z-index: 40;
            place-items: center;
        }
        .modal-card {
            background: #0d1314;
            border: 1px solid var(--border);
            border-radius: 10px;
            width: min(720px, 92vw);
            padding: 24px;
            max-height: 85vh;
            overflow-y: auto;
        }
    </style>
    <script src="/dashboard.js" defer></script>
</head>
<body>
    <!-- Top Navigation Bar -->
    <header class="topbar">
        <button class="brand" type="button" onclick="switchTab('playground')">
            <span class="brand-mark">⌁</span>
            <span>tmc/lib</span>
        </button>

        <div class="global-search">
            <span>⌕</span>
            <input type="text" id="global-search-input" placeholder="Search models, nodes... ⌘K" oninput="searchQuery = this.value; renderModelsList();">
            <kbd>⌘ K</kbd>
        </div>

        <nav aria-label="Primary navigation">
            <button class="nav-link active" data-tab="playground" onclick="switchTab('playground')">Playground</button>
            <button class="nav-link" data-tab="models" onclick="switchTab('models')">Models</button>
            <button class="nav-link" data-tab="cluster" onclick="switchTab('cluster')">Cluster Mesh</button>
            <button class="nav-link" data-tab="egraph" onclick="switchTab('egraph')">E-Graph &amp; Compiler</button>
            <button class="nav-link" data-tab="telemetry" onclick="switchTab('telemetry')">Telemetry</button>
        </nav>

        <div class="top-actions">
            <button id="btn-perfetto" class="btn-top btn-perfetto" onclick="openInPerfetto()">🚀 Open in Perfetto</button>
            <a href="/debug/pprof/" class="btn-top" target="_blank">📊 pprof</a>
        </div>
    </header>

    <div class="main-content">
        <!-- Live P2P Sync Status & Bandwidth Bar -->
        <div id="sync-banner">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span class="badge">⚡ IROH P2P MODEL SYNC</span>
                    <strong id="sync-model-id" style="color: var(--accent); font-family: monospace; font-size: 13px;">Selected model</strong>
                    <span style="font-size: 11px; color: #77817e;">From: <span id="sync-peer" style="color: #ebefee;">peer</span></span>
                </div>
                <div style="display: flex; align-items: center; gap: 14px;">
                    <span style="font-size: 12px; color: #77817e;">Speed: <b id="sync-speed" style="color: var(--accent);">124.5 MB/s</b></span>
                    <span id="sync-pct" style="font-weight: 700; color: #ebefee; font-size: 12px;">0%%</span>
                </div>
            </div>
            <div class="progress-bar-bg">
                <div id="sync-progress" class="progress-bar-fill"></div>
            </div>
            <div id="sync-file" style="font-size: 11px; color: #77817e; margin-top: 5px; font-family: monospace;"></div>
        </div>

        <!-- Telemetry Metrics Strip -->
        <div class="metric-strip">
            <div class="metric-card">
                <div class="metric-label">Active E-Graph Plan</div>
                <div class="metric-value" id="plan" style="font-size: 15px; color: var(--accent);">%s</div>
                <div class="metric-sub">%s</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total GPU Dispatches</div>
                <div class="metric-value" id="dispatches">%d</div>
                <div class="metric-sub">FlightRecorder Events</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Median Latency (P50)</div>
                <div class="metric-value" id="p50">%.1f µs</div>
                <div class="metric-sub">Forward Token Step</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Tail Latency (P95)</div>
                <div class="metric-value" id="p95">%.1f µs</div>
                <div class="metric-sub">Hardware Synchronized</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Effective Generation</div>
                <div class="metric-value" id="toks" style="color: var(--accent);">%.1f tok/s</div>
                <div class="metric-sub">Layer Model Decode</div>
            </div>
        </div>

        <!-- TAB 1: PLAYGROUND VIEW -->
        <div id="view-playground" class="tab-view">
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">💬 Live Streaming Inference Playground (/v1/chat/completions)</div>
                    <div style="font-size: 12px; color: #77817e;">OpenAI Compatible Endpoint</div>
                </div>

                <div class="prompt-chips">
                    <span style="font-size: 11px; color: #77817e; align-self: center; margin-right: 4px;">Suggestions:</span>
                    <button class="prompt-chip" onclick="setPromptText('The key to scalable AI architectures is')">AI Architecture</button>
                    <button class="prompt-chip" onclick="setPromptText('Which number is larger: 9.9 or 9.11?')">9.9 vs 9.11</button>
                    <button class="prompt-chip" onclick="setPromptText('Should you walk or drive to a nearby car wash?')">Car Wash</button>
                    <button class="prompt-chip" onclick="setPromptText('Write a Go function for zero-copy memory allocation with Metal GPU')">Metal Go</button>
                </div>

                <div class="chat-container">
                    <div class="composer-box">
                        <textarea id="prompt-input" rows="3" placeholder="Type a prompt for the model... (⌘ Enter to send)" onkeydown="if(event.key==='Enter' && (event.metaKey || event.ctrlKey)){ event.preventDefault(); sendPrompt(); }">The key to scalable AI architectures is</textarea>
                        <div class="composer-footer">
                            <span style="font-size: 11px; color: #77817e;">⌘ Enter to stream completion</span>
                            <button id="gen-btn" class="primary" onclick="sendPrompt()">↑ Send</button>
                        </div>
                    </div>

                    <div class="chat-bubbles">
                        <div class="assistant-card">
                            <div class="assistant-header">
                                <span class="model-mark">⚡</span>
                                <strong>Model Output (Live Stream)</strong>
                                <span id="routing-badge" style="display:none; margin-left:auto; font-size:11px; color:#77817e;"></span>
                            </div>
                            <div id="output-box" class="assistant-text">Ready for inference. Click Send to stream tokens live across the cluster mesh.</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Quick Sync Selector Panel -->
            <div class="panel" id="mesh-model-picker">
                <div class="panel-header">
                    <div class="panel-title">📦 Mesh Model Selection</div>
                    <div style="font-size: 12px; color: #77817e;">Hugging Face Hub Sync</div>
                </div>
                <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
                    <select id="mesh-model-select" class="button" style="flex:1; min-width:280px; height:34px; padding:0 10px; background:#101616; color:#ebefee;" disabled>
                        <option>Discovering remote model snapshots…</option>
                    </select>
                    <button id="mesh-model-sync" class="primary" onclick="syncSelectedMeshModel()" disabled>⚡ Sync Selected Model</button>
                </div>
            </div>
        </div>

        <!-- TAB 2: MODELS CATALOG VIEW -->
        <div id="view-models" class="tab-view" style="display: none;">
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">📦 Discovered Models Catalog (~/.cache/huggingface/hub)</div>
                    <span id="models-count-summary" style="font-size: 12px; color: #77817e;">Discovered models in cluster</span>
                </div>

                <div class="modality-tabs">
                    <button class="active" data-modality="All" onclick="filterModality('All')">All</button>
                    <button data-modality="Text" onclick="filterModality('Text')">Text</button>
                    <button data-modality="NVFP4" onclick="filterModality('NVFP4')">NVFP4</button>
                    <button data-modality="MXFP4" onclick="filterModality('MXFP4')">MXFP4</button>
                    <button data-modality="FP8" onclick="filterModality('FP8')">FP8</button>
                    <button data-modality="BF16" onclick="filterModality('BF16')">BF16</button>
                    <button data-modality="GGUF" onclick="filterModality('GGUF')">GGUF</button>

                    <div class="view-switch">
                        <button class="active" data-view="table" onclick="setModelViewMode('table')">▦ Table</button>
                        <button data-view="cards" onclick="setModelViewMode('cards')">☷ Cards</button>
                    </div>
                </div>

                <div id="models-table-view" class="model-table">
                    <div class="model-row table-head">
                        <span>Model Identifier</span>
                        <span>Size</span>
                        <span>Files</span>
                        <span style="grid-column: span 3; text-align: left;">Hosted on Mesh Nodes</span>
                        <span style="text-align: right;">Action</span>
                    </div>
                    <div id="models-tbody">
%s
                    </div>
                </div>

                <div id="models-cards-view" class="model-cards" style="display: none;">
                    <!-- Populated by JS -->
                </div>
            </div>
        </div>

        <!-- TAB 3: CLUSTER MESH VIEW -->
        <div id="view-cluster" class="tab-view" style="display: none;">
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">🌐 go-iroh P2P Cluster Mesh Topology</div>
                    <span class="badge">🟢 %d Nodes Active (%.0f TFLOPs, %.0f GB/s Cluster)</span>
                </div>
                <div style="font-size: 13px; color: #77817e; margin-bottom: 16px;">
                    Active Role Assignment: <strong style="color: #ebefee;">%s</strong>
                </div>

                <canvas id="mesh-topology" width="1120" height="300" style="width:100%%; height:300px; background:#0b1112; border:1px solid var(--border); border-radius:6px;"></canvas>
                <div id="mesh-topology-detail" style="font: 12px ui-monospace, monospace; color: #77817e; margin-top: 10px;">Loading registered mesh peers. Transport path mode is unavailable until connection-path telemetry is reported.</div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">📦 hfsync Transfer Monitor</div>
                    <span class="badge">reported job state</span>
                </div>
                <div id="mesh-sync-hud" style="display: grid; gap: 10px;"></div>
            </div>

            <div class="panel">
                <div class="model-table">
                    <div class="model-row table-head">
                        <span>Node ID (Ed25519)</span>
                        <span>OS</span>
                        <span>Arch</span>
                        <span>Compute</span>
                        <span>Bandwidth</span>
                        <span>E-Graph Role</span>
                        <span style="text-align: right;">Cached Models</span>
                    </div>
                    <div id="mesh-nodes-table">
%s
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 4: E-GRAPH & COMPILER VIEW -->
        <div id="view-egraph" class="tab-view" style="display: none;">
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">🧩 E-Graph Equality Saturation &amp; Megakernel Plan</div>
                    <div style="display: flex; gap: 8px;">
                        <button class="button" onclick="showGraphPlan('before')">Before Saturation</button>
                        <button class="button" onclick="showGraphPlan('after')">Extracted Optimal Plan</button>
                    </div>
                </div>
                <div id="graph-plan" style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px;"></div>
                <div id="graph-rewrites" style="font: 12px ui-monospace, monospace; color: #77817e; margin-top: 14px;"></div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">📈 GPU Decode-Step Latency Waterfall</div>
                    <button class="button" onclick="openInPerfetto()">🚀 Open in Perfetto</button>
                </div>
                <canvas id="kernel-waterfall" width="1120" height="180" style="width:100%%; height:180px; background:#0b1112; border:1px solid var(--border); border-radius:6px;"></canvas>
                <div id="kernel-waterfall-detail" style="font: 12px ui-monospace, monospace; color: #77817e; margin-top: 10px;">Waiting for recorded GPU dispatches.</div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">📐 Hardware Roofline</div>
                    <span class="badge" title="Nominal capability ceilings; workload points require FLOP and byte counters.">heuristic ceilings</span>
                </div>
                <canvas id="roofline-canvas" width="1120" height="300" style="width:100%%; height:300px; background:#0b1112; border:1px solid var(--border); border-radius:6px;"></canvas>
                <div id="roofline-detail" style="font: 12px ui-monospace, monospace; color: #77817e; margin-top: 10px;">Loading heuristic capacity ceilings. Operational points are unmeasured until FLOP and byte counters are available.</div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">🧪 PGO &amp; Adaptive JIT</div>
                    <span id="pgo-state" class="badge">loading</span>
                </div>
                <canvas id="pgo-trials" width="1120" height="180" style="width:100%%; height:180px; background:#0b1112; border:1px solid var(--border); border-radius:6px;"></canvas>
                <div id="pgo-detail" style="font: 12px ui-monospace, monospace; color: #77817e; margin-top: 10px;">Loading PGO telemetry.</div>
                <div id="pgo-history" style="display: grid; gap: 6px; margin-top: 12px;"></div>
                <div id="pgo-hotswaps" style="display: grid; gap: 6px; margin-top: 12px;"></div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">🗄️ KV &amp; Prefix Cache Capability</div>
                    <span class="badge">evidence-bounded</span>
                </div>
                <div id="memory-kv" style="display: grid; gap: 8px;"></div>
                <div id="memory-prefix" style="display: grid; gap: 8px; margin-top: 10px;"></div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">⚙️ Precision, Speculation &amp; Batching</div>
                    <span class="badge">runtime facts only</span>
                </div>
                <div id="memory-precision" style="display: grid; gap: 8px;"></div>
                <div id="memory-speculation" style="display: grid; gap: 8px; margin-top: 10px;"></div>
                <div id="memory-batching" style="display: grid; gap: 8px; margin-top: 10px;"></div>
            </div>
        </div>

        <!-- TAB 5: TELEMETRY & EVENT LOGS -->
        <div id="view-telemetry" class="tab-view" style="display: none;">
            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">🧵 OpenAI Request-Slot Queue</div>
                    <span class="badge">actual handler admission</span>
                </div>
                <div id="request-slot-queue" role="status" aria-live="polite" style="display: grid; gap: 8px;"></div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">🌳 Speculative Draft Tree</div>
                    <span class="badge">paired execution only</span>
                </div>
                <canvas id="speculative-tree-canvas" role="img" aria-describedby="speculative-tree-detail" width="1120" height="220" style="width:100%%; height:220px; background:#0b1112; border:1px solid var(--border); border-radius:6px;"></canvas>
                <div id="speculative-tree-detail" style="font: 12px ui-monospace, monospace; color: #77817e; margin-top: 10px;">Waiting for paired draft and target execution telemetry.</div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">📜 Recent Cluster Generations &amp; Attribution</div>
                </div>
                <div class="model-table">
                    <div class="model-row table-head">
                        <span>Time</span>
                        <span>Prompt</span>
                        <span>Prefill Worker</span>
                        <span>Decode Worker</span>
                        <span>Tokens / Speed</span>
                        <span style="text-align: right;">Duration</span>
                    </div>
                    <div id="gen-table">
%s
                    </div>
                </div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <div class="panel-title">Recent FlightRecorder Kernel Events</div>
                </div>
                <div class="model-table">
                    <div class="model-row table-head">
                        <span>Timestamp (ns)</span>
                        <span>Kernel Name</span>
                        <span>GPU (µs)</span>
                        <span>Grid</span>
                        <span>Threadgroup</span>
                        <span style="text-align: right;">Labels</span>
                    </div>
                    <div id="event-table">
%s
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Model Detail Drawer -->
    <div id="model-drawer" class="drawer">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px;">
            <div>
                <span style="font-size: 11px; color: var(--accent); font-weight: 700; text-transform: uppercase;">Model Specification</span>
                <h2 id="drawer-model-name" style="margin: 4px 0 0; color: #ebefee; font-size: 18px; word-break: break-all;">Model</h2>
            </div>
            <button class="button" style="padding: 4px 8px;" onclick="closeModelDrawer()">✕</button>
        </div>

        <dl style="margin: 16px 0; padding: 14px 0; border-top: 1px solid var(--border); border-bottom: 1px solid var(--border);">
            <div><dt>Disk Footprint</dt><dd id="drawer-model-size">-</dd></div>
            <div><dt>Files</dt><dd id="drawer-model-files">-</dd></div>
        </dl>

        <h3 style="font-size: 13px; color: #ebefee; margin: 16px 0 10px;">Available Across Mesh Nodes</h3>
        <div id="drawer-model-nodes"></div>
    </div>

    <!-- Node Detail Modal -->
    <div id="node-modal" class="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="node-modal-title" tabindex="-1" onclick="if(event.target===this) closeNodeModal()">
        <div class="modal-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid var(--border); padding-bottom: 14px; margin-bottom: 16px;">
                <div>
                    <h2 id="node-modal-title" style="margin: 0; font-size: 18px; color: var(--text-bright);">🖥️ Node Detail &amp; Model Storage</h2>
                    <div style="font-size: 12px; color: var(--accent); font-family: monospace; margin-top: 4px;" id="modal-node-id"></div>
                </div>
                <button id="node-modal-close" type="button" onclick="closeNodeModal()" class="button" style="padding: 4px 10px;">✕ Close</button>
            </div>

            <div class="metric-strip" style="grid-template-columns: repeat(3, 1fr); margin-bottom: 20px;">
                <div class="metric-card">
                    <div class="metric-label">Platform / Arch</div>
                    <div id="modal-node-platform" style="font-weight: 600; color: var(--text-bright); font-size: 14px;"></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Compute Power</div>
                    <div id="modal-node-tflops" style="font-weight: 600; color: var(--accent); font-size: 14px;"></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Memory Bandwidth</div>
                    <div id="modal-node-bw" style="font-weight: 600; color: var(--accent-blue); font-size: 14px;"></div>
                </div>
            </div>

            <h3 style="font-size: 14px; color: var(--text-bright); margin-bottom: 10px;">📁 Local Cached Models on this Node</h3>
            <div id="modal-local-models" style="margin-bottom: 24px;"></div>

            <h3 style="font-size: 14px; color: var(--text-bright); margin-bottom: 10px;">🌐 Remote Models Available in Mesh (P2P Sync)</h3>
            <div id="modal-remote-models"></div>
        </div>
    </div>
</body>
</html>`
