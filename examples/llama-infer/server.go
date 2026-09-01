package main

import (
	"context"
	"encoding/json"
	"fmt"
	"html"
	"net/http"
	"net/http/pprof"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/tmc/lib/apple/x/metal/metalprof"
)

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

// TelemetryResponse represents the JSON payload served at /api/telemetry.
type TelemetryResponse struct {
	ModelName         string                  `json:"model_name"`
	HiddenDim         int                     `json:"hidden_dim"`
	NumLayers         int                     `json:"num_layers"`
	EGraphSats        int                     `json:"egraph_saturations"`
	ActivePlan        string                  `json:"active_plan"`
	TotalDispatches   int                     `json:"total_dispatches"`
	MeanLatencyUs     float64                 `json:"mean_latency_us"`
	P50LatencyUs      float64                 `json:"p50_latency_us"`
	P95LatencyUs      float64                 `json:"p95_latency_us"`
	EffectiveTokS     float64                 `json:"effective_tok_s"`
	ClusterMesh       ClusterMeshState        `json:"cluster_mesh"`
	RecentGenerations []GenerationRecord      `json:"recent_generations"`
	ActiveSyncs       []*SyncJobStatus        `json:"active_syncs"`
	RecentEvents      []metalprof.KernelEvent `json:"recent_events"`
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
		engine:   engine,
		cfg:      cfg,
		mux:      http.NewServeMux(),
		syncJobs: make(map[string]*SyncJobStatus),
	}
	s.routes()
	return s
}

// SetClusterNode attaches the active go-iroh P2P cluster node to the telemetry server.
func (s *TelemetryServer) SetClusterNode(node *IrohClusterNode) {
	s.clusterNode = node
}

func (s *TelemetryServer) routes() {
	s.mux.HandleFunc("/", s.handleDashboard)
	s.mux.HandleFunc("/api/telemetry", s.handleAPITelemetry)
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
		s.clusterNode.RegisterPeer(peer)
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
	if s.engine != nil && s.engine.recorder != nil {
		events = s.engine.recorder.Events()
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
	if !s.cfg.FuseEGraph {
		activePlan = "Discrete Multi-Pass Pipeline"
	}

	// Truncate to most recent 50 events for UI view
	recent := events
	if len(recent) > 50 {
		recent = recent[len(recent)-50:]
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

	return TelemetryResponse{
		ModelName:         s.cfg.Name,
		HiddenDim:         s.cfg.Dim,
		NumLayers:         s.cfg.NumLayers,
		EGraphSats:        s.engine.egraphSats,
		ActivePlan:        activePlan,
		TotalDispatches:   len(events),
		MeanLatencyUs:     meanUs,
		P50LatencyUs:      p50Us,
		P95LatencyUs:      p95Us,
		EffectiveTokS:     tokS,
		ClusterMesh:       mesh,
		RecentGenerations: s.getRecentGenerations(),
		ActiveSyncs:       s.getActiveSyncs(),
		RecentEvents:      recent,
	}
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
                    <tr style="cursor:pointer;" onclick="openNodeModal('%s')">
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
		stats.ActivePlan, stats.EGraphSats,
		stats.TotalDispatches,
		stats.P50LatencyUs,
		stats.P95LatencyUs,
		stats.EffectiveTokS,
		stats.ClusterMesh.TotalNodes, stats.ClusterMesh.ClusterTFLOPs, stats.ClusterMesh.ClusterBandwidthGBS,
		stats.ClusterMesh.ActiveAssignment.Reason,
		meshRows.String(),
		modelCatalogRows.String(),
		genRows.String(),
		eventRows.String(),
	)

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write([]byte(htmlOut))
}

const htmlTemplate = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>tmc/lib - E-Graph & FlightRecorder Telemetry</title>
    <style>
        :root {
            --bg: #0d1117;
            --card-bg: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --text-bright: #ffffff;
            --accent: #58a6ff;
            --accent-green: #3fb950;
            --accent-purple: #bc8cff;
            --accent-orange: #d29922;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 24px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
        h1 { margin: 0; font-size: 24px; color: var(--text-bright); }
        .badge { background: #238636; color: white; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; display: inline-flex; align-items: center; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
        .card-label { font-size: 12px; text-transform: uppercase; color: #8b949e; margin-bottom: 6px; }
        .card-value { font-size: 26px; font-weight: 700; color: var(--text-bright); }
        .card-sub { font-size: 12px; color: var(--accent); margin-top: 4px; }
        .section { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 24px; }
        .section-title { font-size: 16px; font-weight: 600; color: var(--text-bright); margin-bottom: 14px; }
        table { width: 100%%; border-collapse: collapse; font-size: 13px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
        th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border); }
        th { color: #8b949e; font-weight: 600; }
        tr:hover { background-color: rgba(255,255,255,0.03); }
        .btn { background: #21262d; border: 1px solid var(--border); color: var(--accent); padding: 6px 14px; border-radius: 6px; text-decoration: none; font-size: 13px; font-weight: 500; cursor: pointer; }
        .btn:hover { background: #30363d; }
        .btn-sync { background: #238636; color: white; border: none; font-weight: 600; padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 12px; }
        .btn-sync:hover { background: #2ea043; }

        /* Modal styling */
        .modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%%; height: 100%%; background: rgba(0,0,0,0.75); z-index: 1000; justify-content: center; align-items: center; }
        .modal-content { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; width: 90%%; max-width: 800px; max-height: 85vh; overflow-y: auto; padding: 24px; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }

        /* Sync status bar */
        #sync-banner { display: none; margin-bottom: 24px; padding: 16px; border-radius: 8px; background: #161b22; border: 1px solid var(--accent); }
        .progress-bar-bg { width: 100%%; height: 10px; background: #21262d; border-radius: 5px; overflow: hidden; margin-top: 10px; }
        .progress-bar-fill { height: 100%%; background: linear-gradient(90deg, #1f6feb, #3fb950); width: 0%%; transition: width 0.3s ease; }
    </style>
    <script>
        let clusterTelemetry = null;

        function formatBytesJS(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
        }

        async function refreshTelemetry() {
            try {
                const res = await fetch('/api/telemetry');
                const data = await res.json();
                clusterTelemetry = data;

                document.getElementById('dispatches').innerText = data.total_dispatches;
                document.getElementById('p50').innerText = data.p50_latency_us.toFixed(1) + ' µs';
                document.getElementById('p95').innerText = data.p95_latency_us.toFixed(1) + ' µs';
                document.getElementById('toks').innerText = data.effective_tok_s.toFixed(1) + ' tok/s';
                document.getElementById('plan').innerText = data.active_plan;

                // Sync status banner update
                const banner = document.getElementById('sync-banner');
                if (data.active_syncs && data.active_syncs.length > 0) {
                    const latest = data.active_syncs[data.active_syncs.length - 1];
                    banner.style.display = 'block';
                    document.getElementById('sync-model-id').innerText = latest.model_id;
                    document.getElementById('sync-peer').innerText = (latest.source_platform || 'Peer') + ' (' + (latest.source_node_id ? latest.source_node_id.slice(0, 8) : 'cluster') + ')';
                    document.getElementById('sync-speed').innerText = latest.throughput_mbs.toFixed(1) + ' MB/s';
                    document.getElementById('sync-pct').innerText = latest.percent.toFixed(1) + '%%';
                    document.getElementById('sync-progress').style.width = latest.percent + '%%';
                    document.getElementById('sync-file').innerText = latest.current_file ? 'File: ' + latest.current_file : (latest.state === 'completed' ? 'Synchronization Complete!' : '');
                    if (latest.state === 'completed') {
                        document.getElementById('sync-speed').innerText = 'Done (' + latest.throughput_mbs.toFixed(1) + ' MB/s avg)';
                    }
                }

                if (data.recent_generations && data.recent_generations.length > 0) {
                    const tbody = document.getElementById('gen-table');
                    if (tbody) {
                        tbody.innerHTML = data.recent_generations.slice().reverse().map(function(g) {
                            const pShort = g.prefill_node ? g.prefill_node.slice(0, 8) : '';
                            const dShort = g.decode_node ? g.decode_node.slice(0, 8) : '';
                            const timeStr = new Date(g.timestamp).toLocaleTimeString();
                            return '<tr>' +
                                '<td style="color:#8b949e;">' + timeStr + '</td>' +
                                '<td style="color:var(--text-bright); max-width:220px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="' + g.prompt + '">' + g.prompt + '</td>' +
                                '<td><span class="badge" style="background:#1f6feb;">' + g.prefill_platform + '</span> <code>' + pShort + '</code></td>' +
                                '<td><span class="badge" style="background:#8957e5;">' + g.decode_platform + '</span> <code>' + dShort + '</code></td>' +
                                '<td style="color:#3fb950; font-weight:600;">' + g.tokens_generated + ' tok (' + g.tok_per_sec.toFixed(1) + ' tok/s)</td>' +
                                '<td style="color:#8b949e;">' + g.duration_ms.toFixed(1) + ' ms</td>' +
                            '</tr>';
                        }).join('');
                    }
                }
            } catch (e) {}
        }
        refreshTelemetry();
        setInterval(refreshTelemetry, 1000);

        async function triggerSync(modelId, sourceNodeId) {
            try {
                const banner = document.getElementById('sync-banner');
                banner.style.display = 'block';
                document.getElementById('sync-model-id').innerText = modelId;
                document.getElementById('sync-peer').innerText = sourceNodeId ? sourceNodeId.slice(0, 8) : 'cluster';
                document.getElementById('sync-pct').innerText = 'Starting...';
                document.getElementById('sync-progress').style.width = '5%%';

                const res = await fetch('/api/mesh/sync', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_id: modelId, source_node_id: sourceNodeId })
                });
                const job = await res.json();
                closeNodeModal();
            } catch (err) {
                alert('Sync trigger failed: ' + err.message);
            }
        }

        function openNodeModal(nodeId) {
            if (!clusterTelemetry || !clusterTelemetry.cluster_mesh) return;
            const mesh = clusterTelemetry.cluster_mesh;
            const allNodes = [mesh.self_node].concat(mesh.peers || []);
            const node = allNodes.find(n => n.node_id === nodeId) || mesh.self_node;

            document.getElementById('modal-node-id').innerText = node.node_id;
            document.getElementById('modal-node-platform').innerText = node.os.toUpperCase() + ' / ' + node.arch;
            document.getElementById('modal-node-tflops').innerText = node.compute_tflops.toFixed(1) + ' TFLOPs';
            document.getElementById('modal-node-bw').innerText = node.bandwidth_gbs.toFixed(1) + ' GB/s';

            // Render Local Models
            const localList = document.getElementById('modal-local-models');
            if (node.available_models && node.available_models.length > 0) {
                localList.innerHTML = node.available_models.map(m => {
                    return '<div style="display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid #30363d;">' +
                        '<div><b style="color:var(--accent-purple); font-family:monospace;">' + m.model_id + '</b>' +
                        '<div style="font-size:12px; color:#8b949e;">Snapshot: <code>' + (m.revision ? m.revision.slice(0,8) : 'main') + '</code> · ' + m.file_count + ' files</div></div>' +
                        '<span class="badge" style="background:#238636;">' + formatBytesJS(m.total_bytes) + '</span>' +
                    '</div>';
                }).join('');
            } else {
                localList.innerHTML = '<div style="color:#8b949e; padding:10px 0;">No cached models found on this node.</div>';
            }

            // Render Remote Models available from other peers
            const remoteList = document.getElementById('modal-remote-models');
            const otherNodes = allNodes.filter(n => n.node_id !== node.node_id);
            const remoteMap = new Map();
            otherNodes.forEach(other => {
                (other.available_models || []).forEach(m => {
                    const localHas = (node.available_models || []).some(lm => lm.model_id === m.model_id);
                    if (!localHas && !remoteMap.has(m.model_id)) {
                        remoteMap.set(m.model_id, { model: m, sourceNode: other });
                    }
                });
            });

            if (remoteMap.size > 0) {
                let html = '';
                remoteMap.forEach((val, modelId) => {
                    html += '<div style="display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid #30363d;">' +
                        '<div><b style="color:var(--text-bright); font-family:monospace;">' + modelId + '</b>' +
                        '<div style="font-size:12px; color:#8b949e;">Available from: <span class="badge" style="background:#1f6feb;">' + val.sourceNode.os.toUpperCase() + '</span> <code>' + val.sourceNode.node_id.slice(0,8) + '</code> (' + formatBytesJS(val.model.total_bytes) + ')</div></div>' +
                        '<button class="btn-sync" onclick="triggerSync(\'' + modelId + '\', \'' + val.sourceNode.node_id + '\')">⚡ Sync to this node</button>' +
                    '</div>';
                });
                remoteList.innerHTML = html;
            } else {
                remoteList.innerHTML = '<div style="color:#8b949e; padding:10px 0;">All cluster models are already synchronized to this node.</div>';
            }

            document.getElementById('node-modal').style.display = 'flex';
        }

        function closeNodeModal() {
            document.getElementById('node-modal').style.display = 'none';
        }

        async function sendPrompt() {
            const input = document.getElementById('prompt-input');
            const out = document.getElementById('output-box');
            const badge = document.getElementById('routing-badge');
            const btn = document.getElementById('gen-btn');
            const prompt = input.value.trim();
            if (!prompt) return;

            btn.disabled = true;
            btn.innerText = 'Generating...';
            out.innerText = '';
            badge.style.display = 'none';

            try {
                const res = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: prompt, max_tokens: 64, stream: true })
                });

                const reader = res.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let buffer = '';

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        const trimmed = line.trim();
                        if (!trimmed || !trimmed.startsWith('data: ')) continue;
                        const payload = trimmed.slice(6);
                        if (payload === '[DONE]') break;
                        try {
                            const chunk = JSON.parse(payload);
                            const content = chunk.choices[0]?.delta?.content || '';
                            out.innerText += content;

                            if (chunk.node_routing && badge) {
                                const r = chunk.node_routing;
                                badge.innerHTML = 
                                    '<span style="color:var(--accent-green); font-weight:600;">⚡ Prefill Worker:</span> ' +
                                    '<b>' + r.prefill_platform + '</b> (<code style="color:var(--accent);">' + r.prefill_node_id.slice(0, 8) + '</code>, ' + r.prefill_tflops + ' TFLOPs)' +
                                    ' &nbsp;➔&nbsp; ' +
                                    '<span style="color:var(--accent-purple); font-weight:600;">🔥 Decode Worker:</span> ' +
                                    '<b>' + r.decode_platform + '</b> (<code style="color:var(--accent);">' + r.decode_node_id.slice(0, 8) + '</code>, ' + r.decode_bandwidth_gbs + ' GB/s)' +
                                    ' &nbsp;·&nbsp; ' +
                                    '<span style="color:#8b949e;">' + r.total_nodes + ' Nodes in Active Mesh</span>';
                                badge.style.display = 'flex';
                            }
                        } catch (err) {}
                    }
                }
            } catch (e) {
                out.innerText = 'Error: ' + e.message;
            } finally {
                btn.disabled = false;
                btn.innerText = 'Generate (SSE Stream)';
            }
        }
    </script>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>tmc/lib Adaptive E-Graph Telemetry</h1>
                <div style="font-size: 13px; color: #8b949e; margin-top: 4px;">Live Closed-Loop Hardware Profiler &amp; OpenAI Compatible Serving Engine</div>
            </div>
            <div>
                <a href="/debug/pprof/" class="btn" style="margin-right: 8px;">📊 Runtime pprof</a>
                <a href="/debug/pprof/gpu" class="btn">⬇ GPU FlightRecorder Profile</a>
            </div>
        </header>

        <!-- Live P2P Sync Status & Bandwidth Bar -->
        <div id="sync-banner">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span class="badge" style="background: #1f6feb;">⚡ IROH P2P MODEL SYNC</span>
                    <strong id="sync-model-id" style="color: var(--accent-purple); font-family: monospace; font-size: 14px;">unsloth/Llama-3.2-1B-Instruct</strong>
                    <span style="font-size: 12px; color: #8b949e;">From: <span id="sync-peer" style="color: var(--text-bright); font-weight: 600;">DGX Blackwell</span></span>
                </div>
                <div style="display: flex; align-items: center; gap: 16px;">
                    <span style="font-size: 13px; color: #8b949e;">Transfer Speed: <b id="sync-speed" style="color: var(--accent-green);">124.5 MB/s</b></span>
                    <span id="sync-pct" style="font-weight: 700; color: var(--text-bright);">0%%</span>
                </div>
            </div>
            <div class="progress-bar-bg">
                <div id="sync-progress" class="progress-bar-fill"></div>
            </div>
            <div id="sync-file" style="font-size: 11px; color: #8b949e; margin-top: 6px; font-family: monospace;"></div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="card-label">Active E-Graph Plan</div>
                <div class="card-value" id="plan" style="font-size: 18px; color: var(--accent-purple);">%s</div>
                <div class="card-sub">%d Saturation Iterations</div>
            </div>
            <div class="card">
                <div class="card-label">Total GPU Dispatches</div>
                <div class="card-value" id="dispatches">%d</div>
                <div class="card-sub">In-Memory Ring Buffer</div>
            </div>
            <div class="card">
                <div class="card-label">Median Latency (P50)</div>
                <div class="card-value" id="p50">%.1f µs</div>
                <div class="card-sub">Per Token Forward Pass</div>
            </div>
            <div class="card">
                <div class="card-label">Tail Latency (P95)</div>
                <div class="card-value" id="p95">%.1f µs</div>
                <div class="card-sub">Hardware Synchronized</div>
            </div>
            <div class="card">
                <div class="card-label">Effective Rate</div>
                <div class="card-value" id="toks" style="color: var(--accent-green);">%.1f tok/s</div>
                <div class="card-sub">Layer Model Decode</div>
            </div>
        </div>

        <div class="section">
            <div class="section-title">💬 Live Streaming Inference Playground (/v1/chat/completions)</div>
            <div style="display: flex; gap: 10px; margin-bottom: 12px;">
                <input type="text" id="prompt-input" placeholder="Type a prompt..." value="The key to scalable AI architectures is" style="flex: 1; background: #0d1117; border: 1px solid var(--border); color: var(--text-bright); padding: 10px 14px; border-radius: 6px; font-size: 14px;">
                <button id="gen-btn" onclick="sendPrompt()" class="btn" style="background: #238636; color: white; font-weight: 600; border: none;">Generate (SSE Stream)</button>
            </div>
            <div id="output-box" style="background: #0d1117; border: 1px solid var(--border); border-radius: 6px; padding: 14px; min-height: 48px; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 14px; line-height: 1.5; white-space: pre-wrap; color: var(--text-bright);">Ready for inference. Click Generate to stream tokens live.</div>
            <div id="routing-badge" style="display: none; align-items: center; gap: 8px; font-size: 12px; margin-top: 10px; padding: 8px 14px; background: #21262d; border-radius: 6px; border: 1px solid var(--border); color: #c9d1d9;"></div>
        </div>

        <div class="section">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div class="section-title" style="margin-bottom: 0;">🌐 go-iroh P2P Cluster Mesh Topology</div>
                <span class="badge" style="background: #238636; font-size: 13px;">🟢 %d Nodes Active (%.0f TFLOPs, %.0f GB/s Cluster)</span>
            </div>
            <div style="font-size: 13px; color: #8b949e; margin-bottom: 14px;">
                E-Graph Role Assignment: <strong style="color: var(--text-bright);">%s</strong> (Click any node row to inspect cached models &amp; sync)
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Node ID (Ed25519)</th>
                        <th>Platform</th>
                        <th>Architecture</th>
                        <th>Compute Power</th>
                        <th>Memory Bandwidth</th>
                        <th>Assigned E-Graph Role</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody id="mesh-nodes-table">
%s
                </tbody>
            </table>
        </div>

        <div class="section">
            <div class="section-title">📦 Hugging Face Models in Cluster Mesh (~/.cache/huggingface/hub)</div>
            <table>
                <thead>
                    <tr>
                        <th>Model ID</th>
                        <th>Size on Disk</th>
                        <th>Snapshot Files</th>
                        <th>Hosted on Nodes</th>
                    </tr>
                </thead>
                <tbody id="models-table">
%s
                </tbody>
            </table>
        </div>

        <div class="section">
            <div class="section-title">📜 Recent Cluster Generations &amp; Node Attribution</div>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Prompt</th>
                        <th>Prefill Worker</th>
                        <th>Decode Worker</th>
                        <th>Tokens / Speed</th>
                        <th>Duration</th>
                    </tr>
                </thead>
                <tbody id="gen-table">
%s
                </tbody>
            </table>
        </div>

        <div class="section">
            <div class="section-title">Recent Kernel Execution Events (FlightRecorder Stream)</div>
            <table>
                <thead>
                    <tr>
                        <th>Timestamp (ns)</th>
                        <th>Kernel Name</th>
                        <th>GPU Duration (µs)</th>
                        <th>Grid Size</th>
                        <th>Threadgroup</th>
                        <th>Context Labels</th>
                    </tr>
                </thead>
                <tbody id="event-table">
%s
                </tbody>
            </table>
        </div>
    </div>

    <!-- Node Detail Modal -->
    <div id="node-modal" class="modal-overlay" onclick="if(event.target===this) closeNodeModal()">
        <div class="modal-content">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid #30363d; padding-bottom: 14px; margin-bottom: 16px;">
                <div>
                    <h2 style="margin: 0; font-size: 18px; color: var(--text-bright);">🖥️ Node Detail &amp; Model Storage</h2>
                    <div style="font-size: 12px; color: var(--accent); font-family: monospace; margin-top: 4px;" id="modal-node-id"></div>
                </div>
                <button onclick="closeNodeModal()" class="btn" style="padding: 4px 10px;">✕ Close</button>
            </div>

            <div class="grid" style="grid-template-columns: repeat(3, 1fr); margin-bottom: 20px;">
                <div class="card">
                    <div class="card-label">Platform / Arch</div>
                    <div id="modal-node-platform" style="font-weight: 600; color: var(--text-bright); font-size: 15px;"></div>
                </div>
                <div class="card">
                    <div class="card-label">Compute Capability</div>
                    <div id="modal-node-tflops" style="font-weight: 600; color: var(--accent); font-size: 15px;"></div>
                </div>
                <div class="card">
                    <div class="card-label">Memory Bandwidth</div>
                    <div id="modal-node-bw" style="font-weight: 600; color: var(--accent-green); font-size: 15px;"></div>
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
