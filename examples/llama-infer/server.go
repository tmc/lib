package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/pprof"
	"sort"
	"strings"
	"time"

	"github.com/tmc/lib/apple/x/metal/metalprof"
)

// TelemetryResponse represents the JSON payload served at /api/telemetry.
type TelemetryResponse struct {
	ModelName       string                  `json:"model_name"`
	HiddenDim       int                     `json:"hidden_dim"`
	NumLayers       int                     `json:"num_layers"`
	EGraphSats      int                     `json:"egraph_saturations"`
	ActivePlan      string                  `json:"active_plan"`
	TotalDispatches int                     `json:"total_dispatches"`
	MeanLatencyUs   float64                 `json:"mean_latency_us"`
	P50LatencyUs    float64                 `json:"p50_latency_us"`
	P95LatencyUs    float64                 `json:"p95_latency_us"`
	EffectiveTokS   float64                 `json:"effective_tok_s"`
	ClusterMesh     ClusterMeshState        `json:"cluster_mesh"`
	RecentEvents    []metalprof.KernelEvent `json:"recent_events"`
}

// TelemetryServer serves a live diagnostic dashboard for E-Graph and FlightRecorder telemetry.
type TelemetryServer struct {
	engine      *ModelEngine
	clusterNode *IrohClusterNode
	cfg         Config
	mux         *http.ServeMux
}

// NewTelemetryServer initializes a new diagnostic HTTP server.
func NewTelemetryServer(engine *ModelEngine, cfg Config) *TelemetryServer {
	s := &TelemetryServer{
		engine: engine,
		cfg:    cfg,
		mux:    http.NewServeMux(),
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
		ModelName:       s.cfg.Name,
		HiddenDim:       s.cfg.Dim,
		NumLayers:       s.cfg.NumLayers,
		EGraphSats:      s.engine.egraphSats,
		ActivePlan:      activePlan,
		TotalDispatches: len(events),
		MeanLatencyUs:   meanUs,
		P50LatencyUs:    p50Us,
		P95LatencyUs:    p95Us,
		EffectiveTokS:   tokS,
		ClusterMesh:     mesh,
		RecentEvents:    recent,
	}
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

		meshRows.WriteString(fmt.Sprintf(`
                    <tr>
                        <td style="color:var(--accent); font-weight:600; font-family:monospace;">%s</td>
                        <td><span class="badge" style="background:#1f6feb;">%s</span></td>
                        <td style="color:#8b949e;">%s</td>
                        <td style="color:#ffffff; font-weight:600;">%.1f TFLOPs</td>
                        <td style="color:#ffffff; font-weight:600;">%.1f GB/s</td>
                        <td>%s</td>
                    </tr>`,
			n.NodeID,
			strings.ToUpper(n.OS),
			n.Arch,
			n.ComputeTFLOPs,
			n.BandwidthGBS,
			roleBadge,
		))
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
        .badge { background: #238636; color: white; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
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
        .btn { background: #21262d; border: 1px solid var(--border); color: var(--accent); padding: 6px 14px; border-radius: 6px; text-decoration: none; font-size: 13px; font-weight: 500; }
        .btn:hover { background: #30363d; }
    </style>
    <script>
        setInterval(async () => {
            try {
                const res = await fetch('/api/telemetry');
                const data = await res.json();
                document.getElementById('dispatches').innerText = data.total_dispatches;
                document.getElementById('p50').innerText = data.p50_latency_us.toFixed(1) + ' µs';
                document.getElementById('p95').innerText = data.p95_latency_us.toFixed(1) + ' µs';
                document.getElementById('toks').innerText = data.effective_tok_s.toFixed(1) + ' tok/s';
                document.getElementById('plan').innerText = data.active_plan;
            } catch (e) {}
        }, 1000);

        async function sendPrompt() {
            const input = document.getElementById('prompt-input');
            const out = document.getElementById('output-box');
            const btn = document.getElementById('gen-btn');
            const prompt = input.value.trim();
            if (!prompt) return;

            btn.disabled = true;
            btn.innerText = 'Generating...';
            out.innerText = '';

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
                <button id="gen-btn" onclick="sendPrompt()" class="btn" style="background: #238636; color: white; font-weight: 600; cursor: pointer; border: none;">Generate (SSE Stream)</button>
            </div>
            <div id="output-box" style="background: #0d1117; border: 1px solid var(--border); border-radius: 6px; padding: 14px; min-height: 48px; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 14px; line-height: 1.5; white-space: pre-wrap; color: var(--text-bright);">Ready for inference. Click Generate to stream tokens live.</div>
        </div>

        <div class="section">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div class="section-title" style="margin-bottom: 0;">🌐 go-iroh P2P Cluster Mesh Topology</div>
                <span class="badge" style="background: #238636; font-size: 13px;">🟢 %d Nodes Active (%.0f TFLOPs, %.0f GB/s Cluster)</span>
            </div>
            <div style="font-size: 13px; color: #8b949e; margin-bottom: 14px;">
                E-Graph Role Assignment: <strong style="color: var(--text-bright);">%s</strong>
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
                    </tr>
                </thead>
                <tbody>
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
</body>
</html>`
