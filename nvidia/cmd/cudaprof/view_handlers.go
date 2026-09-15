package main

import (
	"encoding/json"
	"github.com/tmc/lib/nvidia/internal/cudaview"
	"net/http"
)

func newPipelineServerHandler(data *cudaview.PipelineViewData) http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		if err := cudaview.Render(w, data); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	})

	mux.HandleFunc("GET /api/pipeline", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(data)
	})

	if len(data.Recompile) > 0 {
		mux.HandleFunc("/api/recompile", newRecompileHandler(data.Recompile, compileNVRTCVariant))
	}

	mux.HandleFunc("GET /api/kernel", func(w http.ResponseWriter, r *http.Request) {
		kName := r.URL.Query().Get("name")
		w.Header().Set("Content-Type", "application/json")
		if tiers, ok := data.KernelTiers[kName]; ok {
			json.NewEncoder(w).Encode(tiers)
		} else {
			json.NewEncoder(w).Encode(data.Tiers)
		}
	})

	mux.HandleFunc("GET /api/inventory", func(w http.ResponseWriter, r *http.Request) {
		type module struct {
			ID      string   `json:"id"`
			Name    string   `json:"name"`
			Kernels []string `json:"kernels"`
		}
		modules := []module{}
		for _, m := range data.Modules {
			modules = append(modules, module{m.ID, m.Name, m.Data.Kernels})
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(modules)
	})
	mux.HandleFunc("GET /api/module", func(w http.ResponseWriter, r *http.Request) {
		id := r.URL.Query().Get("id")
		for _, m := range data.Modules {
			if m.ID == id {
				w.Header().Set("Content-Type", "application/json")
				if err := cudaview.WriteSnapshot(w, m.Data); err != nil {
					http.Error(w, err.Error(), 500)
				}
				return
			}
		}
		http.NotFound(w, r)
	})
	return mux
}
