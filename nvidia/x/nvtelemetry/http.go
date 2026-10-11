package nvtelemetry

import (
	"bytes"
	"fmt"
	"math"
	"mime"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// MetricsHandler serves a bounded, previously published telemetry snapshot.
// Scrapes never query hardware. The zero value returns service unavailable.
// Publication age describes the HTTP snapshot, not sensor freshness.
type MetricsHandler struct {
	mu         sync.RWMutex
	options    ExportOptions
	maxAge     time.Duration
	published  time.Time
	text, open []byte
}

// NewMetricsHandler creates a handler with an explicit publication-age limit.
// It serves Prometheus text by default, or OpenMetrics when options.OpenMetrics
// is set. An Accept header can select either supported format.
func NewMetricsHandler(options ExportOptions, maxAge time.Duration) (*MetricsHandler, error) {
	if _, _, err := exportLimits(options); err != nil {
		return nil, err
	}
	if maxAge <= 0 {
		return nil, fmt.Errorf("metrics HTTP: invalid maximum age")
	}
	return &MetricsHandler{options: options, maxAge: maxAge}, nil
}

// Update validates and copies readings into both exposition formats before
// publishing them atomically. An invalid update invalidates the old snapshot.
func (h *MetricsHandler) Update(readings []Reading) error {
	if h == nil || h.maxAge <= 0 {
		return fmt.Errorf("metrics HTTP: uninitialized handler")
	}
	var text, open bytes.Buffer
	options := h.options
	options.OpenMetrics = false
	err := WriteExposition(&text, readings, options)
	if err == nil {
		options.OpenMetrics = true
		err = WriteExposition(&open, readings, options)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if err != nil {
		h.text, h.open, h.published = nil, nil, time.Time{}
		return err
	}
	h.text, h.open, h.published = text.Bytes(), open.Bytes(), time.Now()
	return nil
}

// Invalidate makes the handler unavailable until the next successful Update.
func (h *MetricsHandler) Invalidate() {
	if h == nil {
		return
	}
	h.mu.Lock()
	h.text, h.open, h.published = nil, nil, time.Time{}
	h.mu.Unlock()
}

// Ready reports whether a published snapshot is within its publication-age limit.
func (h *MetricsHandler) Ready() bool {
	if h == nil {
		return false
	}
	h.mu.RLock()
	defer h.mu.RUnlock()
	return !h.published.IsZero() && time.Since(h.published) <= h.maxAge
}

// ServeHTTP serves GET and HEAD scrapes. Unsupported formats return 406;
// unavailable or expired snapshots return 503 without stale measurements.
func (h *MetricsHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("Vary", "Accept")
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		w.Header().Set("Allow", "GET, HEAD")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if h == nil {
		http.Error(w, "telemetry unavailable", http.StatusServiceUnavailable)
		return
	}
	open, err := metricsFormat(strings.Join(r.Header.Values("Accept"), ","), h.options.OpenMetrics)
	if err != nil {
		http.Error(w, "unsupported exposition format", http.StatusNotAcceptable)
		return
	}
	h.mu.RLock()
	if h.published.IsZero() || time.Since(h.published) > h.maxAge {
		h.mu.RUnlock()
		http.Error(w, "telemetry unavailable", http.StatusServiceUnavailable)
		return
	}
	data := h.text
	contentType := "text/plain; version=0.0.4; charset=utf-8"
	if open {
		data = h.open
		contentType = "application/openmetrics-text; version=1.0.0; charset=utf-8"
	}
	// Update replaces slices; it never modifies a published payload.
	h.mu.RUnlock()
	w.Header().Set("Content-Type", contentType)
	w.Header().Set("Content-Length", strconv.Itoa(len(data)))
	if r.Method == http.MethodGet {
		_, _ = w.Write(data)
	}
}

func metricsFormat(accept string, defaultOpen bool) (bool, error) {
	if accept == "" {
		return defaultOpen, nil
	}
	if len(accept) > 4096 {
		return false, fmt.Errorf("accept header too large")
	}
	parts := strings.Split(accept, ",")
	if len(parts) > 64 {
		return false, fmt.Errorf("too many media ranges")
	}
	quality := [2]float64{-1, -1}
	specificity := [2]int{-1, -1}
	for _, part := range parts {
		media, params, err := mime.ParseMediaType(strings.TrimSpace(part))
		if err != nil {
			return false, err
		}
		q := 1.0
		if value, ok := params["q"]; ok {
			if !metricsQuality(value) {
				return false, fmt.Errorf("invalid quality")
			}
			q, err = strconv.ParseFloat(value, 64)
			if err != nil || math.IsNaN(q) || math.IsInf(q, 0) || q < 0 || q > 1 {
				return false, fmt.Errorf("invalid quality")
			}
		}
		for i, name := range []string{"text/plain", "application/openmetrics-text"} {
			spec := -1
			switch {
			case media == name:
				spec = 2
			case media == strings.SplitN(name, "/", 2)[0]+"/*":
				spec = 1
			case media == "*/*":
				spec = 0
			}
			if spec < 0 {
				continue
			}
			version := "0.0.4"
			if i == 1 {
				version = "1.0.0"
			}
			if v, ok := params["version"]; ok && v != version {
				continue
			}
			if charset, ok := params["charset"]; ok && !strings.EqualFold(charset, "utf-8") {
				continue
			}
			if spec > specificity[i] || spec == specificity[i] && q > quality[i] {
				specificity[i], quality[i] = spec, q
			}
		}
	}
	if quality[0] <= 0 && quality[1] <= 0 {
		return false, fmt.Errorf("no supported format")
	}
	return quality[1] > quality[0] || quality[1] == quality[0] && defaultOpen, nil
}

func metricsQuality(s string) bool {
	if len(s) == 0 || len(s) > 5 || s[0] != '0' && s[0] != '1' {
		return false
	}
	if len(s) > 1 && s[1] != '.' {
		return false
	}
	for i := 2; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' || s[0] == '1' && s[i] != '0' {
			return false
		}
	}
	return true
}
