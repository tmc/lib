package nvtelemetry

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"time"
)

// ErrDeliveryFull means the outstanding batch or byte budget is exhausted.
var ErrDeliveryFull = errors.New("telemetry delivery queue full")

// ErrDeliveryClosed means shutdown has begun and new batches are refused.
var ErrDeliveryClosed = errors.New("telemetry delivery closed")

// ErrDeliveryIncomplete means at least one admitted batch was not fully accepted.
var ErrDeliveryIncomplete = errors.New("telemetry delivery incomplete")

// DeliveryConfig sets explicit HTTP export and lifetime bounds. Endpoint is an
// absolute HTTP(S) OTLP metrics URL. Client supplies TLS and authentication
// behavior; Headers may supply credentials and are cloned. No endpoint, client
// or background exporter is installed by default.
// Redirects are refused so success cannot be attributed to a different endpoint.
//
// An item is one already encoded ExportMetricsServiceRequest or
// ExportLogsServiceRequest, not a reading.
// Outstanding budgets include the batch being sent. MaxBatchDataPoints bounds
// gauge/sum/histogram/summary points or log records independently of the byte
// limit. Delivery preserves item order but retry can duplicate a previously
// accepted request.
type DeliveryConfig struct {
	Endpoint           string
	Client             *http.Client
	Headers            http.Header
	MaxQueuedBatches   int
	MaxQueuedBytes     int
	MaxBatchBytes      int
	MaxBatchDataPoints int
	MaxResponseBytes   int64
	MaxAttempts        int
	InitialBackoff     time.Duration
	MaxBackoff         time.Duration
	BatchTimeout       time.Duration
	RequestTimeout     time.Duration
	ShutdownTimeout    time.Duration
}

// DeliveryStats records cumulative delivery outcomes. SentBatches includes
// partial-success responses, but SucceededBatches only includes full acceptance.
// DroppedBatches counts budget-refused Enqueue calls; FailedBatches counts admitted
// items not fully delivered. Pending budgets include the in-flight item.
type DeliveryStats struct {
	EnqueuedBatches    uint64
	SucceededBatches   uint64
	SentBatches        uint64
	FailedBatches      uint64
	DroppedBatches     uint64
	RetryAttempts      uint64
	PartialBatches     uint64
	RejectedDataPoints uint64
	RejectedLogRecords uint64
	AuthFailures       uint64
	RejectedBatches    uint64
	TransportFailures  uint64
	OutageResponses    uint64
	InvalidResponses   uint64
	PendingBatches     int
	PendingBytes       int
}

// Delivery sends bounded OTLP HTTP JSON batches on one worker. The zero value
// is not usable. Call Close to stop its worker and report incomplete delivery.
// A failing exporter never logs request headers, payloads, response bodies or
// transport error text. Statistics describe acceptance, not backend storage.
type Delivery struct {
	mu              sync.Mutex
	config          DeliveryConfig
	isLogs          bool
	queue           [][]byte
	stats           DeliveryStats
	closing         bool
	terminal        error
	wake            chan struct{}
	done            chan struct{}
	shutdownExpired chan struct{}
	cancel          context.CancelFunc
	closeOnce       sync.Once
}

func validateDeliveryConfig(c DeliveryConfig) error {
	u, err := url.Parse(c.Endpoint)
	if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return fmt.Errorf("telemetry delivery: invalid endpoint")
	}
	if c.Client == nil || c.MaxQueuedBatches <= 0 || c.MaxQueuedBytes <= 0 || c.MaxBatchBytes <= 0 || c.MaxBatchBytes > c.MaxQueuedBytes || c.MaxBatchDataPoints <= 0 || c.MaxResponseBytes <= 0 || c.MaxAttempts <= 0 || c.MaxAttempts > 8 || c.BatchTimeout <= 0 || c.RequestTimeout <= 0 || c.ShutdownTimeout <= 0 || c.InitialBackoff < 0 || c.MaxBackoff < c.InitialBackoff || (c.MaxAttempts > 1 && c.InitialBackoff == 0) {
		return fmt.Errorf("telemetry delivery: invalid client or bounds")
	}
	if c.MaxQueuedBatches > 4096 || c.MaxQueuedBytes > 64<<20 || c.MaxBatchBytes > 4<<20 || c.MaxBatchDataPoints > 4096 || c.MaxResponseBytes > 1<<20 {
		return fmt.Errorf("telemetry delivery: limits exceed bounded exporter capacity")
	}
	return nil
}

// NewDelivery starts a worker for the caller-supplied endpoint. It performs no
// network operation until Enqueue admits a batch. All limits must be explicit.
func NewDelivery(c DeliveryConfig) (*Delivery, error) {
	if err := validateDeliveryConfig(c); err != nil {
		return nil, err
	}
	c.Headers = c.Headers.Clone()
	if c.Headers == nil {
		c.Headers = make(http.Header)
	}
	client := *c.Client
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	c.Client = &client
	ctx, cancel := context.WithCancel(context.Background())
	d := &Delivery{config: c, isLogs: false, wake: make(chan struct{}, 1), done: make(chan struct{}), shutdownExpired: make(chan struct{}), cancel: cancel}
	go d.run(ctx)
	return d, nil
}

// NewLogsDelivery starts an OTLP HTTP JSON logs export worker for the caller-supplied endpoint.
// It shares bounded queueing, worker lifecycle, retry hysteresis, and shutdown semantics
// with NewDelivery while validating ExportLogsServiceRequest batches and reporting rejected
// log records separately.
func NewLogsDelivery(c DeliveryConfig) (*Delivery, error) {
	if err := validateDeliveryConfig(c); err != nil {
		return nil, err
	}
	c.Headers = c.Headers.Clone()
	if c.Headers == nil {
		c.Headers = make(http.Header)
	}
	client := *c.Client
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	c.Client = &client
	ctx, cancel := context.WithCancel(context.Background())
	d := &Delivery{config: c, isLogs: true, wake: make(chan struct{}, 1), done: make(chan struct{}), shutdownExpired: make(chan struct{}), cancel: cancel}
	go d.run(ctx)
	return d, nil
}

// Enqueue copies one encoded OTLP request. It fails immediately when
// full; it never waits for the network. Caller cancellation refuses admission.
func (d *Delivery) Enqueue(ctx context.Context, payload []byte) error {
	if ctx == nil {
		return fmt.Errorf("telemetry delivery: nil context")
	}
	if d == nil || d.done == nil {
		return ErrDeliveryClosed
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(payload) == 0 || len(payload) > d.config.MaxBatchBytes {
		return fmt.Errorf("telemetry delivery: invalid batch size")
	}
	var err error
	if d.isLogs {
		err = validateLogsDeliveryBatch(payload, d.config.MaxBatchDataPoints)
	} else {
		err = validateDeliveryBatch(payload, d.config.MaxBatchDataPoints)
	}
	if err != nil {
		return err
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closing {
		return ErrDeliveryClosed
	}
	if d.stats.PendingBatches >= d.config.MaxQueuedBatches || len(payload) > d.config.MaxQueuedBytes-d.stats.PendingBytes {
		d.stats.DroppedBatches++
		return ErrDeliveryFull
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	d.queue = append(d.queue, append([]byte(nil), payload...))
	d.stats.PendingBatches++
	d.stats.PendingBytes += len(payload)
	d.stats.EnqueuedBatches++
	d.signal()
	return nil
}

// Stats returns a coherent copy of exporter counters without payload metadata.
func (d *Delivery) Stats() DeliveryStats {
	if d == nil {
		return DeliveryStats{}
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.stats
}

// Close stops admission and drains within ShutdownTimeout or the first caller's
// context deadline. A timed-out or partially rejected drain returns
// ErrDeliveryIncomplete. Repeated Close calls return the stored outcome.
// HTTP transports must honor request context cancellation; a custom transport
// that ignores it cannot be forcibly terminated by this package.
func (d *Delivery) Close(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("telemetry delivery: nil context")
	}
	if d == nil || d.done == nil {
		return ErrDeliveryClosed
	}
	d.closeOnce.Do(func() {
		d.mu.Lock()
		d.closing = true
		d.mu.Unlock()
		d.signal()
		go func() {
			timer := time.NewTimer(d.config.ShutdownTimeout)
			defer timer.Stop()
			select {
			case <-d.done:
				return
			case <-ctx.Done():
			case <-timer.C:
			}
			d.mu.Lock()
			d.terminal = ErrDeliveryIncomplete
			d.mu.Unlock()
			d.cancel()
			close(d.shutdownExpired)
		}()
	})
	select {
	case <-d.done:
		d.mu.Lock()
		defer d.mu.Unlock()
		return d.terminal
	case <-ctx.Done():
		d.cancel()
		return errors.Join(ErrDeliveryIncomplete, ctx.Err())
	case <-d.shutdownExpired:
		return ErrDeliveryIncomplete
	}
}

func (d *Delivery) signal() {
	select {
	case d.wake <- struct{}{}:
	default:
	}
}

func (d *Delivery) run(ctx context.Context) {
	defer close(d.done)
	defer d.cancel()
	for {
		d.mu.Lock()
		if ctx.Err() != nil {
			d.stats.FailedBatches += uint64(len(d.queue))
			d.queue = nil
			if d.stats.PendingBatches > 0 {
				d.terminal = ErrDeliveryIncomplete
			}
			d.stats.PendingBatches = 0
			d.stats.PendingBytes = 0
			d.mu.Unlock()
			return
		}
		if len(d.queue) == 0 {
			closing := d.closing
			d.mu.Unlock()
			if closing {
				return
			}
			select {
			case <-ctx.Done():
			case <-d.wake:
			}
			continue
		}
		payload := d.queue[0]
		d.queue[0] = nil
		d.queue = d.queue[1:]
		d.mu.Unlock()
		ok := d.send(ctx, payload)
		d.mu.Lock()
		d.stats.PendingBatches--
		d.stats.PendingBytes -= len(payload)
		if ok {
			d.stats.SucceededBatches++
		} else {
			d.stats.FailedBatches++
			d.terminal = ErrDeliveryIncomplete
		}
		d.mu.Unlock()
	}
}

func (d *Delivery) send(parent context.Context, payload []byte) bool {
	ctx, cancel := context.WithTimeout(parent, d.config.BatchTimeout)
	defer cancel()
	backoff := d.config.InitialBackoff
	for attempt := 0; attempt < d.config.MaxAttempts; attempt++ {
		if ctx.Err() != nil {
			return false
		}
		if attempt > 0 {
			d.mu.Lock()
			d.stats.RetryAttempts++
			d.mu.Unlock()
		}
		requestCtx, requestCancel := context.WithTimeout(ctx, d.config.RequestTimeout)
		req, err := http.NewRequestWithContext(requestCtx, http.MethodPost, d.config.Endpoint, bytes.NewReader(payload))
		if err != nil {
			requestCancel()
			return false
		}
		req.Header = d.config.Headers.Clone()
		req.Header.Set("Content-Type", "application/json")
		response, err := d.config.Client.Do(req)
		retry := false
		retryAfter := time.Duration(0)
		if err != nil {
			requestCancel()
			d.mu.Lock()
			d.stats.TransportFailures++
			d.mu.Unlock()
			retry = true
		} else {
			body, readErr := io.ReadAll(io.LimitReader(response.Body, d.config.MaxResponseBytes+1))
			closeErr := response.Body.Close()
			requestCancel()
			status := response.StatusCode
			retryAfter = deliveryRetryAfter(response.Header.Get("Retry-After"), time.Now())
			if status == http.StatusOK {
				d.mu.Lock()
				d.stats.SentBatches++
				d.mu.Unlock()
				if readErr != nil || closeErr != nil || int64(len(body)) > d.config.MaxResponseBytes {
					d.mu.Lock()
					d.stats.InvalidResponses++
					d.mu.Unlock()
					return false
				}
				partial, rejected, valid := deliveryResponse(body, d.isLogs)
				if !valid {
					d.mu.Lock()
					d.stats.InvalidResponses++
					d.mu.Unlock()
					return false
				}
				if partial {
					d.mu.Lock()
					d.stats.PartialBatches++
					if d.isLogs {
						d.stats.RejectedLogRecords += rejected
					} else {
						d.stats.RejectedDataPoints += rejected
					}
					d.mu.Unlock()
					return false
				}
				return true
			}
			d.mu.Lock()
			if status == http.StatusUnauthorized || status == http.StatusForbidden {
				d.stats.AuthFailures++
			} else if status != 429 && status != 502 && status != 503 && status != 504 {
				d.stats.RejectedBatches++
			} else {
				d.stats.OutageResponses++
			}
			d.mu.Unlock()
			retry = status == 429 || status == 502 || status == 503 || status == 504
		}
		if !retry || attempt+1 >= d.config.MaxAttempts {
			return false
		}
		wait := backoff
		if backoff > 1 {
			wait = backoff/2 + time.Duration(rand.Int64N(int64(backoff-backoff/2)))
		}
		if retryAfter > wait {
			wait = retryAfter
		}
		timer := time.NewTimer(wait)
		select {
		case <-ctx.Done():
			timer.Stop()
			return false
		case <-timer.C:
		}
		if backoff < d.config.MaxBackoff {
			if backoff > d.config.MaxBackoff/2 {
				backoff = d.config.MaxBackoff
			} else {
				backoff *= 2
			}
		}
	}
	return false
}

func deliveryResponse(body []byte, isLogs bool) (bool, uint64, bool) {
	var response struct {
		Partial *struct {
			RejectedDataPoints json.RawMessage `json:"rejectedDataPoints"`
			RejectedLogRecords json.RawMessage `json:"rejectedLogRecords"`
			Message            string          `json:"errorMessage"`
		} `json:"partialSuccess"`
	}
	if len(body) == 0 {
		return false, 0, false
	}
	if err := readDeliveryJSON(body, &response); err != nil {
		return false, 0, false
	}
	if response.Partial == nil {
		return false, 0, true
	}
	if isLogs {
		if len(response.Partial.RejectedDataPoints) > 0 && string(bytes.TrimSpace(response.Partial.RejectedDataPoints)) != "null" {
			return false, 0, false
		}
	} else {
		if len(response.Partial.RejectedLogRecords) > 0 && string(bytes.TrimSpace(response.Partial.RejectedLogRecords)) != "null" {
			return false, 0, false
		}
	}
	raw := response.Partial.RejectedDataPoints
	if isLogs {
		raw = response.Partial.RejectedLogRecords
	}
	n := uint64(0)
	if len(raw) > 0 && string(raw) != "null" {
		text := string(raw)
		if len(text) > 0 && text[0] == '"' {
			if err := json.Unmarshal(raw, &text); err != nil {
				return false, 0, false
			}
		}
		v, err := strconv.ParseInt(text, 10, 64)
		if err != nil || v < 0 {
			return false, 0, false
		}
		n = uint64(v)
	}
	return n != 0 || response.Partial.Message != "", n, true
}

func readDeliveryJSON(b []byte, v any) error {
	trimmed := bytes.TrimSpace(b)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return fmt.Errorf("telemetry delivery: expected json object")
	}
	if err := checkEntityJSON(json.NewDecoder(bytes.NewReader(b)), 0); err != nil {
		return err
	}
	d := json.NewDecoder(bytes.NewReader(b))
	if err := d.Decode(v); err != nil {
		return err
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return fmt.Errorf("telemetry delivery: trailing json")
	}
	return nil
}

func deliveryRetryAfter(header string, now time.Time) time.Duration {
	if header == "" {
		return 0
	}
	if seconds, err := strconv.ParseUint(header, 10, 64); err == nil {
		if seconds > uint64((1<<63-1)/time.Second) {
			return time.Duration(1<<63 - 1)
		}
		return time.Duration(seconds) * time.Second
	}
	if date, err := http.ParseTime(header); err == nil && date.After(now) {
		return date.Sub(now)
	}
	return 0
}

func validateDeliveryBatch(b []byte, limit int) error {
	// Preserve protocol bytes; count only the declared data-point arrays.
	var request struct {
		Resources []struct {
			Resource json.RawMessage `json:"resource"`
			Schema   string          `json:"schemaUrl"`
			Scopes   []struct {
				Scope   json.RawMessage `json:"scope"`
				Schema  string          `json:"schemaUrl"`
				Metrics []struct {
					Name                 string          `json:"name"`
					Description          string          `json:"description"`
					Unit                 string          `json:"unit"`
					Metadata             json.RawMessage `json:"metadata"`
					Gauge                *deliveryPoints `json:"gauge"`
					Sum                  *deliveryPoints `json:"sum"`
					Histogram            *deliveryPoints `json:"histogram"`
					ExponentialHistogram *deliveryPoints `json:"exponentialHistogram"`
					Summary              *deliveryPoints `json:"summary"`
				} `json:"metrics"`
			} `json:"scopeMetrics"`
		} `json:"resourceMetrics"`
		ResourceLogs json.RawMessage `json:"resourceLogs"`
	}
	if err := readDeliveryJSON(b, &request); err != nil {
		return fmt.Errorf("telemetry delivery: invalid metrics request")
	}
	if len(request.ResourceLogs) > 0 && string(bytes.TrimSpace(request.ResourceLogs)) != "null" {
		return fmt.Errorf("telemetry delivery: unexpected resourceLogs in metrics request")
	}
	n := 0
	for _, resource := range request.Resources {
		for _, scope := range resource.Scopes {
			for _, metric := range scope.Metrics {
				instruments := 0
				for _, points := range []*deliveryPoints{metric.Gauge, metric.Sum, metric.Histogram, metric.ExponentialHistogram, metric.Summary} {
					if points != nil {
						instruments++
						if len(points.Points) > limit-n {
							return fmt.Errorf("telemetry delivery: too many data points")
						}
						for _, point := range points.Points {
							value := bytes.TrimSpace(point)
							if len(value) == 0 || value[0] != '{' {
								return fmt.Errorf("telemetry delivery: invalid data point object")
							}
						}
						n += len(points.Points)
					}
				}
				if metric.Name == "" || instruments != 1 {
					return fmt.Errorf("telemetry delivery: invalid metric instrument")
				}
			}
		}
	}
	if n == 0 {
		return fmt.Errorf("telemetry delivery: empty metrics request")
	}
	return nil
}

func validateLogsDeliveryBatch(b []byte, limit int) error {
	var request struct {
		ResourceLogs []struct {
			Resource json.RawMessage `json:"resource"`
			Schema   string          `json:"schemaUrl"`
			Scopes   []struct {
				Scope      json.RawMessage   `json:"scope"`
				Schema     string            `json:"schemaUrl"`
				LogRecords []json.RawMessage `json:"logRecords"`
			} `json:"scopeLogs"`
		} `json:"resourceLogs"`
		ResourceMetrics json.RawMessage `json:"resourceMetrics"`
	}
	if err := readDeliveryJSON(b, &request); err != nil {
		return fmt.Errorf("telemetry delivery: invalid logs request")
	}
	if len(request.ResourceMetrics) > 0 && string(bytes.TrimSpace(request.ResourceMetrics)) != "null" {
		return fmt.Errorf("telemetry delivery: unexpected resourceMetrics in logs request")
	}
	n := 0
	for _, resource := range request.ResourceLogs {
		for _, scope := range resource.Scopes {
			if len(scope.LogRecords) > limit-n {
				return fmt.Errorf("telemetry delivery: too many log records")
			}
			for _, record := range scope.LogRecords {
				value := bytes.TrimSpace(record)
				if len(value) == 0 || value[0] != '{' {
					return fmt.Errorf("telemetry delivery: invalid log record object")
				}
			}
			n += len(scope.LogRecords)
		}
	}
	if n == 0 {
		return fmt.Errorf("telemetry delivery: empty logs request")
	}
	return nil
}

type deliveryPoints struct {
	Points      []json.RawMessage `json:"dataPoints"`
	Temporality json.RawMessage   `json:"aggregationTemporality"`
	Monotonic   json.RawMessage   `json:"isMonotonic"`
}
