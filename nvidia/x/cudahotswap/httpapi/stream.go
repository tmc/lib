package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"

	"github.com/tmc/lib/nvidia/x/cudahotswap"
)

// Frame contains one report event, watched push result, or terminal failure.
type Frame struct {
	Event  *cudahotswap.Event `json:"event,omitempty"`
	Result *Response          `json:"result,omitempty"`
}

func (h *Handler) streamSlot() bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.streams >= 32 {
		return false
	}
	h.streams++
	return true
}
func (h *Handler) releaseStream() { h.mu.Lock(); h.streams--; h.mu.Unlock() }

func streamWriter(w http.ResponseWriter, r *http.Request) (func(Frame) error, error) {
	f, ok := w.(http.Flusher)
	if !ok {
		return nil, errors.New("stream requires flushing response writer")
	}
	ndjson := r.Header.Get("Accept") == "application/x-ndjson"
	if ndjson {
		w.Header().Set("Content-Type", "application/x-ndjson")
	} else {
		w.Header().Set("Content-Type", "text/event-stream")
	}
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	f.Flush()
	return func(frame Frame) error {
		b, err := json.Marshal(frame)
		if err != nil {
			return err
		}
		if ndjson {
			_, err = fmt.Fprintf(w, "%s\n", b)
		} else {
			if frame.Event != nil {
				_, err = fmt.Fprintf(w, "id: %d\n", frame.Event.Sequence)
			}
			if err == nil {
				_, err = fmt.Fprintf(w, "data: %s\n\n", b)
			}
		}
		if err == nil {
			f.Flush()
		}
		return err
	}, nil
}

func (h *Handler) events(ctx context.Context, w http.ResponseWriter, r *http.Request) {
	since := uint64(0)
	if q := r.URL.Query().Get("since"); q != "" {
		var err error
		since, err = strconv.ParseUint(q, 10, 64)
		if err != nil {
			malformed(w, errors.New("invalid report cursor"))
			return
		}
	}
	if !h.streamSlot() {
		malformed(w, errors.New("stream limit exceeded"))
		return
	}
	defer h.releaseStream()
	sub, err := h.service.Subscribe(ctx, since)
	if err != nil {
		status := http.StatusOK
		if errors.Is(err, cudahotswap.ErrPoisoned) {
			status = http.StatusServiceUnavailable
		}
		writeResponse(w, status, Response{Failure: failure(err)})
		return
	}
	if sub == nil {
		writeResponse(w, http.StatusOK, Response{Failure: failure(errors.New("service returned no subscription"))})
		return
	}
	defer sub.Close()
	write, err := streamWriter(w, r)
	if err != nil {
		return
	}
	for {
		select {
		case <-ctx.Done():
			return
		case event, ok := <-sub.Events():
			if !ok {
				if err := sub.Err(); err != nil && !errors.Is(err, context.Canceled) {
					write(Frame{Result: &Response{Failure: failure(err)}})
				}
				return
			}
			if write(Frame{Event: &event}) != nil {
				return
			}
		}
	}
}

func (h *Handler) watch(ctx context.Context, w http.ResponseWriter, q Request) {
	if !validName(q.Label) {
		malformed(w, errors.New("caller label required"))
		return
	}
	if !h.streamSlot() {
		malformed(w, errors.New("stream limit exceeded"))
		return
	}
	defer h.releaseStream()
	r := &http.Request{Header: make(http.Header)}
	r.Header.Set("Accept", "application/x-ndjson")
	write, err := streamWriter(w, r)
	if err != nil {
		return
	}
	err = h.service.Watch(ctx, q.Watch, func(_ string, p *cudahotswap.PushResult, e error) error {
		res := Response{RequestLabel: q.Label, Push: p, Failure: failure(e)}
		if p != nil && e != nil {
			res.Failure.Outcome = p.Report.Outcome
		}
		return write(Frame{Result: &res})
	})
	if err != nil && !errors.Is(err, context.Canceled) {
		write(Frame{Result: &Response{Failure: failure(err)}})
	}
}
