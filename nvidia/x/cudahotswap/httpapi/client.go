package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

// Client calls the HTTP service. HTTP is the only wire protocol used by the
// CLI and MCP adapter. A service Failure is returned in Response, not hidden.
type Client struct {
	URL  string
	HTTP *http.Client
}

// NewClient accepts unix:/absolute/socket or tcp:host:port. TCP is plain HTTP;
// use an SSH forward for remote hosts. The default address is application-owned.
func NewClient(address string) (*Client, error) {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	base := ""
	switch {
	case strings.HasPrefix(address, "unix:"):
		path := strings.TrimPrefix(address, "unix:")
		if !strings.HasPrefix(path, "/") {
			return nil, errors.New("unix socket must be absolute")
		}
		transport.Proxy = nil
		transport.DialContext = func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", path)
		}
		base = "http://unix"
	case strings.HasPrefix(address, "tcp:"):
		peer := strings.TrimPrefix(address, "tcp:")
		if _, _, err := net.SplitHostPort(peer); err != nil {
			return nil, err
		}
		base = "http://" + peer
		transport.Proxy = nil
	default:
		return nil, errors.New("address must be unix:path or tcp:host:port")
	}
	return &Client{URL: base, HTTP: &http.Client{Transport: transport}}, nil
}

// Call round-trips a named endpoint. Non200 statuses return both the structured
// response and an error. A failed candidate check is a200 with Failure and Push.
func (c *Client) Call(ctx context.Context, method string, q Request) (Response, error) {
	var out Response
	if c == nil || c.HTTP == nil || ctx == nil || !validName(method) || strings.Contains(method, "/") {
		return out, errors.New("invalid HTTP client call")
	}
	b, err := json.Marshal(q)
	if err != nil {
		return out, err
	}
	r, err := http.NewRequestWithContext(ctx, http.MethodPost, c.URL+"/v1/"+url.PathEscape(method), bytes.NewReader(b))
	if err != nil {
		return out, err
	}
	r.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTP.Do(r)
	if err != nil {
		return out, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, (8<<20)+1))
	if err != nil {
		return out, err
	}
	if len(data) > 8<<20 {
		return out, errors.New("HTTP response exceeds8MiB")
	}
	if err = json.Unmarshal(data, &out); err != nil {
		return out, err
	}
	if resp.StatusCode != http.StatusOK {
		return out, fmt.Errorf("HTTP status %d", resp.StatusCode)
	}
	return out, nil
}

// Events streams NDJSON events until cancellation or a terminal service error.
// The caller owns the returned response body and must close it. Its decoder
// refuses frames over 512 KiB, including JSON escaping of the service's
// 64 KiB retained-value budget.
func (c *Client) Events(ctx context.Context, since uint64) (*EventStream, error) {
	if c == nil || c.HTTP == nil || ctx == nil {
		return nil, errors.New("invalid HTTP event client")
	}
	r, err := http.NewRequestWithContext(ctx, http.MethodGet, c.URL+"/v1/events?since="+strconv.FormatUint(since, 10), nil)
	if err != nil {
		return nil, err
	}
	r.Header.Set("Accept", "application/x-ndjson")
	resp, err := c.HTTP.Do(r)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK || !strings.HasPrefix(resp.Header.Get("Content-Type"), "application/x-ndjson") {
		defer resp.Body.Close()
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 128<<10))
		return nil, fmt.Errorf("event response status %d: %s", resp.StatusCode, b)
	}
	return newEventStream(resp.Body), nil
}
