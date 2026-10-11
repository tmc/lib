package nvtelemetry

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

// fetchHTTPBytes reads one bounded caller-selected response. It never reports
// URLs, credentials, response bodies or transport error text in errors.
func fetchHTTPBytes(ctx context.Context, client *http.Client, request *http.Request, maxBytes int64) ([]byte, error) {
	if ctx == nil || client == nil || request == nil || request.URL == nil {
		return nil, errors.New("missing endpoint context, client or request")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if maxBytes <= 0 || maxBytes > 64<<20 {
		return nil, errors.New("invalid endpoint byte limit")
	}
	if (request.URL.Scheme != "http" && request.URL.Scheme != "https") || request.URL.Host == "" || request.URL.User != nil {
		return nil, errors.New("invalid HTTP endpoint")
	}
	copyClient := *client
	if copyClient.Timeout <= 0 || copyClient.Timeout > 30*time.Second {
		copyClient.Timeout = 30 * time.Second
	}
	copyClient.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	response, err := copyClient.Do(request.Clone(ctx))
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, errors.New("endpoint request failed")
	}
	if response.Body == nil {
		return nil, errors.New("missing endpoint response body")
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("endpoint HTTP status %d", response.StatusCode)
	}
	if response.ContentLength > maxBytes {
		return nil, errors.New("endpoint response exceeds byte limit")
	}
	raw, err := io.ReadAll(io.LimitReader(workloadContextReader{ctx, response.Body}, maxBytes+1))
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, errors.New("read endpoint response failed")
	}
	if int64(len(raw)) > maxBytes {
		return nil, errors.New("endpoint response exceeds byte limit")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return raw, nil
}
