package nvtelemetry

import (
	"bytes"
	"context"
	"errors"
	"net/http"
)

// FetchManaged reads a bounded caller-selected Prometheus/OpenMetrics endpoint
// using explicit canonical mappings. It records local query bounds with clock;
// these bounds do not establish sensor time, freshness or clock calibration.
// c's query bounds and TimestampUnix must be zero: they are acquired here.
// GET is the only supported method. The client and request are not modified.
// Redirects are refused, and errors omit URLs, credentials and response bodies.
func FetchManaged(ctx context.Context, client *http.Client, request *http.Request, c ManagedImport, clock Clock, maxBytes int64) (Sample, error) {
	if ctx == nil || clock == nil || request == nil || request.Method != http.MethodGet || request.Body != nil && request.Body != http.NoBody {
		return Sample{}, errors.New("managed endpoint requires a clock and GET request")
	}
	if c.QueryStartNanos != 0 || c.QueryEndNanos != 0 || c.TimestampUnix != 0 {
		return Sample{}, errors.New("managed endpoint query timing must be acquired locally")
	}
	if err := validateManagedEndpoint(ctx, c); err != nil {
		return Sample{}, err
	}
	start := clock.NowMonotonic()
	raw, err := fetchHTTPBytes(ctx, client, request, maxBytes)
	end := clock.NowMonotonic()
	if err != nil {
		return Sample{}, err
	}
	if start < 0 || end < start {
		return Sample{}, errors.New("managed endpoint clock order invalid")
	}
	c.QueryStartNanos, c.QueryEndNanos, c.TimestampUnix = start, end, clock.NowUnix()
	sample, err := ReadManaged(ctx, bytes.NewReader(raw), c)
	if err != nil {
		return Sample{}, errors.New("invalid managed endpoint response format")
	}
	return sample, nil
}

func validateManagedEndpoint(ctx context.Context, c ManagedImport) error {
	if c.QueryStartNanos != 0 || c.QueryEndNanos != 0 || c.TimestampUnix != 0 {
		return errors.New("managed endpoint query timing must be acquired locally")
	}
	for _, field := range c.Fields {
		source := &ReadingSource{Mapping: field, MappingDigest: managedMappingDigest(field), Schema: "nvtelemetry.managed.v1", SourceID: c.SourceID, Producer: c.Producer, ProducerVersion: c.ProducerVersion, NativeOrigin: c.NativeOrigin, FieldID: field.Metric, RawValue: "0"}
		if err := ValidateReadingSource(source); err != nil {
			return errors.New("invalid managed endpoint source or mappings")
		}
	}
	// Only validation uses this placeholder, never returned observations.
	c.TimestampUnix = 1
	empty := ""
	if c.Exposition.Format == OpenMetrics100 {
		empty = "# EOF\n"
	}
	if _, err := ReadManaged(ctx, bytes.NewBufferString(empty), c); err != nil {
		return errors.New("invalid managed endpoint source, mappings or limits")
	}
	return nil
}
