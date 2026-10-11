package hostmetrics

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"strings"
	"time"
	"unicode"
)

// RedfishHTTPConfig supplies explicit connection and bounds settings for
// fetching DMTF Redfish artifacts from a caller-selected BMC endpoint.
// No credentials or endpoints are discovered automatically.
// Live physical BMC qualification requires physical BMC endpoint hardware and
// authenticated credentials which remain an external qualification gate.
type RedfishHTTPConfig struct {
	BaseURL              string        // Base BMC URL (e.g. "https://bmc.example.internal"); scheme must be http or https, no query/user/fragment
	Client               *http.Client  // Caller-supplied HTTP client (must be non-nil)
	Headers              http.Header   // Cloned headers for authentication (e.g. X-Auth-Token, Authorization)
	MaxResponseBytes     int64         // Maximum bytes to read per response (1 <= MaxResponseBytes <= 1<<20)
	RequestTimeout       time.Duration // Per-request timeout (1ms <= RequestTimeout <= 60s)
	MaxCollectionMembers int           // Maximum members admitted in a collection (1 <= MaxCollectionMembers <= 1024)
	MaxPages             int           // Maximum pagination pages admitted (1 <= MaxPages <= 32)
}

func validateRedfishHTTPConfig(c RedfishHTTPConfig) error {
	if c.Client == nil {
		return fmt.Errorf("redfish http: nil client")
	}
	u, err := url.Parse(c.BaseURL)
	if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return fmt.Errorf("redfish http: invalid base URL")
	}
	if c.MaxResponseBytes <= 0 || c.MaxResponseBytes > 1<<20 {
		return fmt.Errorf("redfish http: invalid response byte limit")
	}
	if c.RequestTimeout < time.Millisecond || c.RequestTimeout > 60*time.Second {
		return fmt.Errorf("redfish http: request timeout must be between 1ms and 60s")
	}
	if c.MaxCollectionMembers <= 0 || c.MaxCollectionMembers > 1024 {
		return fmt.Errorf("redfish http: invalid collection member limit")
	}
	if c.MaxPages <= 0 || c.MaxPages > 32 {
		return fmt.Errorf("redfish http: invalid collection page limit")
	}
	if len(c.Headers) > 32 {
		return fmt.Errorf("redfish http: header entries limit exceeded (max 32)")
	}
	for k, vs := range c.Headers {
		if k == "" || len(k) > 128 {
			return fmt.Errorf("redfish http: invalid header key length")
		}
		for _, r := range k {
			if r < 33 || r > 126 || r == ':' {
				return fmt.Errorf("redfish http: invalid characters in header key")
			}
		}
		if len(vs) > 16 {
			return fmt.Errorf("redfish http: too many values for header key")
		}
		for _, v := range vs {
			if len(v) > 4096 {
				return fmt.Errorf("redfish http: header value exceeds 4096 bytes")
			}
			for _, r := range v {
				if (r < 32 && r != '\t') || r == 127 {
					return fmt.Errorf("redfish http: control characters in header value")
				}
			}
		}
	}
	return nil
}

// RedfishHTTPClient fetches caller-selected DMTF Redfish resources from a BMC
// endpoint using a caller-configured HTTP client and explicit bounds.
// The zero value is not usable. Construct using NewRedfishHTTPClient.
type RedfishHTTPClient struct {
	baseURL    string
	baseParsed *url.URL
	client     *http.Client
	headers    http.Header
	config     RedfishHTTPConfig
}

// NewRedfishHTTPClient creates a client for fetching Redfish artifacts from an
// explicit base BMC URL. Redirects are refused to prevent credential leakage
// to unintended endpoints.
func NewRedfishHTTPClient(c RedfishHTTPConfig) (*RedfishHTTPClient, error) {
	if err := validateRedfishHTTPConfig(c); err != nil {
		return nil, err
	}
	u, err := url.Parse(c.BaseURL)
	if err != nil {
		return nil, fmt.Errorf("redfish http: invalid base URL")
	}
	client := *c.Client
	// Refuse redirects so authentication headers or requests are never sent
	// to another destination or scheme.
	client.CheckRedirect = func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}
	return &RedfishHTTPClient{
		baseURL:    c.BaseURL,
		baseParsed: u,
		client:     &client,
		headers:    c.Headers.Clone(),
		config:     c,
	}, nil
}

func (c *RedfishHTTPClient) resolveURI(uri string) (string, string, error) {
	if uri == "" {
		return "", "", fmt.Errorf("redfish http: empty uri")
	}
	// Reject fragments, backslashes, encoded backslashes, encoded slashes, and encoded dots.
	if strings.Contains(uri, "#") {
		return "", "", fmt.Errorf("redfish http: fragments refused")
	}
	lower := strings.ToLower(uri)
	if strings.Contains(uri, "\\") || strings.Contains(lower, "%5c") {
		return "", "", fmt.Errorf("redfish http: backslashes refused")
	}
	if strings.Contains(lower, "%2f") {
		return "", "", fmt.Errorf("redfish http: encoded slashes refused")
	}
	if strings.Contains(lower, "%2e") {
		return "", "", fmt.Errorf("redfish http: encoded dots refused")
	}

	u, err := url.Parse(uri)
	if err != nil || u.User != nil {
		return "", "", fmt.Errorf("redfish http: invalid uri")
	}

	// Validate origin if absolute URI was provided.
	if u.IsAbs() || u.Host != "" {
		if !strings.EqualFold(u.Scheme, c.baseParsed.Scheme) || !strings.EqualFold(u.Host, c.baseParsed.Host) {
			return "", "", fmt.Errorf("redfish http: cross-origin uri refused")
		}
	}

	if strings.Contains(u.Path, "%") {
		return "", "", fmt.Errorf("redfish http: nested path escapes refused")
	}
	rawPath := u.Path
	if rawPath == "" {
		return "", "", fmt.Errorf("redfish http: empty uri path")
	}

	// Reject dot segments that could escape the intended namespace.
	padded := "/" + rawPath + "/"
	if strings.Contains(padded, "/./") || strings.Contains(padded, "/../") || strings.HasSuffix(rawPath, "/.") || strings.HasSuffix(rawPath, "/..") {
		return "", "", fmt.Errorf("redfish http: dot segments refused")
	}

	// Path must be canonical (no duplicate slashes, no relative elements).
	cleaned := path.Clean(rawPath)
	if cleaned != rawPath {
		return "", "", fmt.Errorf("redfish http: noncanonical path refused")
	}
	if !strings.HasPrefix(cleaned, "/redfish/v1/") {
		return "", "", fmt.Errorf("redfish http: uri must begin with /redfish/v1/")
	}

	// Validate character set: no control characters or whitespace.
	for _, r := range cleaned {
		if unicode.IsControl(r) || unicode.IsSpace(r) {
			return "", "", fmt.Errorf("redfish http: invalid characters in uri path")
		}
	}
	for _, r := range u.RawQuery {
		if unicode.IsControl(r) || unicode.IsSpace(r) {
			return "", "", fmt.Errorf("redfish http: invalid characters in uri query")
		}
	}

	rel := &url.URL{Path: cleaned, RawQuery: u.RawQuery}
	resolved := c.baseParsed.ResolveReference(rel)
	if !strings.EqualFold(resolved.Scheme, c.baseParsed.Scheme) || !strings.EqualFold(resolved.Host, c.baseParsed.Host) || !strings.HasPrefix(resolved.Path, "/redfish/v1/") {
		return "", "", fmt.Errorf("redfish http: invalid resolved target")
	}
	return resolved.String(), cleaned, nil
}

// GetSensor fetches a DMTF Sensor.v1_9_0 resource from the BMC and validates it
// against the provided host Context, expected URI, and sensor ID.
// BMC timestamps are retained verbatim and do not constitute sensor calibration proof.
// Transport errors and HTTP status errors never leak URLs, query parameters, or credentials.
func (c *RedfishHTTPClient) GetSensor(ctx context.Context, hostCtx Context, uri, id string) (RedfishSensor, error) {
	if c == nil || c.client == nil {
		return RedfishSensor{}, fmt.Errorf("redfish http: uninitialized client")
	}
	if ctx == nil {
		return RedfishSensor{}, fmt.Errorf("redfish http: nil context")
	}
	if err := ctx.Err(); err != nil {
		return RedfishSensor{}, err
	}
	targetURL, canonicalPath, err := c.resolveURI(uri)
	if err != nil {
		return RedfishSensor{}, err
	}
	reqCtx, cancel := context.WithTimeout(ctx, c.config.RequestTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, targetURL, nil)
	if err != nil {
		return RedfishSensor{}, fmt.Errorf("redfish http: failed to create request")
	}
	req.Header = c.headers.Clone()
	if req.Header == nil {
		req.Header = make(http.Header)
	}
	req.Header.Set("Accept", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		if errors.Is(ctx.Err(), context.Canceled) || errors.Is(err, context.Canceled) {
			return RedfishSensor{}, context.Canceled
		}
		if errors.Is(ctx.Err(), context.DeadlineExceeded) || errors.Is(err, context.DeadlineExceeded) {
			return RedfishSensor{}, context.DeadlineExceeded
		}
		return RedfishSensor{}, fmt.Errorf("redfish http: transport error")
	}
	defer resp.Body.Close()
	if err := ctx.Err(); err != nil {
		return RedfishSensor{}, err
	}
	if resp.StatusCode != http.StatusOK {
		return RedfishSensor{}, fmt.Errorf("redfish http: unexpected status %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, c.config.MaxResponseBytes+1))
	if err != nil {
		return RedfishSensor{}, fmt.Errorf("redfish http: read response error")
	}
	if err := ctx.Err(); err != nil {
		return RedfishSensor{}, err
	}
	if int64(len(body)) > c.config.MaxResponseBytes {
		return RedfishSensor{}, fmt.Errorf("redfish http: response exceeds byte limit")
	}
	sensor, err := ReadRedfishSensor(bytes.NewReader(body), hostCtx, canonicalPath, id, c.config.MaxResponseBytes)
	if err != nil && canonicalPath != uri {
		var retryErr error
		sensor, retryErr = ReadRedfishSensor(bytes.NewReader(body), hostCtx, uri, id, c.config.MaxResponseBytes)
		if retryErr != nil {
			return RedfishSensor{}, err
		}
	} else if err != nil {
		return RedfishSensor{}, err
	}
	if err := ctx.Err(); err != nil {
		return RedfishSensor{}, err
	}
	return sensor, nil
}

// RedfishCollectionPage contains members and count metadata from a single DMTF Redfish collection response.
type RedfishCollectionPage struct {
	URI                string
	HasAdvertisedCount bool
	AdvertisedCount    int
	Members            []string
	NextLink           string
}

// ReadRedfishCollection parses a single DMTF Redfish collection artifact.
// The URI must equal the caller-expected collection URI.
func ReadRedfishCollection(r io.Reader, uri string, maxBytes int64) (RedfishCollectionPage, error) {
	if r == nil || maxBytes <= 0 || maxBytes > 1<<20 {
		return RedfishCollectionPage{}, fmt.Errorf("invalid collection reader or byte limit")
	}
	if !validRedfishArtifactURI(uri) {
		return RedfishCollectionPage{}, fmt.Errorf("invalid Redfish collection uri")
	}
	b, err := io.ReadAll(io.LimitReader(r, maxBytes+1))
	if err != nil {
		return RedfishCollectionPage{}, fmt.Errorf("read Redfish collection: %w", err)
	}
	if int64(len(b)) > maxBytes {
		return RedfishCollectionPage{}, fmt.Errorf("Redfish collection exceeds byte limit")
	}
	if len(b) == 0 {
		return RedfishCollectionPage{}, fmt.Errorf("empty Redfish collection")
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	if err := checkJSON(d, 0); err != nil {
		return RedfishCollectionPage{}, err
	}
	if _, err := d.Token(); err != io.EOF {
		return RedfishCollectionPage{}, fmt.Errorf("trailing Redfish JSON")
	}
	var raw struct {
		ODataID  string `json:"@odata.id"`
		Count    *int   `json:"Members@odata.count"`
		NextLink string `json:"Members@odata.nextLink"`
		Members  []struct {
			ODataID string `json:"@odata.id"`
		} `json:"Members"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return RedfishCollectionPage{}, err
	}
	if raw.ODataID != uri {
		return RedfishCollectionPage{}, fmt.Errorf("Redfish collection identity mismatch")
	}
	page := RedfishCollectionPage{
		URI:      raw.ODataID,
		NextLink: raw.NextLink,
	}
	if raw.Count != nil {
		if *raw.Count < 0 {
			return RedfishCollectionPage{}, fmt.Errorf("negative Redfish member count")
		}
		page.HasAdvertisedCount = true
		page.AdvertisedCount = *raw.Count
	} else {
		page.HasAdvertisedCount = false
		page.AdvertisedCount = -1
	}
	seen := make(map[string]bool, len(raw.Members))
	for _, m := range raw.Members {
		mURI := m.ODataID
		if !validRedfishArtifactURI(mURI) {
			return RedfishCollectionPage{}, fmt.Errorf("invalid Redfish member identity")
		}
		if seen[mURI] {
			return RedfishCollectionPage{}, fmt.Errorf("duplicate Redfish member in collection")
		}
		seen[mURI] = true
		page.Members = append(page.Members, mURI)
	}
	return page, nil
}

// RedfishCollection represents a complete DMTF Redfish collection aggregated
// across bounded pagination pages.
type RedfishCollection struct {
	URI                string   // Base collection URI
	Members            []string // Caller-admitted member @odata.id URIs
	Count              int      // Total collected members count
	AdvertisedCount    int      // Advertised Members@odata.count when provided (-1 if omitted)
	HasAdvertisedCount bool     // True if Members@odata.count was specified by the BMC
}

// GetCollection fetches a DMTF Redfish collection resource from the BMC, following
// same-origin pagination nextLinks up to the configured MaxPages and MaxCollectionMembers.
// Cross-origin nextLinks, redirect responses, and pagination cycles are refused.
// Advertised member counts are preserved and enforced against collected totals.
func (c *RedfishHTTPClient) GetCollection(ctx context.Context, collectionURI string) (RedfishCollection, error) {
	if c == nil || c.client == nil {
		return RedfishCollection{}, fmt.Errorf("redfish http: uninitialized client")
	}
	if ctx == nil {
		return RedfishCollection{}, fmt.Errorf("redfish http: nil context")
	}
	if err := ctx.Err(); err != nil {
		return RedfishCollection{}, err
	}
	// Initial URI validation.
	_, collectionCanonicalPath, err := c.resolveURI(collectionURI)
	if err != nil {
		return RedfishCollection{}, err
	}

	currentURI := collectionURI
	var allMembers []string
	seenMembers := make(map[string]bool)
	visitedPages := make(map[string]bool)
	hasAdvertised := false
	advertisedTotal := -1
	pages := 0

	for {
		if pages >= c.config.MaxPages {
			return RedfishCollection{}, fmt.Errorf("redfish http: collection page limit exceeded")
		}
		pages++

		targetURL, canonicalPath, err := c.resolveURI(currentURI)
		if err != nil {
			return RedfishCollection{}, err
		}

		// Detect pagination cycle immediately.
		if visitedPages[targetURL] {
			return RedfishCollection{}, fmt.Errorf("redfish http: pagination cycle detected")
		}
		visitedPages[targetURL] = true

		if err := ctx.Err(); err != nil {
			return RedfishCollection{}, err
		}

		reqCtx, cancel := context.WithTimeout(ctx, c.config.RequestTimeout)
		req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, targetURL, nil)
		if err != nil {
			cancel()
			return RedfishCollection{}, fmt.Errorf("redfish http: failed to create request")
		}
		req.Header = c.headers.Clone()
		if req.Header == nil {
			req.Header = make(http.Header)
		}
		req.Header.Set("Accept", "application/json")

		resp, err := c.client.Do(req)
		if err != nil {
			cancel()
			if errors.Is(ctx.Err(), context.Canceled) || errors.Is(err, context.Canceled) {
				return RedfishCollection{}, context.Canceled
			}
			if errors.Is(ctx.Err(), context.DeadlineExceeded) || errors.Is(err, context.DeadlineExceeded) {
				return RedfishCollection{}, context.DeadlineExceeded
			}
			return RedfishCollection{}, fmt.Errorf("redfish http: transport error")
		}
		if err := ctx.Err(); err != nil {
			_ = resp.Body.Close()
			cancel()
			return RedfishCollection{}, err
		}
		if resp.StatusCode != http.StatusOK {
			_ = resp.Body.Close()
			cancel()
			return RedfishCollection{}, fmt.Errorf("redfish http: unexpected status %d", resp.StatusCode)
		}
		body, err := io.ReadAll(io.LimitReader(resp.Body, c.config.MaxResponseBytes+1))
		_ = resp.Body.Close()
		cancel()
		if err != nil {
			return RedfishCollection{}, fmt.Errorf("redfish http: read response error")
		}
		if err := ctx.Err(); err != nil {
			return RedfishCollection{}, err
		}
		if int64(len(body)) > c.config.MaxResponseBytes {
			return RedfishCollection{}, fmt.Errorf("redfish http: response exceeds byte limit")
		}

		// Parse the page.
		// Redfish DMTF specifies that the page @odata.id must match either the
		// current request URI or the exact original collection URI.
		page, err := ReadRedfishCollection(bytes.NewReader(body), canonicalPath, c.config.MaxResponseBytes)
		if err != nil {
			if canonicalPath != collectionCanonicalPath {
				var retryErr error
				page, retryErr = ReadRedfishCollection(bytes.NewReader(body), collectionCanonicalPath, c.config.MaxResponseBytes)
				if retryErr != nil {
					return RedfishCollection{}, fmt.Errorf("redfish http: collection page identity mismatch: %w", err)
				}
			} else {
				return RedfishCollection{}, err
			}
		}

		// Enforce consistent advertised count across pages.
		if page.HasAdvertisedCount {
			if !hasAdvertised {
				hasAdvertised = true
				advertisedTotal = page.AdvertisedCount
			} else if page.AdvertisedCount != advertisedTotal {
				return RedfishCollection{}, fmt.Errorf("redfish http: inconsistent advertised collection count across pages")
			}
		}

		for _, m := range page.Members {
			if seenMembers[m] {
				return RedfishCollection{}, fmt.Errorf("redfish http: duplicate member across pages")
			}
			seenMembers[m] = true
			allMembers = append(allMembers, m)
			if len(allMembers) > c.config.MaxCollectionMembers {
				return RedfishCollection{}, fmt.Errorf("redfish http: collection member limit exceeded")
			}
		}

		if page.NextLink == "" {
			break
		}

		// Next link must resolve to a valid same-origin Redfish URI.
		if _, _, err := c.resolveURI(page.NextLink); err != nil {
			return RedfishCollection{}, fmt.Errorf("redfish http: invalid nextLink: %w", err)
		}
		currentURI = page.NextLink
	}

	// Enforce terminal equality if advertised count was provided.
	if hasAdvertised && len(allMembers) != advertisedTotal {
		return RedfishCollection{}, fmt.Errorf("redfish http: collected members count contradicts advertised count")
	}

	if err := ctx.Err(); err != nil {
		return RedfishCollection{}, err
	}

	return RedfishCollection{
		URI:                collectionURI,
		Members:            allMembers,
		Count:              len(allMembers),
		AdvertisedCount:    advertisedTotal,
		HasAdvertisedCount: hasAdvertised,
	}, nil
}

// Artifact identities identify entities, not pagination request URLs.
func validRedfishArtifactURI(uri string) bool {
	if !identity(uri) {
		return false
	}
	u, err := url.Parse(uri)
	if err != nil || u.IsAbs() || u.Host != "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || path.Clean(u.Path) != u.Path || !strings.HasPrefix(u.Path, "/redfish/v1/") {
		return false
	}
	lower := strings.ToLower(uri)
	if strings.Contains(lower, "%2e") || strings.Contains(lower, "%2f") || strings.Contains(lower, "%5c") || strings.ContainsAny(u.Path, "%\\") {
		return false
	}
	for _, r := range u.Path {
		if unicode.IsControl(r) || unicode.IsSpace(r) {
			return false
		}
	}
	return true
}
