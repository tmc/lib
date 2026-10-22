// Command ds4-server-corpus replays a finalized server corpus over HTTP.
//
// The command is deliberately a wire-only gate. HTTP servers do not expose
// generated token IDs, so a successful wire comparison is reported as
// UNREAD when the corpus contains token traces.
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"sort"
	"strings"
	"time"

	"github.com/tmc/lib/examples/ds4-min/dsv4server"
)

const maxResponseBytes = 8 << 20

type caseReceipt struct {
	Key          string `json:"key"`
	Status       string `json:"status"`
	HTTPStatus   string `json:"http_status,omitempty"`
	ElapsedNanos int64  `json:"elapsed_nanos"`
	TokenStatus  string `json:"token_status"`
	Error        string `json:"error,omitempty"`
}

type receipt struct {
	Status       string        `json:"status"`
	Corpus       string        `json:"corpus"`
	BaseURL      string        `json:"base_url"`
	Cases        []caseReceipt `json:"cases"`
	Matched      int           `json:"matched"`
	Mismatched   int           `json:"mismatched"`
	TokensUnread int           `json:"tokens_unread"`
}

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }

func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("ds4-server-corpus", flag.ContinueOnError)
	fs.SetOutput(stderr)
	corpusPath := fs.String("corpus", "", "finalized server corpus root")
	baseURL := fs.String("base-url", "", "server base URL, for example http://127.0.0.1:8080")
	outPath := fs.String("out", "", "write the JSON receipt to this path instead of stdout")
	timeout := fs.Duration("timeout", 5*time.Minute, "per-request timeout")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || *corpusPath == "" || *baseURL == "" {
		fmt.Fprintln(stderr, "ds4-server-corpus: -corpus and -base-url are required")
		return 2
	}
	base, err := parseBaseURL(*baseURL)
	if err != nil {
		fmt.Fprintf(stderr, "ds4-server-corpus: %v\n", err)
		return 2
	}
	if *timeout <= 0 {
		fmt.Fprintln(stderr, "ds4-server-corpus: timeout must be positive")
		return 2
	}
	corpus, err := dsv4server.LoadCorpus(*corpusPath)
	if err != nil {
		fmt.Fprintf(stderr, "ds4-server-corpus: load corpus: %v\n", err)
		return 2
	}
	r := replay(corpus, base, &http.Client{Timeout: *timeout})
	data, err := json.MarshalIndent(r, "", "  ")
	if err == nil {
		data = append(data, '\n')
	}
	if err == nil && *outPath != "" {
		err = os.WriteFile(*outPath, data, 0600)
	} else if err == nil {
		_, err = stdout.Write(data)
	}
	if err != nil {
		fmt.Fprintf(stderr, "ds4-server-corpus: write receipt: %v\n", err)
		return 2
	}
	if r.Status != "PASS" {
		return 1
	}
	return 0
}

func parseBaseURL(raw string) (*url.URL, error) {
	u, err := url.Parse(raw)
	if err != nil || u.Scheme == "" || u.Host == "" || u.Path != "" && u.Path != "/" {
		return nil, errors.New("base-url must be an absolute URL without a path")
	}
	u.Path = "/"
	u.RawQuery, u.Fragment = "", ""
	return u, nil
}

func replay(corpus dsv4server.Corpus, base *url.URL, client *http.Client) receipt {
	r := receipt{Status: "PASS", Corpus: "finalized", BaseURL: base.String()}
	keys := make([]string, 0)
	byKey := make(map[string]dsv4server.CorpusCase)
	for group, cases := range corpus.Cases {
		for _, c := range cases {
			key := group + "/" + c.Step
			keys = append(keys, key)
			byKey[key] = c
		}
	}
	sort.Strings(keys)
	for _, key := range keys {
		c := byKey[key]
		cr := caseReceipt{Key: key, Status: "PASS", TokenStatus: "PASS"}
		if len(c.Tokens) != 0 {
			cr.TokenStatus = "UNREAD"
			r.TokensUnread++
		}
		started := time.Now()
		got, err := replayCase(c, base, client)
		cr.ElapsedNanos = time.Since(started).Nanoseconds()
		if err == nil {
			cr.HTTPStatus = got.Status
			err = dsv4server.CompareCorpusHTTP(c, got)
		}
		if err != nil {
			cr.Status, cr.Error = "FAIL", err.Error()
			r.Mismatched++
		} else {
			r.Matched++
		}
		r.Cases = append(r.Cases, cr)
	}
	if r.Mismatched != 0 {
		r.Status = "FAIL"
	} else if r.TokensUnread != 0 {
		r.Status = "UNREAD"
	}
	return r
}

func replayCase(c dsv4server.CorpusCase, base *url.URL, client *http.Client) (dsv4server.CorpusObservation, error) {
	u := *base
	endpoint, err := url.Parse(c.Endpoint)
	if err != nil || !strings.HasPrefix(endpoint.Path, "/") || endpoint.Host != "" {
		return dsv4server.CorpusObservation{}, fmt.Errorf("invalid endpoint %q", c.Endpoint)
	}
	u.Path = path.Join("/", endpoint.Path)
	u.RawQuery, u.Fragment = endpoint.RawQuery, ""
	req, err := http.NewRequest(c.Method, u.String(), bytes.NewReader(c.Request))
	if err != nil {
		return dsv4server.CorpusObservation{}, fmt.Errorf("create request: %w", err)
	}
	if len(c.Request) != 0 {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := client.Do(req)
	if err != nil {
		return dsv4server.CorpusObservation{}, fmt.Errorf("request: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes+1))
	if err != nil {
		return dsv4server.CorpusObservation{}, fmt.Errorf("read response: %w", err)
	}
	if len(body) > maxResponseBytes {
		return dsv4server.CorpusObservation{}, errors.New("response exceeds replay limit")
	}
	return dsv4server.CorpusObservation{
		Status:   resp.Proto + " " + resp.Status,
		Request:  append([]byte(nil), c.Request...),
		Response: body,
	}, nil
}
