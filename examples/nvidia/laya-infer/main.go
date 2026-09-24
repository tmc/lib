// Laya-infer answers structured decision questions with the Laya RL agent
// (a ModernBERT encoder plus a small decision head) on an NVIDIA GPU.
//
// Usage:
//
//	laya-infer MODEL_DIR REQUEST.json|-
//	laya-infer MODEL_DIR --tokenize TEXT
//	laya-infer MODEL_DIR --list-tensors
//	laya-infer MODEL_DIR --serve [PORT] [--bind ADDR]
//
// A request is {"state": ..., "questions": {ID: {...}}}; the reply gives one
// answer per question. It is a port of laya.c, and answers agree with it to
// within float32 rounding. Unlike laya.c, every question in a request is
// evaluated in a single padded batch.
//
// Matrix products run on the tensor cores in tf32 by default. --exact runs
// them in float32 instead, which reproduces laya.c byte for byte but is
// slower: full float32 GEMMs forgo the tf32 tensor-core path and the fused
// attention kernel.
//
// --lines answers the request a second time under CUPTI PC sampling and
// reports the device time by line of forward.cu. --pprof writes those same
// samples to a file as a pprof profile, which keeps the stall reasons the
// text report cannot carry.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/x/safetensors"
)

// defaultPort is below the Linux ephemeral range and unassigned by IANA.
const defaultPort = 29417

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "laya:", err)
		os.Exit(1)
	}
}

func usage() error {
	fmt.Fprintf(os.Stderr, `usage: laya-infer MODEL_DIR REQUEST.json|-
       laya-infer MODEL_DIR --tokenize TEXT
       laya-infer MODEL_DIR --list-tensors
       laya-infer MODEL_DIR --serve [PORT] [--bind ADDR]   (HTTP server, default 127.0.0.1:%d)

       --exact   compute the matrix products in float32 rather than tf32,
                 reproducing laya.c byte for byte; slower, since float32
                 GEMMs do not use the tf32 tensor-core path
       --lines[=N]
                 answer the request under CUPTI PC sampling and report the
                 device time by line of forward.cu, sampling every 2^N
                 cycles (default 2^%d)
       --pprof FILE
                 also write those samples to FILE as a pprof profile, with
                 the stall reasons the report leaves out; implies --lines
`, defaultPort, linesPeriod)
	os.Exit(2)
	return nil
}

// parseExact removes --exact from args wherever it appears and reports which
// precision is left. Everything else on this command line is positional, so a
// flag package would have to see the option before the model directory, and a
// caller adding it to a command line that already works puts it at the end as
// often as not.
func parseExact(args []string) ([]string, precision) {
	prec := fast
	rest := make([]string, 0, len(args))
	for _, arg := range args {
		if arg == "--exact" || arg == "-exact" {
			prec = exact
			continue
		}
		rest = append(rest, arg)
	}
	return rest, prec
}

// parseLines removes --lines from args wherever it appears, following
// parseExact's convention, and reports the sampling period it asks for.
// --lines=N overrides the default period; zero means the flag was absent.
func parseLines(args []string) ([]string, uint32, error) {
	var period uint32
	rest := make([]string, 0, len(args))
	for _, arg := range args {
		name, value, hasValue := strings.Cut(strings.TrimPrefix(arg, "-"), "=")
		if name != "-lines" && name != "lines" {
			rest = append(rest, arg)
			continue
		}
		period = linesPeriod
		if hasValue {
			n, err := strconv.Atoi(value)
			if err != nil || n < 5 || n > 31 {
				return nil, 0, fmt.Errorf("--lines=%s: the sampling period exponent is 5 through 31", value)
			}
			period = uint32(n)
		}
	}
	return rest, period, nil
}

// parsePprof removes --pprof from args wherever it appears, following
// parseLines's convention, and reports the file it names. Both --pprof=FILE
// and --pprof FILE are accepted, since the flag takes a path and the one
// other option here that takes one, --bind, is written with a space. An
// empty path means the flag was absent.
func parsePprof(args []string) ([]string, string, error) {
	var path string
	rest := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		name, value, hasValue := strings.Cut(strings.TrimPrefix(args[i], "-"), "=")
		if name != "-pprof" && name != "pprof" {
			rest = append(rest, args[i])
			continue
		}
		if !hasValue {
			if i+1 == len(args) {
				return nil, "", errors.New("--pprof needs a file argument")
			}
			i++
			value = args[i]
		}
		if value == "" {
			return nil, "", errors.New("--pprof needs a file argument")
		}
		path = value
	}
	return rest, path, nil
}

func run(args []string) error {
	args, prec := parseExact(args)
	args, period, err := parseLines(args)
	if err != nil {
		return err
	}
	args, pprof, err := parsePprof(args)
	if err != nil {
		return err
	}
	// The profile is made of the PC samples --lines collects, so naming a
	// file is asking for the sampled run whether or not --lines is there.
	if pprof != "" && period == 0 {
		period = linesPeriod
	}
	if len(args) < 2 {
		return usage()
	}
	dir := args[0]
	switch args[1] {
	case "--tokenize":
		if len(args) < 3 {
			return errors.New("--tokenize needs a text argument")
		}
		return tokenize(dir, args[2])
	case "--list-tensors":
		return listTensors(dir)
	case "--serve":
		addr, err := serveAddr(args[2:])
		if err != nil {
			return err
		}
		return serve(dir, addr, prec)
	}

	if period != 0 {
		return predictLines(dir, args[1], prec, period, pprof)
	}

	start := time.Now()
	m, err := newModel(dir, prec)
	if err != nil {
		return err
	}
	defer m.Close()
	if os.Getenv("LAYA_DEBUG") != "" {
		fmt.Fprintf(os.Stderr, "[load] %.0f ms\n", float64(time.Since(start).Milliseconds()))
	}

	req, err := readInput(args[1])
	if err != nil {
		return err
	}
	res, err := m.Predict(req)
	if err != nil {
		return err
	}
	fmt.Println(string(res))
	return nil
}

func readInput(path string) ([]byte, error) {
	if path == "-" {
		return io.ReadAll(os.Stdin)
	}
	return os.ReadFile(path)
}

func tokenize(dir, text string) error {
	t, err := loadTokenizer(dir)
	if err != nil {
		return err
	}
	ids, err := t.encode(text, -1)
	if err != nil {
		return err
	}
	parts := make([]string, len(ids))
	for i, id := range ids {
		parts[i] = strconv.Itoa(int(id))
	}
	fmt.Println(strings.Join(parts, " "))
	return nil
}

func listTensors(dir string) error {
	st, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return err
	}
	infos := slices.SortedFunc(slices.Values(st.Tensors), func(a, b safetensors.TensorInfo) int {
		return strings.Compare(a.Name, b.Name)
	})
	for _, info := range infos {
		dims := make([]string, len(info.Shape))
		for i, n := range info.Shape {
			dims[i] = strconv.Itoa(n)
		}
		fmt.Printf("%s  %s [%s]\n", info.Name, info.Dtype, strings.Join(dims, ","))
	}
	return nil
}

func serveAddr(args []string) (string, error) {
	host, port := "127.0.0.1", defaultPort
	for i := 0; i < len(args); i++ {
		switch {
		case args[i] == "--bind" && i+1 < len(args):
			i++
			host = args[i]
		default:
			n, err := strconv.Atoi(args[i])
			if err != nil || n <= 0 || n >= 65536 {
				return "", fmt.Errorf("unknown argument %q (usage: MODEL_DIR --serve [PORT] [--bind ADDR])", args[i])
			}
			port = n
		}
	}
	return net.JoinHostPort(host, strconv.Itoa(port)), nil
}

// server serializes inference so concurrent requests queue instead of
// oversubscribing the GPU, while /status stays answerable.
type server struct {
	m      *model
	loadMS float64
	start  time.Time

	mu       sync.Mutex // held for the duration of one inference
	stats    sync.Mutex
	served   int64
	failed   int64
	inFlight int64
	running  bool
	lastMS   float64
	totalMS  float64
}

func serve(dir, addr string, prec precision) error {
	start := time.Now()
	m, err := newModel(dir, prec)
	if err != nil {
		return err
	}
	defer m.Close()
	s := &server{m: m, loadMS: float64(time.Since(start).Milliseconds()), start: time.Now()}

	mux := http.NewServeMux()
	for _, p := range []string{"/status", "/health", "/healthz"} {
		mux.HandleFunc(p, s.status)
	}
	for _, p := range []string{"/predict", "/questions", "/"} {
		mux.HandleFunc(p, s.predict)
	}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "laya: model loaded in %.0f ms; listening on http://%s  (GET /status, POST /predict)\n",
		s.loadMS, ln.Addr())
	return http.Serve(ln, mux)
}

func (s *server) status(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		httpError(w, http.StatusMethodNotAllowed, "use GET")
		return
	}
	c := s.m.cfg
	s.stats.Lock()
	served, failed, inFlight := s.served, s.failed, s.inFlight
	running, lastMS, totalMS := s.running, s.lastMS, s.totalMS
	s.stats.Unlock()
	avg := 0.0
	if served > 0 {
		avg = totalMS / float64(served)
	}
	name, _ := s.m.dev.Name()
	writeJSON(w, http.StatusOK, map[string]any{
		"status":     "ready",
		"model":      "laya-rl-agent",
		"uptime_s":   round1(time.Since(s.start).Seconds()),
		"load_ms":    round1(s.loadMS),
		"requests":   map[string]any{"served": served, "failed": failed, "in_flight": inFlight, "inference_running": running},
		"latency_ms": map[string]any{"last": round1(lastMS), "avg": round1(avg)},
		"engine": map[string]any{
			"hidden": c.D, "layers": c.NumLayers, "heads": c.NumHeads, "vocab": c.Vocab,
			"max_len": c.MaxLen, "head_max_len": c.HeadMaxLen, "device": name,
		},
	})
}

func (s *server) predict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httpError(w, http.StatusMethodNotAllowed,
			`use POST with a JSON body {"state": ..., "questions": {...}}`)
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 32<<20))
	if err != nil {
		s.record(false, 0)
		httpError(w, http.StatusRequestEntityTooLarge, "request body too large")
		return
	}

	s.stats.Lock()
	s.inFlight++
	s.stats.Unlock()

	s.mu.Lock()
	s.stats.Lock()
	s.running = true
	s.stats.Unlock()
	t0 := time.Now()
	res, err := s.m.Predict(body)
	ms := float64(time.Since(t0).Microseconds()) / 1e3
	s.stats.Lock()
	s.running = false
	s.stats.Unlock()
	s.mu.Unlock()

	s.record(err == nil, ms)
	if err != nil {
		httpError(w, http.StatusBadRequest, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Inference-Ms", strconv.FormatFloat(ms, 'f', 1, 64))
	w.Write(res)
	w.Write([]byte("\n"))
}

func (s *server) record(ok bool, ms float64) {
	s.stats.Lock()
	defer s.stats.Unlock()
	s.inFlight--
	if ok {
		s.served++
		s.lastMS = ms
		s.totalMS += ms
	} else {
		s.failed++
	}
}

func httpError(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]any{"error": msg})
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func round1(x float64) float64 {
	s := strconv.FormatFloat(x, 'f', 1, 64)
	v, _ := strconv.ParseFloat(s, 64)
	return v
}
