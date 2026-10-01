package stack

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// GoTrace retains timestamped CPU samples and clock calibration evidence.
// ClockStatus requires producer platform evidence and native clock validation.
// The caller must separately verify native process identity and unified clocks.
type GoTrace struct {
	InputSHA256        string          `json:"input_sha256"`
	Backend            string          `json:"backend"`
	SnapshotReadSkewNs uint64          `json:"snapshot_read_skew_ns,omitempty"`
	ClockUncertaintyNs uint64          `json:"clock_uncertainty_ns,omitempty"`
	ClockStatus        string          `json:"clock_status"`
	ClockSnapshots     []ClockSnapshot `json:"clock_snapshots"`
	Samples            []Sample        `json:"samples"`
}

// ClockSnapshot maps a Go trace timestamp to host monotonic nanoseconds.
type ClockSnapshot struct {
	Trace     uint64 `json:"trace"`
	Monotonic uint64 `json:"monotonic"`
}

var goEventHeader = regexp.MustCompile(`^M=(-?[0-9]+) P=(-?[0-9]+) G=(-?[0-9]+) ([A-Za-z]+) Time=([0-9]+)(.*)$`)
var goClockSnapshot = regexp.MustCompile(` Trace=([0-9]+) Mono=([0-9]+) `)

// ReadGoTraceDebug decodes the Go 1.27.1 parsed debug format. Other Go debug
// formats are unsupported. Frames are retained leaf first; samples are CPU
// samples, not wall snapshots or submission stacks.
func ReadGoTraceDebug(rd io.Reader, identity Record, limits Limits) (*GoTrace, error) {
	l := limits.defaults()
	s := bufio.NewScanner(io.LimitReader(rd, l.Bytes+1))
	s.Buffer(make([]byte, 4096), 1<<20)
	out := &GoTrace{Backend: "go1.27.1 trace -d=parsed", ClockStatus: "unavailable"}
	var size int64
	var sample *Sample
	var inStack bool
	var pending string
	finish := func() error {
		if sample == nil {
			return nil
		}
		if pending != "" {
			return fmt.Errorf("incomplete Go CPU sample frame")
		}
		if len(sample.Frames) == 0 {
			return fmt.Errorf("go CPU sample missing stack")
		}
		out.Samples = append(out.Samples, *sample)
		if len(out.Samples) > l.Records {
			return fmt.Errorf("go CPU sample limit exceeded")
		}
		sample = nil
		inStack = false
		return nil
	}
	for s.Scan() {
		line := s.Text()
		size += int64(len(line) + 1)
		if size > l.Bytes {
			return nil, fmt.Errorf("go trace debug byte limit exceeded")
		}
		if strings.Contains(line, " StackSample ") && !goEventHeader.MatchString(line) {
			return nil, fmt.Errorf("invalid Go sample event")
		}
		if m := goEventHeader.FindStringSubmatch(line); m != nil {
			if err := finish(); err != nil {
				return nil, err
			}
			if m[4] == "Sync" {
				if c := goClockSnapshot.FindStringSubmatch(m[6]); c != nil {
					a, _ := strconv.ParseUint(c[1], 10, 64)
					b, _ := strconv.ParseUint(c[2], 10, 64)
					out.ClockSnapshots = append(out.ClockSnapshots, ClockSnapshot{a, b})
				}
				continue
			}
			if m[4] != "StackSample" {
				continue
			}
			if identity.PID <= 0 || identity.Process == "" {
				return nil, fmt.Errorf("missing Go trace process identity")
			}
			ts, err := strconv.ParseUint(m[5], 10, 64)
			if err != nil {
				return nil, err
			}
			tid, _ := strconv.ParseUint(m[1], 10, 64)
			gid, _ := strconv.ParseUint(m[3], 10, 64)
			sample = &Sample{Version: 1, Type: "sample", Process: identity.Process, PID: identity.PID, Session: identity.Session, Generation: identity.Generation, Language: "go", TID: tid, Goroutine: gid, Time: ts, TraceTime: ts, Clock: "go_trace", Status: "clock_unavailable", Provenance: "cpu_sample"}
			continue
		}
		if sample == nil {
			continue
		}
		if line == "Stack=" {
			if inStack {
				return nil, fmt.Errorf("duplicate Go sample stack")
			}
			inStack = true
			continue
		}
		if !inStack {
			if line != "" {
				return nil, fmt.Errorf("unexpected Go sample data")
			}
			continue
		}
		if line == "" {
			continue
		}
		if pending == "" {
			if !strings.HasPrefix(line, "\t") || strings.HasPrefix(line, "\t\t") {
				return nil, fmt.Errorf("invalid Go stack function")
			}
			i := strings.LastIndex(line, " @ 0x")
			if i <= 1 {
				return nil, fmt.Errorf("invalid Go stack function")
			}
			pending = line[1:i]
			continue
		}
		if !strings.HasPrefix(line, "\t\t") {
			return nil, fmt.Errorf("invalid Go stack location")
		}
		loc := strings.TrimPrefix(line, "\t\t")
		i := strings.LastIndexByte(loc, ':')
		if i < 0 {
			return nil, fmt.Errorf("invalid Go stack location")
		}
		n, err := strconv.ParseUint(loc[i+1:], 10, 32)
		if err != nil {
			return nil, fmt.Errorf("invalid Go stack line")
		}
		sample.Frames = append(sample.Frames, Frame{Function: pending, File: loc[:i], Line: uint32(n)})
		pending = ""
		if len(sample.Frames) > l.Frames {
			return nil, fmt.Errorf("go stack frame limit exceeded")
		}
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	if err := finish(); err != nil {
		return nil, err
	}

	return out, nil
}

// AnalyzeGoTrace invokes an explicitly selected local Go 1.27.1 tool. It does
// not download tools. Input size, debug output and execution time are bounded.
func AnalyzeGoTrace(ctx context.Context, goTool, path string, identity Record, limits Limits) (*GoTrace, error) {
	if goTool == "" {
		return nil, fmt.Errorf("go trace backend unavailable")
	}
	in, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer in.Close()
	st, err := in.Stat()
	if err != nil {
		return nil, err
	}
	if !st.Mode().IsRegular() || st.Size() > limits.defaults().Bytes {
		return nil, fmt.Errorf("go trace input limit exceeded")
	}
	return analyzeGoTraceReader(ctx, goTool, in, identity, limits)
}

// AnalyzeGoTraceBytes decodes a retained trace through a private bounded
// snapshot using an explicitly selected local Go 1.27.1 tool.
func AnalyzeGoTraceBytes(ctx context.Context, goTool string, data []byte, identity Record, limits Limits) (*GoTrace, error) {
	if int64(len(data)) > limits.defaults().Bytes {
		return nil, fmt.Errorf("go trace input limit exceeded")
	}
	return analyzeGoTraceReader(ctx, goTool, bytes.NewReader(data), identity, limits)
}

func analyzeGoTraceReader(ctx context.Context, goTool string, in io.Reader, identity Record, limits Limits) (*GoTrace, error) {
	if goTool == "" {
		return nil, fmt.Errorf("go trace backend unavailable")
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	l := limits.defaults()

	var version cappedBuffer
	version.limit = 4096
	cmd := exec.CommandContext(ctx, goTool, "version")
	cmd.Stdout = &version
	cmd.Stderr = &version
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("qualify Go trace backend: %w", err)
	}
	fields := strings.Fields(version.String())
	if len(fields) != 4 || fields[0] != "go" || fields[1] != "version" || fields[2] != "go1.27.1" {
		return nil, fmt.Errorf("unsupported Go trace backend %q", strings.TrimSpace(version.String()))
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}
	scratch := filepath.Join(home, "tmp")
	if err := os.MkdirAll(scratch, 0700); err != nil {
		return nil, err
	}
	dir, err := os.MkdirTemp(scratch, "cudaprof-go-trace-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	snapshot := filepath.Join(dir, "runtime.trace")
	f, err := os.OpenFile(snapshot, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return nil, err
	}
	hash := sha256.New()
	n, copyErr := io.Copy(io.MultiWriter(f, hash), io.LimitReader(in, l.Bytes+1))
	closeErr := f.Close()
	if copyErr != nil {
		return nil, copyErr
	}
	if closeErr != nil {
		return nil, closeErr
	}
	if n > l.Bytes {
		return nil, fmt.Errorf("go trace input limit exceeded")
	}
	var output, stderr cappedBuffer
	output.limit = l.Bytes
	stderr.limit = 1 << 20
	cmd = exec.CommandContext(ctx, goTool, "tool", "trace", "-d=parsed", snapshot)
	cmd.Stdout = &output
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("decode Go trace: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	out, err := ReadGoTraceDebug(&output.Buffer, identity, l)
	if err != nil {
		return nil, err
	}
	out.InputSHA256 = hex.EncodeToString(hash.Sum(nil))
	return out, nil
}

type cappedBuffer struct {
	bytes.Buffer
	limit int64
}

func (b *cappedBuffer) Write(p []byte) (int, error) {
	if int64(b.Len())+int64(len(p)) > b.limit {
		return 0, fmt.Errorf("go trace backend output limit exceeded")
	}
	return b.Buffer.Write(p)
}

// QualifyGoMonotonic checks the pinned producer's platform and clock snapshots.
// Go 1.27.1 on Linux records nanotime divided by 64; parsed timestamps retain
// that quantization. Snapshot clocks are read separately, so their observed
// skew is retained without treating it as an offset.
// A caller must still verify the exact native process generation and its
// unified-clock capability before placing samples on native timeline tracks.
func QualifyGoMonotonic(out *GoTrace, goos, version string) error {
	if goos != "linux" || version != "go1.27.1" {
		return fmt.Errorf("unsupported Go trace producer clock")
	}
	if len(out.ClockSnapshots) == 0 {
		return fmt.Errorf("go trace clock snapshot unavailable")
	}
	for _, c := range out.ClockSnapshots {
		if c.Trace == 0 || c.Monotonic == 0 {
			return fmt.Errorf("invalid Go trace clock snapshot")
		}
		var skew uint64
		if c.Trace > c.Monotonic {
			skew = c.Trace - c.Monotonic
		} else {
			skew = c.Monotonic - c.Trace
		}
		if skew > out.SnapshotReadSkewNs {
			out.SnapshotReadSkewNs = skew
		}
	}

	for i := range out.Samples {
		r := &out.Samples[i]
		r.Time = r.TraceTime
		r.Clock = "monotonic"
		r.Status = "native_clock_verification_required"
	}
	out.ClockStatus = "linux_monotonic"
	out.ClockUncertaintyNs = 64
	return nil
}
