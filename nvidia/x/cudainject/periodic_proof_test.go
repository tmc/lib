package cudainject_test

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
	"sync"

	"github.com/tmc/lib/nvidia/x/cudainject"
)

const periodicKernel = "lane87_periodic"

//go:embed testdata/periodic_worker.c
var periodicWorkerSource []byte

// These are fixture predicates, not a second capture controller.
func periodicEvidence(recovery cudainject.Recovery, pid int, final bool) error {
	if pid <= 1 || len(recovery.Records) == 0 || recovery.Truncated || !recovery.Complete || recovery.Stats.Gaps != 0 || recovery.Stats.Dropped != 0 {
		return errors.New("incomplete periodic evidence")
	}
	identity := recovery.Records[0].Identity
	if identity.Process != uint64(pid) || identity.Generation == 0 {
		return errors.New("periodic worker identity differs")
	}
	for _, b := range identity.Session {
		if b != 1 {
			return errors.New("periodic session differs")
		}
	}
	if recovery.Records[0].Kind != cudainject.KindBootstrap {
		return errors.New("periodic bootstrap missing")
	}
	capability, finalized := false, false
	capabilities, finalizations := 0, 0
	kernels := make(map[uint64]string)
	for i, r := range recovery.Records {
		if r.Identity != identity || r.Sequence != uint64(i+1) || r.Dropped != 0 {
			return errors.New("periodic record identity or loss differs")
		}
		switch r.Kind {
		case cudainject.KindFailure:
			return errors.New("periodic producer failure")
		case cudainject.KindCapability:
			capabilities++
			var c struct {
				FlushInterval int  `json:"flush_interval_ms"`
				DriverAPI     bool `json:"driver_api"`
				APIRecords    bool `json:"api_records"`
			}
			if err := json.Unmarshal(r.Payload, &c); err != nil {
				return err
			}
			capability = c.FlushInterval == 5 && c.DriverAPI && c.APIRecords
		case cudainject.KindActivity:
			var activity struct {
				Subtype, Name string
				Correlation   uint64
			}
			if err := json.Unmarshal(r.Payload, &activity); err != nil {
				return err
			}
			if activity.Subtype == "kernel" {
				kernels[activity.Correlation] = activity.Name
			}
		case cudainject.KindFinalized:
			finalizations++
			if i != len(recovery.Records)-1 {
				return errors.New("periodic records follow finalization")
			}
			var f struct {
				State         string `json:"state"`
				Dropped       uint64 `json:"dropped"`
				NativeDropped uint64 `json:"native_dropped"`
				QueueDropped  uint64 `json:"queue_dropped"`
			}
			if err := json.Unmarshal(r.Payload, &f); err != nil {
				return err
			}
			if f.State != "collected" || f.Dropped != 0 || f.NativeDropped != 0 || f.QueueDropped != 0 {
				return errors.New("periodic final state or loss differs")
			}
			finalized = true
		}
	}
	if !capability || capabilities != 1 || finalized != final || finalizations > 1 {
		return errors.New("periodic capability or finalization differs")
	}
	joins, err := cudainject.JoinCorrelations(recovery.Records)
	if err != nil {
		return err
	}
	if len(joins.Matched) != 32 || len(joins.Ambiguous) != 0 || len(joins.MissingAPI) != 0 {
		return errors.New("periodic kernel population differs")
	}
	for _, join := range joins.Matched {
		// The existing tracker retains the runtime's optional KernelName. CUPTI
		// API payloads omit it, so bind the name from the actual kernel activity.
		if kernels[join.CorrelationID] != periodicKernel || join.HostThreadID == 0 || join.CorrelationID == 0 {
			return errors.New("periodic kernel or launch attribution differs")
		}
	}
	return nil
}

func releasePeriodicWorker(input io.WriteCloser, cancel context.CancelFunc, wait func() error) error {
	n, writeErr := input.Write([]byte("x"))
	if n != 1 && writeErr == nil {
		writeErr = io.ErrShortWrite
	}
	closeErr := input.Close()
	if writeErr != nil || closeErr != nil {
		cancel()
	}
	workerErr := wait()
	return errors.Join(writeErr, closeErr, workerErr)
}

type periodicWorkerIdentity struct {
	Driver int
	UUID   [16]byte
	PID    int
}

func periodicWorkerOutput(output []byte, pid int, released bool) (periodicWorkerIdentity, error) {
	var result periodicWorkerIdentity
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	want := 2
	if released {
		want = 3
	}
	if len(lines) != want || lines[1] != "waiting-before-exit" || released && lines[2] != "released" {
		return result, errors.New("periodic worker handshake differs")
	}
	fields := strings.Fields(lines[0])
	if len(fields) != 3 {
		return result, errors.New("periodic worker identity missing")
	}
	if !strings.HasPrefix(fields[0], "driver=") || !strings.HasPrefix(fields[1], "uuid=") || !strings.HasPrefix(fields[2], "pid=") {
		return result, errors.New("periodic worker identity malformed")
	}
	var err error
	result.Driver, err = strconv.Atoi(strings.TrimPrefix(fields[0], "driver="))
	if err != nil || result.Driver <= 0 {
		return result, errors.New("periodic driver version missing")
	}
	result.PID, err = strconv.Atoi(strings.TrimPrefix(fields[2], "pid="))
	if err != nil || pid <= 1 || result.PID != pid {
		return result, errors.New("periodic stdout PID differs")
	}
	uuid, err := hex.DecodeString(strings.TrimPrefix(fields[1], "uuid="))
	if err != nil || len(uuid) != 16 {
		return result, errors.New("periodic UUID malformed")
	}
	copy(result.UUID[:], uuid)
	if result.UUID == [16]byte{} {
		return result, errors.New("periodic UUID missing")
	}
	return result, nil
}

type periodicOutput struct {
	mu        sync.Mutex
	buffer    bytes.Buffer
	remaining int
}

func (w *periodicOutput) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if len(p) > w.remaining {
		return 0, errors.New("periodic output exceeds limit")
	}
	n, err := w.buffer.Write(p)
	w.remaining -= n
	return n, err
}
func (w *periodicOutput) Bytes() []byte {
	w.mu.Lock()
	defer w.mu.Unlock()
	return bytes.Clone(w.buffer.Bytes())
}

func retainPeriodicOutput(write func(string, []byte) error, stdout, stderr []byte) error {
	return errors.Join(write("worker.stdout", stdout), write("worker.stderr", stderr))
}

func finishPeriodicWorker(found bool, pid int, input io.WriteCloser, cancel context.CancelFunc, wait func() error, stdout, stderr *periodicOutput, write func(string, []byte) error) (periodicWorkerIdentity, error) {
	releaseErr := releasePeriodicWorker(input, cancel, wait)
	retainErr := retainPeriodicOutput(write, stdout.Bytes(), stderr.Bytes())
	identity, outputErr := periodicWorkerOutput(stdout.Bytes(), pid, true)
	var evidenceErr error
	if !found {
		evidenceErr = errors.New("periodic before-exit evidence missing")
	}
	return identity, errors.Join(evidenceErr, releaseErr, retainErr, outputErr)
}

func periodicHeader(pid int) string {
	return fmt.Sprintf("driver=13000 uuid=01010101010101010101010101010101 pid=%d\nwaiting-before-exit\n", pid)
}
