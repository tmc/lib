//go:build linux

package cudaschedule

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
)

const recordedConfigEnv = "CUDASCHEDULE_RECORDED_CONFIG"
const recordedConfigDigestEnv = "CUDASCHEDULE_RECORDED_CONFIG_SHA256"

type recordedOutcome struct {
	Worker        recordedProcess `json:"worker"`
	Status        int             `json:"status"`
	Reaped        bool            `json:"reaped"`
	TimedOut      bool            `json:"timed_out"`
	HarnessErr    string          `json:"harness_error"`
	StdoutSHA256  string          `json:"stdout_sha256"`
	StderrSHA256  string          `json:"stderr_sha256"`
	StdoutDrained bool            `json:"stdout_drained"`
	StderrDrained bool            `json:"stderr_drained"`
}

type recordedResult struct {
	Custody    RecordedCustody `json:"custody"`
	Outcome    recordedOutcome `json:"outcome"`
	HarnessErr string          `json:"harness_error"`
}

func recordedStore(path string, body []byte, mode os.FileMode) (string, error) {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, mode)
	if err != nil {
		return "", err
	}
	n, writeErr := f.Write(body)
	closeErr := f.Close()
	if writeErr != nil {
		return "", writeErr
	}
	if n != len(body) {
		return "", fmt.Errorf("recorded helper: incomplete artifact write")
	}
	if closeErr != nil {
		return "", closeErr
	}
	return digestPTX(string(body)), nil
}

func recordedStoreJSON(path string, value any) (string, error) {
	body, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	if len(body) > recordedMessageLimit {
		return "", fmt.Errorf("recorded helper: receipt exceeds limit")
	}
	return recordedStore(path, body, 0600)
}

func recordedLoadJSON(path, digest string, value any) error {
	body, err := recordedPinFile(path, digest, recordedMessageLimit)
	if err != nil {
		return err
	}
	d := json.NewDecoder(bytes.NewReader(body))
	d.DisallowUnknownFields()
	if err := d.Decode(value); err != nil {
		return err
	}
	canonical, err := json.Marshal(value)
	if err != nil || !bytes.Equal(body, canonical) {
		return fmt.Errorf("recorded helper: noncanonical receipt")
	}
	return nil
}

func recordedLoadConfig(path, digest string) (recordedConfig, error) {
	body, err := recordedPinFile(path, digest, recordedMessageLimit)
	if err != nil {
		return recordedConfig{}, err
	}
	return readRecordedConfig(body)
}

func recordedEndpoint(config recordedConfig) *net.UnixAddr {
	return &net.UnixAddr{Name: "@cudarecorded-" + config.Nonce, Net: "unixpacket"}
}

func recordedDial(ctx context.Context, config recordedConfig) (*net.UnixConn, error) {
	var dialer net.Dialer
	conn, err := dialer.DialContext(ctx, "unixpacket", recordedEndpoint(config).Name)
	if err != nil {
		return nil, err
	}
	unix, ok := conn.(*net.UnixConn)
	if !ok {
		conn.Close()
		return nil, fmt.Errorf("recorded helper: non-Unix connection")
	}
	return unix, nil
}

func recordedKeeperPeer(conn *net.UnixConn, config recordedConfig) (recordedProcess, error) {
	peer, err := recordedAuthenticatedPeer(conn)
	if err != nil {
		return recordedProcess{}, err
	}
	if peer.Parent != config.Controller.PID || peer.Group != peer.PID || peer.Executable != config.HelperDigest {
		return recordedProcess{}, fmt.Errorf("recorded helper: keeper peer mismatch")
	}
	return peer, nil
}

func recordedMessageOf(config recordedConfig, kind, digest string) recordedMessage {
	return recordedMessage{Version: recordedProtocolVersion, Nonce: config.Nonce, Kind: kind, Digest: digest}
}

func recordedExpect(ctx context.Context, conn *net.UnixConn, config recordedConfig, kind string) (recordedMessage, error) {
	msg, err := recordedReceive(ctx, conn, config.Nonce)
	if err != nil {
		return msg, err
	}
	if msg.Kind != kind {
		return msg, fmt.Errorf("recorded helper: expected %s, got %s", kind, msg.Kind)
	}
	return msg, nil
}

func runRecordedHelperMode(ctx context.Context, mode, path, digest string) error {
	config, err := recordedLoadConfig(path, digest)
	if err != nil {
		return err
	}
	switch mode {
	case "keeper":
		if err := recordedOutsideLease(); err != nil {
			return err
		}
		return recordedKeeper(ctx, config, path, digest)
	case "supervisor":
		return recordedSupervisor(ctx, config, path, digest)
	}
	return fmt.Errorf("recorded helper: unknown role")
}

func runRecordedWorkerReady(ctx context.Context) error {
	path, digest := os.Getenv(recordedConfigEnv), os.Getenv(recordedConfigDigestEnv)
	config, err := recordedLoadConfig(path, digest)
	if err != nil {
		return err
	}
	if !recordedAlive(config.Controller) {
		return fmt.Errorf("recorded helper: controller not alive")
	}
	owner, err := cudalock.Inherited(os.Getenv)
	if err != nil {
		return fmt.Errorf("recorded helper: worker lease: %w", err)
	}
	parent, _, err := recordedProcStat(os.Getppid())
	if err != nil || parent.Parent != owner.PID || parent.Group != parent.PID {
		return fmt.Errorf("recorded helper: worker supervisor ancestry mismatch")
	}
	bounded, cancel := context.WithTimeout(ctx, config.Limits.Admission)
	defer cancel()
	conn, err := recordedDial(bounded, config)
	if err != nil {
		return err
	}
	defer conn.Close()
	if _, err := recordedKeeperPeer(conn, config); err != nil {
		return err
	}
	if err := recordedSend(bounded, conn, recordedMessageOf(config, "worker-ready", "")); err != nil {
		return err
	}
	_, err = recordedExpect(bounded, conn, config, "permit")
	return err
}

func recordedSupervisor(ctx context.Context, config recordedConfig, path, digest string) error {
	owner, err := cudalock.Inherited(os.Getenv)
	if err != nil {
		return fmt.Errorf("recorded helper: supervisor lease: %w", err)
	}
	self, err := recordedIdentify(os.Getpid())
	if err != nil {
		return err
	}
	if self.Parent != owner.PID || self.Group != self.PID || self.Executable != config.HelperDigest {
		return fmt.Errorf("recorded helper: supervisor ancestry or image mismatch")
	}
	budget, cancel := context.WithTimeout(ctx, config.Limits.Admission+config.Limits.Runtime+config.Limits.Cleanup)
	defer cancel()
	conn, err := recordedDial(budget, config)
	if err != nil {
		return err
	}
	defer conn.Close()
	keeper, err := recordedKeeperPeer(conn, config)
	if err != nil {
		return err
	}
	if err := recordedSend(budget, conn, recordedMessageOf(config, "admit", "")); err != nil {
		return err
	}
	if _, err := recordedExpect(budget, conn, config, "exec"); err != nil {
		return err
	}
	env := recordedEnvironment(os.Environ(), recordedConfigEnv+"="+path, recordedConfigDigestEnv+"="+digest)
	args := append(append([]string(nil), config.Args...), config.PTXPath)
	child, startErr := recordedStartWorker(config.Worker, args, env, config.WorkDir, config.OutputLimit)
	if child == nil {
		return startErr
	}
	outcome := recordedOutcome{Worker: child.identity}
	if startErr != nil {
		outcome.HarnessErr = recordedPacketError(startErr)
	}
	// Keeper authenticates actual W and controller final ACK before sending
	// permit to W and notifying S. No guardian status substitutes for W status.
	admission, admissionCancel := context.WithTimeout(budget, config.Limits.Admission)
	_, permitErr := recordedExpect(admission, conn, config, "permit")
	admissionCancel()
	if permitErr != nil {
		outcome.HarnessErr = recordedPacketError(permitErr)
	}
	runtimeEnd := time.Now().Add(config.Limits.Runtime)
	stop := func(reason error) {
		if outcome.HarnessErr == "" && reason != nil {
			outcome.HarnessErr = recordedPacketError(reason)
		}
		if err := child.signal(syscall.SIGKILL); err != nil {
			outcome.HarnessErr = recordedPacketError(err)
		}
	}
	if outcome.HarnessErr != "" {
		stop(nil)
	}
	cleanupEnd := time.Time{}
	for {
		if err := child.poll(); err != nil {
			stop(err)
		}
		_, overflow := child.out.snapshot()
		_, errOverflow := child.errOut.snapshot()
		if overflow || errOverflow {
			stop(fmt.Errorf("recorded helper: output limit exceeded"))
		}
		if !recordedAlive(config.Controller) || !recordedAlive(keeper) {
			stop(fmt.Errorf("recorded helper: controller or keeper died"))
		}
		if err := budget.Err(); err != nil {
			stop(err)
		}
		if !child.reaped && time.Now().After(runtimeEnd) {
			outcome.TimedOut = true
			stop(nil)
		}
		if child.reaped {
			if cleanupEnd.IsZero() {
				cleanupEnd = time.Now().Add(config.Limits.Cleanup)
			}
			if child.outEOF && child.errEOF {
				break
			}
		}
		if !cleanupEnd.IsZero() && time.Now().After(cleanupEnd) {
			break
		}
		if budget.Err() != nil {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	outcome.Worker = child.identity
	outcome.Reaped = child.reaped
	outcome.Status = int(child.status)
	outcome.StdoutDrained = child.outEOF
	outcome.StderrDrained = child.errEOF
	out, outOverflow := child.out.snapshot()
	errOut, errOverflow := child.errOut.snapshot()
	if !child.reaped || !child.outEOF || !child.errEOF || outOverflow || errOverflow {
		outcome.HarnessErr = "recorded helper: incomplete worker status or drain"
	}
	if err := child.closePipes(); err != nil {
		outcome.HarnessErr = recordedPacketError(err)
	}
	outcome.StdoutSHA256, err = recordedStore(filepath.Join(config.Directory, "worker.stdout"), []byte(out), 0600)
	if err != nil {
		return err
	}
	outcome.StderrSHA256, err = recordedStore(filepath.Join(config.Directory, "worker.stderr"), []byte(errOut), 0600)
	if err != nil {
		return err
	}
	outcomeDigest, err := recordedStoreJSON(filepath.Join(config.Directory, "worker-outcome.json"), outcome)
	if err != nil {
		return err
	}
	finish, finishCancel := context.WithTimeout(ctx, config.Limits.Cleanup)
	defer finishCancel()
	if !recordedAlive(keeper) {
		return fmt.Errorf("recorded helper: keeper lost before outcome transfer")
	}
	return recordedSend(finish, conn, recordedMessageOf(config, "done", outcomeDigest))
}

func recordedSeconds(d time.Duration) string {
	return strconv.FormatInt(int64(d/time.Second), 10)
}

// Replace explicit protocol/lock settings without duplicate environment keys.
// Inherited lease/refusal flags are not among the overrides.
func recordedEnvironment(base []string, settings ...string) []string {
	keys := make(map[string]bool)
	for _, setting := range settings {
		key, _, _ := strings.Cut(setting, "=")
		keys[key] = true
	}
	out := make([]string, 0, len(base)+len(settings))
	for _, entry := range base {
		key, _, _ := strings.Cut(entry, "=")
		if !keys[key] {
			out = append(out, entry)
		}
	}
	return append(out, settings...)
}
