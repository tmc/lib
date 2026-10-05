//go:build linux

package cudaschedule

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"syscall"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
)

// These waits check the controller independently of protocol progress. An
// abandoned controller cannot leave the keeper waiting a whole queue budget.
func recordedLifeline(connections []*net.UnixConn) error {
	for _, conn := range connections {
		if conn == nil {
			continue
		}
		raw, err := conn.SyscallConn()
		if err != nil {
			return err
		}
		var probeErr error
		err = raw.Control(func(fd uintptr) {
			var b [1]byte
			n, _, readErr := syscall.Recvfrom(int(fd), b[:], syscall.MSG_PEEK|syscall.MSG_DONTWAIT)
			if readErr == syscall.EAGAIN || readErr == syscall.EWOULDBLOCK {
				return
			}
			if readErr != nil {
				probeErr = readErr
				return
			}
			if n == 0 {
				probeErr = fmt.Errorf("recorded helper: controller lifeline closed")
			}
		})
		if err != nil {
			return err
		}
		if probeErr != nil {
			return probeErr
		}
	}
	return nil
}

func recordedAcceptOwned(ctx context.Context, listener *net.UnixListener, controller recordedProcess, holder *recordedChild, lifeline ...*net.UnixConn) (*net.UnixConn, error) {
	for {
		if err := recordedMonitorHolder(holder); err != nil {
			return nil, err
		}
		if err := recordedLifeline(lifeline); err != nil {
			return nil, err
		}
		if !recordedAlive(controller) {
			return nil, fmt.Errorf("recorded helper: controller died")
		}
		slice, cancel := context.WithTimeout(ctx, 25*time.Millisecond)
		conn, err := recordedAccept(slice, listener)
		cancel()
		if err == nil {
			return conn, nil
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		if !errors.Is(err, context.DeadlineExceeded) {
			return nil, err
		}
	}
}

func recordedExpectOwned(ctx context.Context, conn *net.UnixConn, config recordedConfig, kind string, holder *recordedChild, lifeline ...*net.UnixConn) (recordedMessage, error) {
	for {
		if err := recordedMonitorHolder(holder); err != nil {
			return recordedMessage{}, err
		}
		if err := recordedLifeline(lifeline); err != nil {
			return recordedMessage{}, err
		}
		if !recordedAlive(config.Controller) {
			return recordedMessage{}, fmt.Errorf("recorded helper: controller died")
		}
		slice, cancel := context.WithTimeout(ctx, 25*time.Millisecond)
		msg, err := recordedExpect(slice, conn, config, kind)
		cancel()
		if err == nil {
			return msg, nil
		}
		if ctx.Err() != nil {
			return msg, ctx.Err()
		}
		if !errors.Is(err, context.DeadlineExceeded) {
			return msg, err
		}
	}
}

func recordedLockOwner(config recordedConfig, holder recordedProcess) error {
	body, err := recordedReadRegular(filepath.Join(config.LockDirectory, "owner"), 8192)
	if err != nil {
		return err
	}
	owner, err := cudalock.ParseOwner(body)
	host, hostErr := os.Hostname()
	if err != nil || hostErr != nil || owner.PID != holder.PID || owner.Host != host || !recordedAlive(holder) {
		return fmt.Errorf("recorded helper: actual lock owner mismatch")
	}
	return nil
}

func recordedKeeper(ctx context.Context, config recordedConfig, path, digest string) (retErr error) {
	self, err := recordedIdentify(os.Getpid())
	if err != nil {
		return err
	}
	if self.Parent != config.Controller.PID || self.Group != self.PID || self.Executable != config.HelperDigest || !recordedAlive(config.Controller) {
		return fmt.Errorf("recorded helper: keeper ancestry or image mismatch")
	}
	total, err := config.Limits.validate()
	if err != nil {
		return err
	}
	budget, cancel := context.WithTimeout(ctx, total)
	defer cancel()
	listener, err := net.ListenUnix("unixpacket", recordedEndpoint(config))
	if err != nil {
		return err
	}
	defer listener.Close()
	initial, initialCancel := context.WithTimeout(budget, config.Limits.Admission)
	controller, err := recordedAcceptOwned(initial, listener, config.Controller, nil)
	initialCancel()
	if err != nil {
		return err
	}
	defer controller.Close()
	peer, err := recordedPeer(controller, config.Controller.Parent, config.Controller.Group, config.Controller.Executable)
	if err != nil || peer != config.Controller {
		return fmt.Errorf("recorded helper: controller peer mismatch")
	}
	if err := recordedSend(budget, controller, recordedMessageOf(config, "ready", "")); err != nil {
		return err
	}
	if _, err := recordedExpectOwned(budget, controller, config, "start", nil); err != nil {
		return err
	}

	result := recordedResult{Custody: RecordedCustody{Controller: config.Controller, Keeper: self, Directory: config.Directory, Phase: "keeper-ready", SupervisorCreation: "not-started"}}
	var holder *recordedChild
	var supervisor *net.UnixConn
	var worker *net.UnixConn
	var groups []int
	// Cleanup remains independent of canceled operation contexts. The result
	// refuses incomplete group retirement, literal status or pipe drain.
	defer func() {
		if supervisor != nil {
			supervisor.Close()
		}
		if worker != nil {
			worker.Close()
		}
		cleanup, cleanupCancel := context.WithTimeout(context.Background(), config.Limits.Cleanup)
		defer cleanupCancel()
		cleanupErr := recordedRetire(cleanup, groups)
		if holder != nil {
			for !holder.reaped || !holder.outEOF || !holder.errEOF {
				if err := holder.poll(); err != nil {
					cleanupErr = errors.Join(cleanupErr, err)
					break
				}
				if cleanup.Err() != nil {
					cleanupErr = errors.Join(cleanupErr, cleanup.Err())
					break
				}
				if holder.reaped && holder.outEOF && holder.errEOF {
					break
				}
				time.Sleep(5 * time.Millisecond)
			}
			cleanupErr = errors.Join(cleanupErr, holder.closePipes())
			hout, overflow := holder.out.snapshot()
			herr, errOverflow := holder.errOut.snapshot()
			_, storeErr := recordedStore(filepath.Join(config.Directory, "holder.stdout"), []byte(hout), 0600)
			_, stderrErr := recordedStore(filepath.Join(config.Directory, "holder.stderr"), []byte(herr), 0600)
			cleanupErr = errors.Join(cleanupErr, storeErr, stderrErr)
			if overflow || errOverflow || !holder.reaped || !holder.outEOF || !holder.errEOF {
				cleanupErr = errors.Join(cleanupErr, fmt.Errorf("recorded helper: incomplete holder status or drain"))
			}
		}
		if holder != nil {
			body, ownerErr := recordedReadRegular(filepath.Join(config.LockDirectory, "owner"), 8192)
			if ownerErr == nil {
				owner, parseErr := cudalock.ParseOwner(body)
				if parseErr != nil || owner.PID == holder.identity.PID {
					cleanupErr = errors.Join(cleanupErr, fmt.Errorf("recorded helper: owned lease not released"))
				}
			} else if !errors.Is(ownerErr, os.ErrNotExist) {
				cleanupErr = errors.Join(cleanupErr, ownerErr)
			}
		}
		result.Custody.Closed = recordedClosedCustody(result.Custody, cleanupErr)
		if retErr != nil {
			result.HarnessErr = recordedPacketError(retErr)
		}
		if cleanupErr != nil {
			result.HarnessErr = recordedPacketError(errors.Join(retErr, cleanupErr))
		}
		result.Custody.WorkerReaped = result.Outcome.Reaped
		result.Custody.StdoutDrained = result.Outcome.StdoutDrained
		result.Custody.StderrDrained = result.Outcome.StderrDrained
		resultDigest, storeErr := recordedStoreJSON(filepath.Join(config.Directory, "result.json"), result)
		retErr = errors.Join(retErr, cleanupErr, storeErr)
		if storeErr == nil && recordedAlive(config.Controller) {
			sendCtx, sendCancel := context.WithTimeout(context.Background(), config.Limits.Cleanup)
			sendErr := recordedSend(sendCtx, controller, recordedMessageOf(config, "done", resultDigest))
			sendCancel()
			retErr = errors.Join(retErr, sendErr)
		}
	}()

	// Source-pinned gpu-lock creates H in its own group, then S in the script's
	// controlled job group. No inherited lease flags are cleared or forged.
	env := recordedHolderEnvironment(config)
	holder, err = recordedStartChild(config.LockScript, []string{config.Helper, "supervisor", path, digest}, env, config.WorkDir, config.OutputLimit)
	if holder != nil {
		result.Custody.Holder = holder.identity
		result.Custody.Phase = "holder-started"
		result.Custody.SupervisorCreation = "unknown"
		groups = append(groups, holder.identity.Group)
	}
	if err != nil {
		return err
	}
	holderDigest, err := recordedStoreJSON(filepath.Join(config.Directory, "holder-ready.json"), result.Custody)
	if err != nil {
		return err
	}
	if err := recordedSend(budget, controller, recordedMessageOf(config, "holder", holderDigest)); err != nil {
		return err
	}
	if _, err := recordedExpectOwned(budget, controller, config, "held", holder); err != nil {
		return err
	}
	queue, queueCancel := context.WithTimeout(budget, config.Limits.Queue+config.Limits.Admission)
	defer queueCancel()
	supervisor, err = recordedAcceptOwned(queue, listener, config.Controller, holder, controller)
	if err != nil {
		return err
	}
	s, err := recordedAuthenticatedPeer(supervisor)
	if err != nil {
		return err
	}
	if s.Parent != holder.identity.PID || s.Group != s.PID || s.Executable != config.HelperDigest {
		return fmt.Errorf("recorded helper: supervisor peer mismatch")
	}
	result.Custody.Supervisor = s
	result.Custody.Phase = "supervisor-observed"
	result.Custody.SupervisorCreation = "observed"
	groups = append(groups, s.Group)
	if err := recordedLockOwner(config, holder.identity); err != nil {
		return err
	}
	admission, admissionCancel := context.WithTimeout(budget, config.Limits.Admission)
	defer admissionCancel()
	if _, err := recordedExpectOwned(admission, supervisor, config, "admit", holder, controller); err != nil {
		return err
	}
	custodyDigest, err := recordedStoreJSON(filepath.Join(config.Directory, "admission.json"), result.Custody)
	if err != nil {
		return err
	}
	if err := recordedSend(admission, controller, recordedMessageOf(config, "admit", custodyDigest)); err != nil {
		return err
	}
	// Controller now retains both owned groups before the supervisor can exec.
	if _, err := recordedExpectOwned(admission, controller, config, "retained", holder); err != nil {
		return err
	}
	if err := recordedSend(admission, supervisor, recordedMessageOf(config, "exec", "")); err != nil {
		return err
	}
	worker, err = recordedAcceptOwned(admission, listener, config.Controller, holder, controller)
	if err != nil {
		return err
	}
	w, err := recordedPeer(worker, s.PID, s.Group, config.WorkerDigest)
	if err != nil || w.PID == s.PID {
		return fmt.Errorf("recorded helper: worker peer mismatch: %v", err)
	}
	result.Custody.Worker = w
	result.Custody.Phase = "worker-observed"
	if _, err := recordedExpectOwned(admission, worker, config, "worker-ready", holder, controller); err != nil {
		return err
	}
	if err := recordedLockOwner(config, holder.identity); err != nil {
		return err
	}
	workerDigest, err := recordedStoreJSON(filepath.Join(config.Directory, "worker-ready.json"), result.Custody)
	if err != nil {
		return err
	}
	if err := recordedSend(admission, controller, recordedMessageOf(config, "worker-ready", workerDigest)); err != nil {
		return err
	}
	if _, err := recordedExpectOwned(admission, controller, config, "verified", holder); err != nil {
		return err
	}
	if !recordedAlive(s) || !recordedAlive(w) {
		return fmt.Errorf("recorded helper: worker or supervisor died before permit")
	}
	if err := recordedLockOwner(config, holder.identity); err != nil {
		return err
	}
	if err := recordedSend(admission, worker, recordedMessageOf(config, "permit", "")); err != nil {
		return err
	}
	result.Custody.Phase = "worker-permitted"
	if err := recordedSend(admission, supervisor, recordedMessageOf(config, "permit", "")); err != nil {
		return err
	}

	runtime, runtimeCancel := context.WithTimeout(budget, config.Limits.Runtime+config.Limits.Cleanup)
	defer runtimeCancel()
	msg, err := recordedExpectOwned(runtime, supervisor, config, "done", holder, controller)
	if err != nil {
		return err
	}
	if err := recordedLoadJSON(filepath.Join(config.Directory, "worker-outcome.json"), msg.Digest, &result.Outcome); err != nil {
		return err
	}
	if result.Outcome.Worker != w || !result.Outcome.Reaped || !result.Outcome.StdoutDrained || !result.Outcome.StderrDrained {
		return fmt.Errorf("recorded helper: missing literal worker status or drain")
	}
	// H must independently wait and finish. Its status never replaces W's.
	for !holder.reaped {
		if err := holder.poll(); err != nil {
			return err
		}
		if runtime.Err() != nil {
			return runtime.Err()
		}
		if !recordedAlive(config.Controller) {
			return fmt.Errorf("recorded helper: controller died")
		}
		time.Sleep(5 * time.Millisecond)
	}
	if !holder.status.Exited() || holder.status.ExitStatus() != 0 {
		return fmt.Errorf("recorded helper: holder interrupted or failed")
	}
	return nil
}

// Holder output is drained while protocol/admission waits are in progress,
// so a full diagnostic pipe cannot stop a valid holder from reaching S.
func recordedMonitorHolder(holder *recordedChild) error {
	if holder == nil {
		return nil
	}
	if err := holder.poll(); err != nil {
		return err
	}
	if holder.reaped && (!holder.status.Exited() || holder.status.ExitStatus() != 0) {
		return fmt.Errorf("recorded helper: holder interrupted before worker outcome")
	}

	_, outOverflow := holder.out.snapshot()
	_, errOverflow := holder.errOut.snapshot()
	if outOverflow || errOverflow {
		return fmt.Errorf("recorded helper: holder output limit exceeded")
	}
	return nil
}

func recordedHolderEnvironment(config recordedConfig) []string {
	return recordedEnvironment(os.Environ(),
		"GPU_LOCKDIR="+config.LockDirectory,
		"GPU_LOCK_MAXWAIT="+recordedSeconds(config.Limits.Queue),
		"GPU_LOCK_EXPECT="+recordedSeconds(config.Expected),
		"GPU_LOCK_LIMIT="+recordedSeconds(config.Limits.Lease),
		"GPU_LOCK_LABEL="+config.Label,
		"GPU_LOCK_POLL=1", "GPU_LOCK_KILLWAIT=1")
}

// H can create a separate S group before its peer connects. Empty retained
// H-group inventory therefore does not prove retirement of an unknown S.
func recordedClosedCustody(c RecordedCustody, cleanupErr error) bool {
	if cleanupErr != nil {
		return false
	}
	if c.Holder.PID <= 1 {
		return c.Phase == "keeper-ready" && c.SupervisorCreation == "not-started" && c.Supervisor.PID == 0
	}
	return c.SupervisorCreation == "observed" && c.Supervisor.PID > 1 && c.Supervisor.Group > 1
}
