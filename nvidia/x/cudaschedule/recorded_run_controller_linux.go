//go:build linux

package cudaschedule

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"syscall"
	"time"
)

func recordedCopy(path, digest, destination string) error {
	body, err := recordedPinFile(path, digest, 128<<20)
	if err != nil {
		return err
	}
	_, err = recordedStore(destination, body, 0500)
	return err
}

func recordedVerifyLive(want recordedProcess) error {
	if _, err := recordedSignalTarget(want); err != nil {
		return err
	}
	actual, err := recordedIdentify(want.PID)
	if err != nil {
		return err
	}
	if actual != want {
		return fmt.Errorf("recorded run: changed process identity")
	}
	return nil
}

func recordedReadCustody(path, digest string, config recordedConfig, keeper recordedProcess, stage string) (RecordedCustody, error) {
	var c RecordedCustody
	if err := recordedLoadJSON(path, digest, &c); err != nil {
		return c, err
	}
	phase, creation := "holder-started", "unknown"
	if stage == "admit" {
		phase, creation = "supervisor-observed", "observed"
	}
	if stage == "worker-ready" {
		phase, creation = "worker-observed", "observed"
	}
	if c.Phase != phase || c.SupervisorCreation != creation {
		return c, fmt.Errorf("recorded run: custody phase mismatch")
	}
	if c.Controller != config.Controller || c.Keeper != keeper || c.Directory != config.Directory || c.Closed {
		return c, fmt.Errorf("recorded run: custody owner mismatch")
	}
	if c.Holder.Parent != keeper.PID || c.Holder.Group != c.Holder.PID {
		return c, fmt.Errorf("recorded run: holder topology mismatch")
	}
	if err := recordedVerifyLive(c.Holder); err != nil {
		return c, err
	}
	if stage != "holder" {
		if c.Supervisor.Parent != c.Holder.PID || c.Supervisor.Group != c.Supervisor.PID || c.Supervisor.Executable != config.HelperDigest {
			return c, fmt.Errorf("recorded run: supervisor topology mismatch")
		}
		if err := recordedVerifyLive(c.Supervisor); err != nil {
			return c, err
		}
		if err := recordedLockOwner(config, c.Holder); err != nil {
			return c, err
		}
	}
	if stage == "worker-ready" {
		if c.Worker.Parent != c.Supervisor.PID || c.Worker.Group != c.Supervisor.Group || c.Worker.PID == c.Supervisor.PID || c.Worker.Executable != config.WorkerDigest {
			return c, fmt.Errorf("recorded run: worker topology mismatch")
		}
		if err := recordedVerifyLive(c.Worker); err != nil {
			return c, err
		}
	}
	return c, nil
}

func runRecordedAttempt(ctx context.Context, seed uint64, ptx string, opts RecordedRunOptions, runtime time.Duration, pristine bool) (rec RunRecord, retErr error) {
	rec.ExpectedPTXDigest = digestPTX(ptx)
	rec.OracleAttached = opts.OracleAttached
	rec.PristineRun = pristine
	controller, err := recordedIdentify(os.Getpid())
	if err != nil {
		return rec, err
	}
	var nonceBytes [32]byte
	if _, err := rand.Read(nonceBytes[:]); err != nil {
		return rec, err
	}
	dir, err := os.MkdirTemp(opts.ArtifactDir, fmt.Sprintf("recorded-%d-", seed))
	if err != nil {
		return rec, err
	}
	config := recordedConfig{
		Version: recordedProtocolVersion, Nonce: hex.EncodeToString(nonceBytes[:]), Controller: controller, Directory: dir,
		Helper: filepath.Join(dir, "helper"), HelperDigest: opts.HelperSHA256,
		Worker: filepath.Join(dir, "worker"), WorkerDigest: opts.ExecutableSHA256,
		Args: append([]string{}, opts.Args...), WorkDir: opts.WorkDir,
		PTXPath: filepath.Join(dir, "module.ptx"), PTXDigest: rec.ExpectedPTXDigest,
		LockScript: filepath.Join(dir, "gpu-lock"), LockDigest: recordedLockDigest, LockDirectory: opts.LockDirectory, Label: opts.Label,
		Limits:   recordedLimits{Queue: opts.QueueTimeout, Admission: opts.AdmissionTimeout, Runtime: runtime, Cleanup: opts.CleanupTimeout, Lease: opts.LeaseLimit},
		Expected: opts.LeaseExpected, OutputLimit: opts.OutputLimit,
	}
	if err := config.validate(); err != nil {
		return rec, err
	}
	for _, input := range []struct{ source, digest, dest string }{
		{opts.Helper, config.HelperDigest, config.Helper}, {opts.Executable, config.WorkerDigest, config.Worker}, {opts.LockScript, config.LockDigest, config.LockScript},
	} {
		if err := recordedCopy(input.source, input.digest, input.dest); err != nil {
			return rec, err
		}
	}
	if _, err := recordedStore(config.PTXPath, []byte(ptx), 0400); err != nil {
		return rec, err
	}
	path := filepath.Join(dir, "config.json")
	digest, err := recordedStoreJSON(path, config)
	if err != nil {
		return rec, err
	}
	rec.PTXPath = config.PTXPath
	rec.StdoutPath = filepath.Join(dir, "worker.stdout")
	rec.StderrPath = filepath.Join(dir, "worker.stderr")
	total, err := config.Limits.validate()
	if err != nil {
		return rec, err
	}
	budget, cancel := context.WithTimeout(ctx, total)
	defer cancel()
	keeper, err := recordedStartKeeper(config.Helper, []string{"keeper", path, digest}, os.Environ(), config.WorkDir, config.OutputLimit)
	if keeper == nil {
		return rec, err
	}
	var groups = []int{keeper.identity.Group}
	rec.Custody = &RecordedCustody{Controller: controller, Keeper: keeper.identity, Directory: dir, Phase: "keeper-ready", SupervisorCreation: "not-started"}
	defer func() {
		// Closing the lifeline (deferred below after connection setup) asks
		// the keeper to retire its discovered groups. Do not kill it while
		// its cleanup is still running.
		grace, graceCancel := context.WithTimeout(context.Background(), config.Limits.Cleanup+config.Limits.Admission+50*time.Millisecond)
		for !keeper.reaped && grace.Err() == nil {
			if err := keeper.poll(); err != nil {
				retErr = errors.Join(retErr, err)
				break
			}
			if keeper.reaped {
				break
			}
			time.Sleep(5 * time.Millisecond)
		}
		graceCancel()
		cleanup, cleanupCancel := context.WithTimeout(context.Background(), config.Limits.Cleanup)
		defer cleanupCancel()
		cleanupErr := recordedRetire(cleanup, groups)
		for (!keeper.reaped || !keeper.outEOF || !keeper.errEOF) && cleanup.Err() == nil {
			if err := keeper.poll(); err != nil {
				cleanupErr = errors.Join(cleanupErr, err)
				break
			}
			if keeper.reaped && keeper.outEOF && keeper.errEOF {
				break
			}
			time.Sleep(5 * time.Millisecond)
		}
		cleanupErr = errors.Join(cleanupErr, keeper.closePipes())
		out, overflow := keeper.out.snapshot()
		errOut, errOverflow := keeper.errOut.snapshot()
		_, outErr := recordedStore(filepath.Join(dir, "keeper.stdout"), []byte(out), 0600)
		_, errErr := recordedStore(filepath.Join(dir, "keeper.stderr"), []byte(errOut), 0600)
		cleanupErr = errors.Join(cleanupErr, outErr, errErr)
		if !keeper.reaped || !keeper.outEOF || !keeper.errEOF || overflow || errOverflow {
			cleanupErr = errors.Join(cleanupErr, fmt.Errorf("recorded run: incomplete keeper status or drain"))
		}
		if cleanupErr != nil {
			rec.HarnessErr = recordedPacketError(errors.Join(retErr, cleanupErr))
		}
		if ctx.Err() != nil {
			retErr = ctx.Err()
		}
		if retErr != nil && rec.HarnessErr == "" {
			rec.HarnessErr = recordedPacketError(retErr)
		}
		// Controller backup retirement is necessary in addition to the keeper's
		// receipt. A failed cleanup cannot authorize health or another attempt.
		rec.Custody.Closed = rec.Custody.Closed && cleanupErr == nil
	}()
	if err != nil {
		rec.HarnessErr = recordedPacketError(err)
		return rec, nil
	}
	var conn *net.UnixConn
	for {
		if budget.Err() != nil {
			rec.HarnessErr = recordedPacketError(budget.Err())
			return rec, nil
		}
		conn, err = recordedDial(budget, config)
		if err == nil {
			break
		}
		if pollErr := keeper.poll(); pollErr != nil {
			rec.HarnessErr = recordedPacketError(pollErr)
			return rec, nil
		}
		if keeper.reaped {
			rec.HarnessErr = "keeper exited before readiness"
			return rec, nil
		}
		time.Sleep(5 * time.Millisecond)
	}
	defer conn.Close()
	peer, err := recordedKeeperPeer(conn, config)
	if err != nil || peer != keeper.identity {
		rec.HarnessErr = "keeper identity mismatch"
		return rec, nil
	}
	receive := func(kind string) (recordedMessage, error) { return recordedExpect(budget, conn, config, kind) }
	send := func(kind string) error { return recordedSend(budget, conn, recordedMessageOf(config, kind, "")) }
	if _, err := receive("ready"); err != nil {
		rec.HarnessErr = recordedPacketError(err)
		return rec, nil
	}
	if err := send("start"); err != nil {
		rec.HarnessErr = recordedPacketError(err)
		return rec, nil
	}
	msg, err := receive("holder")
	if err != nil {
		rec.HarnessErr = recordedPacketError(err)
		return rec, nil
	}
	custody, err := recordedReadCustody(filepath.Join(dir, "holder-ready.json"), msg.Digest, config, keeper.identity, "holder")
	if err != nil {
		rec.HarnessErr = recordedPacketError(err)
		return rec, nil
	}
	groups = append(groups, custody.Holder.Group)
	*rec.Custody = custody
	if err := send("held"); err != nil {
		rec.HarnessErr = recordedPacketError(err)
		return rec, nil
	}
	msg, err = receive("admit")
	if err != nil {
		rec.HarnessErr = recordedPacketError(err)
		return rec, nil
	}
	custody, err = recordedReadCustody(filepath.Join(dir, "admission.json"), msg.Digest, config, keeper.identity, "admit")
	if err != nil {
		rec.HarnessErr = recordedPacketError(err)
		return rec, nil
	}
	if custody.Holder != rec.Custody.Holder {
		rec.HarnessErr = "holder changed after retention"
		return rec, nil
	}
	groups = append(groups, custody.Supervisor.Group)
	*rec.Custody = custody
	if err := send("retained"); err != nil {
		rec.HarnessErr = recordedPacketError(err)
		return rec, nil
	}
	msg, err = receive("worker-ready")
	if err != nil {
		rec.HarnessErr = recordedPacketError(err)
		return rec, nil
	}
	custody, err = recordedReadCustody(filepath.Join(dir, "worker-ready.json"), msg.Digest, config, keeper.identity, "worker-ready")
	if err != nil {
		rec.HarnessErr = recordedPacketError(err)
		return rec, nil
	}
	if custody.Holder != rec.Custody.Holder || custody.Supervisor != rec.Custody.Supervisor {
		rec.HarnessErr = "admission owners changed"
		return rec, nil
	}
	*rec.Custody = custody
	if err := send("verified"); err != nil {
		rec.HarnessErr = recordedPacketError(err)
		return rec, nil
	}
	msg, err = receive("done")
	if err != nil {
		rec.HarnessErr = recordedPacketError(err)
		return rec, nil
	}
	var result recordedResult
	if err := recordedLoadJSON(filepath.Join(dir, "result.json"), msg.Digest, &result); err != nil {
		rec.HarnessErr = recordedPacketError(err)
		return rec, nil
	}
	expected := *rec.Custody
	actual := result.Custody
	if actual.Controller != expected.Controller || actual.Keeper != expected.Keeper || actual.Holder != expected.Holder || actual.Supervisor != expected.Supervisor || actual.Worker != expected.Worker || actual.Directory != dir || result.Outcome.Worker != expected.Worker {
		rec.HarnessErr = "final custody identity mismatch"
		return rec, nil
	}
	if actual.Phase != "worker-permitted" || actual.SupervisorCreation != "observed" {
		rec.HarnessErr = "missing final worker permission phase"
		return rec, nil
	}
	*rec.Custody = actual
	rec.HarnessErr = result.HarnessErr
	if result.Outcome.HarnessErr != "" {
		rec.HarnessErr = result.Outcome.HarnessErr
	}
	if !actual.Closed || !result.Outcome.Reaped || !result.Outcome.StdoutDrained || !result.Outcome.StderrDrained {
		rec.HarnessErr = "incomplete final worker custody"
		return rec, nil
	}
	status := syscall.WaitStatus(result.Outcome.Status)
	rec.Exited = status.Exited()
	rec.ExitCode = status.ExitStatus()
	rec.TimedOut = result.Outcome.TimedOut
	if status.Signaled() {
		rec.Signal = status.Signal().String()
	}
	stdout, err := recordedReadOutput(rec.StdoutPath, result.Outcome.StdoutSHA256, opts.OutputLimit)
	if err != nil {
		rec.HarnessErr = recordedPacketError(err)
		return rec, nil
	}
	stderr, err := recordedReadOutput(rec.StderrPath, result.Outcome.StderrSHA256, opts.OutputLimit)
	if err != nil {
		rec.HarnessErr = recordedPacketError(err)
		return rec, nil
	}
	rec.Stdout, rec.Stderr = string(stdout), string(stderr)
	rec.Stdout, rec.PTXDigest = takeLoadedPTXDigest(rec.Stdout)
	if rec.PTXDigest != "" {
		rec.PTXDigestSource = "runner_receipt"
	}
	cleanOutput, output, outputErr := takeOutputReceipt(rec.Stdout, actual.Worker.PID, rec.PTXDigest)
	if outputErr != nil {
		rec.HarnessErr = recordedPacketError(outputErr)
	} else {
		rec.Stdout, rec.Output = cleanOutput, output
	}
	if opts.RequireOutput && output == nil {
		rec.HarnessErr = "missing worker output receipt"
	}
	rec.Stdout, rec.OraclePID, rec.OraclePatchedModules, rec.OracleInstrumentedAccesses = takeOracleLiveness(rec.Stdout)
	if opts.OracleAttached && rec.OraclePID != actual.Worker.PID {
		rec.HarnessErr = "oracle receipt worker pid mismatch"
	}
	clean, identity, identityErr := takeMismatchIdentity(rec.Stdout)
	if identityErr != nil {
		rec.HarnessErr = recordedPacketError(identityErr)
	} else {
		rec.Stdout, rec.Mismatch = clean, identity
	}
	if rec.Mismatch != nil && rec.Mismatch.Policy != "exact-uint64" {
		rec.HarnessErr = "unsupported mismatch identity policy"
	}
	return rec, nil
}

// Empty output is valid, unlike executable/configuration pins. Its actual
// regular-file bytes still require the supplied digest and explicit limit.
func recordedReadOutput(path, digest string, limit int) ([]byte, error) {
	if !validPTXDigest(digest) || limit <= 0 {
		return nil, fmt.Errorf("recorded run: invalid output pin")
	}
	body, err := recordedReadRegular(path, int64(limit))
	if err != nil || digestPTX(string(body)) != digest {
		return nil, fmt.Errorf("recorded run: output digest or read mismatch")
	}
	return body, nil
}
