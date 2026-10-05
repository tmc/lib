//go:build linux

package cudaschedule

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

func recordedSignalProcess(proc recordedProcess, signal syscall.Signal) error {
	pid, err := recordedSignalTarget(proc)
	if err != nil {
		return err
	}
	if pid == os.Getpid() || proc.Parent != os.Getpid() {
		return fmt.Errorf("recorded helper: refuse own or non-child process signal")
	}
	current, state, err := recordedProcStat(pid)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if current.Start != proc.Start || current.Group != proc.Group || current.Parent != proc.Parent {
		return fmt.Errorf("recorded helper: refuse changed process identity")
	}
	if state == 'Z' || state == 'X' {
		return nil
	}
	if !validPTXDigest(proc.Executable) {
		return fmt.Errorf("recorded helper: refuse signal without retained executable identity")
	}
	image, err := recordedIdentify(pid)
	if err != nil {
		return err
	}
	if image.Start != proc.Start || image.Group != proc.Group || image.Executable != proc.Executable {
		return fmt.Errorf("recorded helper: refuse changed executable identity")
	}
	if err := syscall.Kill(pid, signal); err != nil && err != syscall.ESRCH {
		return err
	}
	return nil
}

func recordedAuthenticatedPeer(conn *net.UnixConn) (recordedProcess, error) {
	var credential *syscall.Ucred
	raw, err := conn.SyscallConn()
	if err != nil {
		return recordedProcess{}, fmt.Errorf("recorded helper: peer connection: %w", err)
	}
	var credentialErr error
	if err := raw.Control(func(fd uintptr) {
		credential, credentialErr = syscall.GetsockoptUcred(int(fd), syscall.SOL_SOCKET, syscall.SO_PEERCRED)
	}); err != nil {
		return recordedProcess{}, fmt.Errorf("recorded helper: peer credentials: %w", err)
	}
	if credentialErr != nil || credential == nil || credential.Uid != uint32(os.Getuid()) {
		return recordedProcess{}, fmt.Errorf("recorded helper: peer uid unavailable or different")
	}
	proc, err := recordedIdentify(int(credential.Pid))
	if err != nil {
		return recordedProcess{}, err
	}
	return proc, nil
}

func recordedPeer(conn *net.UnixConn, parent, group int, image string) (recordedProcess, error) {
	proc, err := recordedAuthenticatedPeer(conn)
	if err != nil {
		return recordedProcess{}, err
	}
	if proc.Parent != parent || proc.Group != group || proc.Executable != image {
		return recordedProcess{}, fmt.Errorf("recorded helper: peer ancestry, group or image mismatch")
	}
	return proc, nil
}

func recordedProcStat(pid int) (recordedProcess, byte, error) {
	var proc recordedProcess
	if pid <= 1 {
		return proc, 0, fmt.Errorf("recorded helper: invalid pid")
	}
	data, err := recordedReadFile(fmt.Sprintf("/proc/%d/stat", pid), 8192)
	if err != nil {
		return proc, 0, err
	}
	// comm can contain spaces and parentheses. The final ')' ends field 2.
	end := strings.LastIndexByte(string(data), ')')
	if end < 0 {
		return proc, 0, fmt.Errorf("recorded helper: malformed process stat")
	}
	fields := strings.Fields(string(data[end+1:]))
	if len(fields) < 20 || len(fields[0]) != 1 {
		return proc, 0, fmt.Errorf("recorded helper: short process stat")
	}
	proc.PID = pid
	proc.Parent, err = strconv.Atoi(fields[1])
	if err != nil {
		return recordedProcess{}, 0, err
	}
	proc.Group, err = strconv.Atoi(fields[2])
	if err != nil || proc.Group < 0 {
		return recordedProcess{}, 0, fmt.Errorf("recorded helper: invalid process group")
	}
	proc.Start, err = strconv.ParseUint(fields[19], 10, 64)
	if err != nil {
		return recordedProcess{}, 0, fmt.Errorf("recorded helper: invalid process start")
	}
	return proc, fields[0][0], nil
}

func recordedIdentify(pid int) (recordedProcess, error) {
	before, state, err := recordedProcStat(pid)
	if err != nil {
		return recordedProcess{}, err
	}
	if state == 'Z' || state == 'X' || before.Group <= 0 || before.Start == 0 {
		return recordedProcess{}, fmt.Errorf("recorded helper: process %d already exited", pid)
	}
	file, err := os.Open(fmt.Sprintf("/proc/%d/exe", pid))
	if err != nil {
		return recordedProcess{}, fmt.Errorf("recorded helper: process executable: %w", err)
	}
	image, statErr := file.Stat()
	hash := sha256.New()
	n, copyErr := io.Copy(hash, io.LimitReader(file, (128<<20)+1))
	closeErr := file.Close()
	if statErr != nil || !image.Mode().IsRegular() || copyErr != nil || closeErr != nil || n > 128<<20 {
		return recordedProcess{}, fmt.Errorf("recorded helper: executable read failed or exceeds limit")
	}
	after, state, err := recordedProcStat(pid)
	if err != nil || state == 'Z' || state == 'X' || before != after {
		return recordedProcess{}, fmt.Errorf("recorded helper: process changed during identification")
	}
	currentImage, err := os.Stat(fmt.Sprintf("/proc/%d/exe", pid))
	if err != nil || !os.SameFile(image, currentImage) || image.Size() != currentImage.Size() || image.ModTime() != currentImage.ModTime() {
		return recordedProcess{}, fmt.Errorf("recorded helper: executable changed during identification")
	}
	before.Executable = hex.EncodeToString(hash.Sum(nil))
	return before, nil
}

func recordedAlive(proc recordedProcess) bool {
	current, state, err := recordedProcStat(proc.PID)
	return err == nil && state != 'Z' && state != 'X' && current.Start == proc.Start && current.Group == proc.Group
}

func recordedReadFile(path string, limit int64) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("recorded helper: open: %w", err)
	}
	data, readErr := io.ReadAll(io.LimitReader(f, limit+1))
	closeErr := f.Close()
	if readErr != nil {
		return nil, fmt.Errorf("recorded helper: read: %w", readErr)
	}
	if closeErr != nil {
		return nil, fmt.Errorf("recorded helper: close: %w", closeErr)
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("recorded helper: file exceeds limit")
	}
	return data, nil
}

// recordedPinFile refuses symlinks, devices and blocking FIFOs before reading.
// The returned bytes are the body whose digest was checked, suitable for an
// exclusive task-private copy. A path alone is not an executable identity.
func recordedPinFile(path, want string, limit int64) ([]byte, error) {
	if !filepath.IsAbs(path) || !validPTXDigest(want) || limit <= 0 || limit > 128<<20 {
		return nil, fmt.Errorf("recorded helper: invalid file pin")
	}
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_CLOEXEC|syscall.O_NOFOLLOW|syscall.O_NONBLOCK, 0)
	if err != nil {
		return nil, fmt.Errorf("recorded helper: open pinned file: %w", err)
	}
	f := os.NewFile(uintptr(fd), path)
	defer f.Close()
	before, err := f.Stat()
	if err != nil || !before.Mode().IsRegular() || before.Size() <= 0 || before.Size() > limit {
		return nil, fmt.Errorf("recorded helper: pinned file not bounded regular input")
	}
	data, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil || int64(len(data)) != before.Size() || int64(len(data)) > limit {
		return nil, fmt.Errorf("recorded helper: pinned file read failed or size changed")
	}
	after, err := f.Stat()
	if err != nil || !os.SameFile(before, after) || before.Size() != after.Size() || before.ModTime() != after.ModTime() {
		return nil, fmt.Errorf("recorded helper: pinned file changed during read")
	}
	sum := sha256.Sum256(data)
	if hex.EncodeToString(sum[:]) != want {
		return nil, fmt.Errorf("recorded helper: pinned file digest mismatch")
	}
	return data, nil
}

func recordedGroupMembers(group int) ([]int, error) {
	if group <= 1 || group == syscall.Getpgrp() {
		return nil, fmt.Errorf("recorded helper: refuse own or invalid group")
	}
	f, err := os.Open("/proc")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var members []int
	count := 0
	for {
		names, err := f.Readdirnames(256)
		for _, name := range names {
			count++
			if count > 1<<18 {
				return nil, fmt.Errorf("recorded helper: process inventory exceeds limit")
			}
			pid, parseErr := strconv.Atoi(name)
			if parseErr != nil || pid <= 1 {
				continue
			}
			proc, state, statErr := recordedProcStat(pid)
			if statErr != nil {
				// A disappearance during the scan is normal. Other failures do
				// not establish that an owned group has retired.
				if errors.Is(statErr, os.ErrNotExist) {
					continue
				}
				return nil, fmt.Errorf("recorded helper: inspect %s: %w", filepath.Join("/proc", name), statErr)
			}
			if proc.Group == group && state != 'Z' && state != 'X' {
				members = append(members, pid)
			}
		}
		if err == io.EOF {
			return members, nil
		}
		if err != nil {
			return nil, err
		}
	}
}

// recordedRetire is for groups previously retained while their exact leaders
// were authenticated. A successful return proves a live inventory is empty;
// it does not rely on the leader exiting or a signal being delivered.
func recordedRetire(ctx context.Context, groups []int) error {
	unique := make(map[int]bool)
	for _, group := range groups {
		if group <= 1 || group == syscall.Getpgrp() {
			return fmt.Errorf("recorded helper: refuse own or invalid group")
		}
		unique[group] = true
	}
	started := time.Now()
	killed := false
	for {
		live := false
		for group := range unique {
			members, err := recordedGroupMembers(group)
			if err != nil {
				return err
			}
			if len(members) == 0 {
				delete(unique, group)
				continue
			}
			live = true
			signal := syscall.SIGTERM
			if killed || time.Since(started) >= 100*time.Millisecond {
				signal = syscall.SIGKILL
				killed = true
			}
			if err := syscall.Kill(-group, signal); err != nil && err != syscall.ESRCH {
				return fmt.Errorf("recorded helper: retire group: %w", err)
			}
		}
		if !live {
			return nil
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("recorded helper: incomplete group cleanup: %w", ctx.Err())
		case <-time.After(10 * time.Millisecond):
		}
	}
}

// recordedReadRegular validates a nonblocking descriptor before reading it.
// It permits empty output and never reopens the mutable path after validation.
func recordedReadRegular(path string, limit int64) ([]byte, error) {
	if !filepath.IsAbs(path) || limit <= 0 || limit > 128<<20 {
		return nil, fmt.Errorf("recorded helper: invalid regular-file bound")
	}
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_NOFOLLOW|syscall.O_NONBLOCK|syscall.O_CLOEXEC, 0)
	if err != nil {
		return nil, err
	}
	f := os.NewFile(uintptr(fd), path)
	defer f.Close()
	before, err := f.Stat()
	if err != nil || !before.Mode().IsRegular() || before.Size() < 0 || before.Size() > limit {
		return nil, fmt.Errorf("recorded helper: not bounded regular file")
	}
	body, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil || int64(len(body)) != before.Size() || int64(len(body)) > limit {
		return nil, fmt.Errorf("recorded helper: incomplete regular-file read")
	}
	after, err := f.Stat()
	if err != nil || !os.SameFile(before, after) || before.Size() != after.Size() || before.ModTime() != after.ModTime() {
		return nil, fmt.Errorf("recorded helper: regular file changed")
	}
	return body, nil
}
