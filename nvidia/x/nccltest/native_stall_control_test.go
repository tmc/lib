//go:build linux && native

package nccltest

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/ncclinspector"
	"github.com/tmc/lib/nvidia/x/ncclras"
	"golang.org/x/sys/unix"
)

const stallBytes = 65536

var errStallLogOverflow = errors.New("nccl stall: child log exceeds limit")

// stallLog preserves the raw prefix and cancels the control on any write failure.
type stallLog struct {
	mu        sync.Mutex
	w         io.Writer
	remaining int
	cancel    context.CancelFunc
	err       error
}

func (w *stallLog) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.err != nil {
		return 0, w.err
	}
	data := p
	if len(data) > w.remaining {
		data = data[:w.remaining]
	}
	n, err := w.w.Write(data)
	w.remaining -= n
	if err == nil && n != len(data) {
		err = io.ErrShortWrite
	}
	if err == nil && len(p) > len(data) {
		err = errStallLogOverflow
	}
	if err != nil {
		w.err = err
		w.cancel()
	}
	return n, err
}

func (w *stallLog) Err() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.err
}

type stallChild struct {
	done chan struct{}
	err  error
}

func stallWatch(command *exec.Cmd, cancel context.CancelFunc) *stallChild {
	child := &stallChild{done: make(chan struct{})}
	go func() {
		child.err = command.Wait()
		close(child.done)
		if child.err != nil {
			cancel()
		}
	}()
	return child
}

type stallBootstrap struct {
	Version       int
	RunID, Source string
	UniqueID      nccl.UniqueId
	Devices       [2]int
	DelayNS       int64
}

type stallStart struct {
	Hash        string
	NotBeforeNS int64
}

type stallRankReceipt struct {
	Version                                            int
	RunID, Source, BootID, Hostname, UUID              string
	Rank, PID, TID, Device                             int
	NCCLVersion                                        int32
	ProcessStartTicks                                  uint64
	CommunicatorHash                                   string
	BarrierNS, DelayStartNS, DelayEndNS                int64
	SubmitStartNS, SubmitEndNS, SyncStartNS, SyncEndNS int64
	Comparisons, Wrong                                 int
	OutputSHA256                                       string
	InputSHA256                                        string
	NCCLLibraryPath, NCCLLibrarySHA256                 string
}

// TestNativeTwoRankStallControl runs exactly a normal and delayed two-process
// numerical control. It is not a GPU timeline, diagnosis or overhead receipt.
// Root must freeze and admit this test explicitly under an outer watchdog.
func TestNativeTwoRankStallControl(t *testing.T) {
	if os.Getenv("NCCLSTALL_NATIVE") != "1" {
		t.Skip("explicit native admission required")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatal(err)
	}
	source := os.Getenv("NCCLSTALL_SOURCE")
	if raw, err := hex.DecodeString(source); err != nil || len(raw) != 20 {
		t.Fatal("exact source revision required")
	}
	parts := strings.Split(os.Getenv("NCCLSTALL_DEVICES"), ",")
	if len(parts) != 2 {
		t.Fatal("two explicit device ordinals required")
	}
	var devices [2]int
	for rank, part := range parts {
		ordinal, err := strconv.Atoi(part)
		if err != nil || ordinal < 0 {
			t.Fatal("invalid device")
		}
		devices[rank] = ordinal
	}
	if devices[0] == devices[1] {
		t.Fatal("distinct devices required")
	}
	address := os.Getenv("NCCLSTALL_RAS_ADDR")
	host, port, err := net.SplitHostPort(address)
	n, e := strconv.Atoi(port)
	if err != nil || host != "127.0.0.1" || e != nil || n < 1024 || n > 65535 {
		t.Fatal("explicit loopback RAS port required")
	}
	out := os.Getenv("NCCLSTALL_OUTPUT")
	if !filepath.IsAbs(out) || filepath.Clean(out) != out {
		t.Fatal("absolute fresh output directory required")
	}
	if err := os.Mkdir(out, 0700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("NCCL_RAS_ENABLE", "1")
	t.Setenv("NCCL_RAS_ADDR", address)
	t.Setenv("NCCL_INSPECTOR_ENABLE", "0")
	if _, _, err := stallLoadNCCL(); err != nil {
		t.Fatal(err)
	}
	for _, control := range []struct {
		name  string
		delay time.Duration
	}{{"normal", 0}, {"delayed", 20 * time.Millisecond}} {
		dir := filepath.Join(out, control.name)
		if err := os.Mkdir(dir, 0700); err != nil {
			t.Fatal(err)
		}
		// Never kill an unrelated listener to make the test run.
		listener, err := net.Listen("tcp", address)
		if err != nil {
			t.Fatal(err)
		}
		if err := listener.Close(); err != nil {
			t.Fatal(err)
		}
		id, err := nccl.GetUniqueId()
		if err != nil {
			t.Fatal(err)
		}
		var nonce [16]byte
		if _, err := rand.Read(nonce[:]); err != nil {
			t.Fatal(err)
		}
		boot := stallBootstrap{Version: 1, RunID: hex.EncodeToString(nonce[:]), Source: source, UniqueID: id, Devices: devices, DelayNS: int64(control.delay)}
		if err := stallWrite(filepath.Join(dir, "bootstrap.json"), boot); err != nil {
			t.Fatal(err)
		}
		if err := stallControl(dir, address, boot); err != nil {
			t.Fatalf("%s control: %v", control.name, err)
		}
	}
}

func stallControl(dir, address string, boot stallBootstrap) (err error) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	var commands []*exec.Cmd
	var files []*os.File
	var logs []*stallLog
	var children []*stallChild
	defer func() {
		cancel()
		for rank, child := range children {
			<-child.done
			if child.err != nil {
				err = errors.Join(err, fmt.Errorf("rank %d exited: %w (see rank-%d.stdout and rank-%d.stderr)", rank, child.err, rank, rank))
			}
		}
		for _, log := range logs {
			err = errors.Join(err, log.Err())
		}
		for _, file := range files {
			err = errors.Join(err, file.Close())
		}
	}()
	for rank := 0; rank < 2; rank++ {
		stdout, e := os.OpenFile(filepath.Join(dir, fmt.Sprintf("rank-%d.stdout", rank)), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if e != nil {
			return e
		}
		files = append(files, stdout)
		stderr, e := os.OpenFile(filepath.Join(dir, fmt.Sprintf("rank-%d.stderr", rank)), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if e != nil {
			return e
		}
		files = append(files, stderr)
		command := exec.CommandContext(ctx, os.Args[0], "-test.v", "-test.run=^TestNativeStallRank$")
		for _, entry := range os.Environ() {
			key, _, _ := strings.Cut(entry, "=")
			// Keep diagnostics in the bounded streams, not an inherited file.
			if key != "NCCL_DEBUG_FILE" && key != "NCCL_DEBUG" && key != "NCCL_DEBUG_SUBSYS" {
				command.Env = append(command.Env, entry)
			}
		}
		command.Env = append(command.Env, "NCCLSTALL_CHILD="+strconv.Itoa(rank), "NCCLSTALL_CASE="+dir, "NCCL_DEBUG=INFO", "NCCL_DEBUG_SUBSYS=INIT,ENV")
		out := &stallLog{w: stdout, remaining: 1 << 20, cancel: cancel}
		errout := &stallLog{w: stderr, remaining: 1 << 20, cancel: cancel}
		logs = append(logs, out, errout)
		command.Stdout = out
		command.Stderr = errout
		if e := command.Start(); e != nil {
			return e
		}
		commands = append(commands, command)
		children = append(children, stallWatch(command, cancel))
	}
	var ready [2]stallRankReceipt
	for rank := 0; rank < 2; rank++ {
		if e := stallWaitChildren(ctx, filepath.Join(dir, fmt.Sprintf("ready-%d.json", rank)), &ready[rank], children); e != nil {
			return e
		}
		actual := ready[rank]
		if actual.Version != 1 || actual.RunID != boot.RunID || actual.Source != boot.Source || actual.Rank != rank || actual.PID != commands[rank].Process.Pid || actual.TID <= 0 || actual.Device != boot.Devices[rank] || len(actual.UUID) != 32 || actual.NCCLVersion <= 0 || actual.ProcessStartTicks == 0 {
			return errors.New("child bootstrap identity differs")
		}
		if actual.NCCLLibraryPath != os.Getenv("NCCLSTALL_NCCL_LIBRARY") || actual.NCCLLibrarySHA256 != os.Getenv("NCCLSTALL_NCCL_SHA256") {
			return errors.New("rank loaded NCCL library identity differs")
		}
	}
	if ready[0].PID == ready[1].PID || ready[0].UUID == ready[1].UUID || ready[0].BootID == "" || ready[0].BootID != ready[1].BootID || ready[0].Hostname != ready[1].Hostname {
		return errors.New("rank process/device/clock identity differs")
	}
	client := ncclras.Client{Address: address, Timeout: 3 * time.Second}
	before, e := client.Status(ctx)
	if e != nil {
		return e
	}
	hash, e := stallMembership(before, ready, 0)
	if e != nil {
		return e
	}
	if e := os.WriteFile(filepath.Join(dir, "ras-before.json"), before.Raw, 0600); e != nil {
		return e
	}
	now, e := stallNow()
	if e != nil {
		return e
	}
	if e := stallWrite(filepath.Join(dir, "start.json"), stallStart{Hash: hash, NotBeforeNS: now + int64(200*time.Millisecond)}); e != nil {
		return e
	}
	var receipts [2]stallRankReceipt
	for rank := 0; rank < 2; rank++ {
		if e := stallWaitChildren(ctx, filepath.Join(dir, fmt.Sprintf("result-%d.json", rank)), &receipts[rank], children); e != nil {
			return e
		}
		actual := receipts[rank]
		if actual.PID != ready[rank].PID || actual.TID != ready[rank].TID || actual.UUID != ready[rank].UUID || actual.RunID != boot.RunID || actual.Source != boot.Source || actual.BootID != ready[rank].BootID || actual.ProcessStartTicks != ready[rank].ProcessStartTicks {
			return errors.New("rank identity changed")
		}
		if actual.BarrierNS <= 0 || actual.DelayStartNS < actual.BarrierNS || actual.DelayEndNS < actual.DelayStartNS || actual.SubmitStartNS < actual.DelayEndNS || actual.SubmitEndNS < actual.SubmitStartNS || actual.SyncStartNS < actual.SubmitEndNS || actual.SyncEndNS < actual.SyncStartNS {
			return errors.New("reversed actual host interval")
		}
		if receipts[rank].CommunicatorHash != hash || receipts[rank].Comparisons != stallBytes/4 || receipts[rank].Wrong != 0 {
			return errors.New("rank numerical or communicator control failed")
		}
		output, e := os.ReadFile(filepath.Join(dir, fmt.Sprintf("rank-%d-output.bin", rank)))
		if e != nil {
			return e
		}
		count, wrong := stallOracle(output)
		if count != stallBytes/4 || wrong != 0 || fmt.Sprintf("%x", sha256.Sum256(output)) != receipts[rank].OutputSHA256 {
			return errors.New("independent retained output oracle failed")
		}
	}
	if boot.DelayNS > 0 && receipts[1].DelayEndNS-receipts[1].DelayStartNS < boot.DelayNS {
		return errors.New("actual delayed-rank wait not observed")
	}
	if boot.DelayNS > 0 && receipts[1].SubmitStartNS-receipts[0].SubmitStartNS < boot.DelayNS/2 {
		return errors.New("delayed peer submission ordering not established")
	}
	after, e := client.Status(ctx)
	if e != nil {
		return e
	}
	finalHash, e := stallMembership(after, ready, 1)
	if e != nil {
		return e
	}
	if finalHash != hash {
		return errors.New("communicator hash changed")
	}
	if e := os.WriteFile(filepath.Join(dir, "ras-after.json"), after.Raw, 0600); e != nil {
		return e
	}
	if e := stallWrite(filepath.Join(dir, "release.json"), true); e != nil {
		return e
	}
	for _, child := range children {
		<-child.done
		if child.err != nil {
			return child.err
		}
	}
	return nil
}

func stallMembership(status *ncclras.Status, ready [2]stallRankReceipt, count uint64) (string, error) {
	if status.Collection.Timeouts != 0 || len(status.Communicators) != 1 {
		return "", errors.New("incomplete actual RAS collection")
	}
	comm := status.Communicators[0]
	if err := ncclinspector.ValidateCommunicatorHash(comm.Hash); err != nil {
		return "", err
	}
	if comm.Size != 2 || comm.RanksCount != 2 || comm.MissingRanksCount != 0 || len(comm.Ranks) != 2 {
		return "", errors.New("RAS membership differs")
	}
	var seen [2]bool
	for _, rank := range comm.Ranks {
		if rank.Rank < 0 || rank.Rank > 1 || seen[rank.Rank] {
			return "", errors.New("RAS duplicate or invalid rank")
		}
		seen[rank.Rank] = true
		actual := ready[rank.Rank]
		version := fmt.Sprintf("%d.%d.%d", actual.NCCLVersion/10000, (actual.NCCLVersion%10000)/100, actual.NCCLVersion%100)
		observed, present := rank.Counts["AllReduce"]
		if rank.PID != actual.PID || rank.CUDADevice != actual.Device || !present || observed != count || status.Version != version {
			return "", errors.New("actual rank PID/device/count/version differs")
		}
	}
	return comm.Hash, nil
}

// TestNativeStallRank is an internal subprocess entry point, never a standalone
// hardware gate. All GPU work is owned by the parent's inherited lease.
func TestNativeStallRank(t *testing.T) {
	if os.Getenv("NCCLSTALL_NATIVE") != "1" || os.Getenv("NCCLSTALL_CHILD") == "" {
		t.Skip("internal native rank entry point")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatal(err)
	}
	rank, err := strconv.Atoi(os.Getenv("NCCLSTALL_CHILD"))
	if err != nil || rank < 0 || rank > 1 {
		t.Fatal("invalid child rank")
	}
	dir := os.Getenv("NCCLSTALL_CASE")
	if !filepath.IsAbs(dir) {
		t.Fatal("absolute case directory required")
	}
	var boot stallBootstrap
	if err := stallRead(filepath.Join(dir, "bootstrap.json"), &boot); err != nil {
		t.Fatal(err)
	}
	if boot.Version != 1 || boot.DelayNS < 0 || boot.DelayNS > int64(20*time.Millisecond) {
		t.Fatal("invalid bootstrap")
	}
	if err := stallRank(dir, rank, boot); err != nil {
		t.Fatal(err)
	}
}

func stallRank(dir string, rank int, boot stallBootstrap) (err error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	libraryPath, libraryHash, e := stallLoadNCCL()
	if e != nil {
		return e
	}
	if err := driver.Init(0); err != nil {
		return err
	}
	device, e := driver.DeviceGet(boot.Devices[rank])
	if e != nil {
		return e
	}
	contextHandle, e := device.PrimaryCtxRetain()
	if e != nil {
		return e
	}
	var stream driver.Stream
	var send, recv driver.DevicePtr
	var comm nccl.Comm
	completed := false
	defer func() {
		// A failed abort/destroy/drain cannot authorize releasing buffers still
		// referenced by the clique. Process exit owns the remaining resources.
		if comm != 0 {
			var cleanup error
			if completed {
				cleanup = comm.Destroy()
			} else {
				cleanup = comm.Abort()
			}
			if cleanup != nil {
				err = errors.Join(err, fmt.Errorf("communicator cleanup failed; resources retained: %w", cleanup))
				return
			}
			if stream != 0 {
				if cleanup = stream.Synchronize(); cleanup != nil {
					err = errors.Join(err, fmt.Errorf("stream drain failed; resources retained: %w", cleanup))
					return
				}
			}
		}
		if recv != 0 {
			err = errors.Join(err, recv.Free())
		}
		if send != 0 {
			err = errors.Join(err, send.Free())
		}
		if stream != 0 {
			err = errors.Join(err, stream.Close())
		}
		err = errors.Join(err, device.PrimaryCtxRelease())
	}()
	if e := contextHandle.SetCurrent(); e != nil {
		return e
	}
	stream, e = driver.NewStream(1)
	if e != nil {
		return e
	}
	send, e = driver.MemAllocBytes(stallBytes)
	if e != nil {
		return e
	}
	recv, e = driver.MemAllocBytes(stallBytes)
	if e != nil {
		return e
	}
	input := make([]byte, stallBytes)
	poison := make([]byte, stallBytes)
	for i := 0; i < stallBytes; i += 4 {
		binary.NativeEndian.PutUint32(input[i:], uint32(0x3f800000+rank*0x800000))
		binary.NativeEndian.PutUint32(poison[i:], 0x7fc00000)
	}
	if e := send.CopyFromHost(input); e != nil {
		return e
	}
	if e := recv.CopyFromHost(poison); e != nil {
		return e
	}
	if e := driver.CtxSynchronize(); e != nil {
		return e
	}
	comm, e = nccl.CommInitRank(2, boot.UniqueID, int32(rank))
	if e != nil {
		return e
	}
	uuid, e := device.Uuid()
	if e != nil {
		return e
	}
	version, e := nccl.GetVersion()
	if e != nil {
		return e
	}
	bootID, e := os.ReadFile("/proc/sys/kernel/random/boot_id")
	if e != nil {
		return e
	}
	hostname, e := os.Hostname()
	if e != nil {
		return e
	}
	stat, e := os.ReadFile("/proc/self/stat")
	if e != nil {
		return e
	}
	closeParen := strings.LastIndexByte(string(stat), ')')
	if closeParen < 0 {
		return errors.New("invalid process stat")
	}
	fields := strings.Fields(string(stat[closeParen+1:]))
	if len(fields) < 20 {
		return errors.New("process start ticks missing")
	}
	startTicks, e := strconv.ParseUint(fields[19], 10, 64)
	if e != nil || startTicks == 0 {
		return errors.New("invalid process start ticks")
	}
	receipt := stallRankReceipt{Version: 1, RunID: boot.RunID, Source: boot.Source, BootID: strings.TrimSpace(string(bootID)), Hostname: hostname, UUID: fmt.Sprintf("%x", uuid.Bytes), Rank: rank, PID: os.Getpid(), TID: unix.Gettid(), Device: boot.Devices[rank], NCCLVersion: version}
	receipt.ProcessStartTicks = startTicks
	receipt.NCCLLibraryPath = libraryPath
	receipt.NCCLLibrarySHA256 = libraryHash
	receipt.InputSHA256 = fmt.Sprintf("%x", sha256.Sum256(input))
	if e := stallWrite(filepath.Join(dir, fmt.Sprintf("ready-%d.json", rank)), receipt); e != nil {
		return e
	}
	waitCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	var start stallStart
	if e := stallWait(waitCtx, filepath.Join(dir, "start.json"), &start); e != nil {
		return e
	}
	if e := ncclinspector.ValidateCommunicatorHash(start.Hash); e != nil {
		return e
	}
	receipt.CommunicatorHash = start.Hash
	now, e := stallNow()
	if e != nil {
		return e
	}
	if start.NotBeforeNS-now > int64(time.Second) || start.NotBeforeNS <= 0 {
		return errors.New("invalid start barrier")
	}
	if now < start.NotBeforeNS {
		time.Sleep(time.Duration(start.NotBeforeNS - now))
	}
	receipt.BarrierNS, e = stallNow()
	if e != nil {
		return e
	}
	receipt.DelayStartNS = receipt.BarrierNS
	if rank == 1 && boot.DelayNS > 0 {
		time.Sleep(time.Duration(boot.DelayNS))
	}
	receipt.DelayEndNS, e = stallNow()
	if e != nil {
		return e
	}
	receipt.SubmitStartNS, e = stallNow()
	if e != nil {
		return e
	}
	if e := comm.AllReduce(send, recv, stallBytes/4, nccl.Float32, nccl.Sum, stream); e != nil {
		return e
	}
	receipt.SubmitEndNS, e = stallNow()
	if e != nil {
		return e
	}
	receipt.SyncStartNS, e = stallNow()
	if e != nil {
		return e
	}
	if e := stream.Synchronize(); e != nil {
		return e
	}
	completed = true
	receipt.SyncEndNS, e = stallNow()
	if e != nil {
		return e
	}
	output := make([]byte, stallBytes)
	if e := recv.CopyToHost(output); e != nil {
		return e
	}
	receipt.Comparisons, receipt.Wrong = stallOracle(output)
	receipt.OutputSHA256 = fmt.Sprintf("%x", sha256.Sum256(output))
	if e := os.WriteFile(filepath.Join(dir, fmt.Sprintf("rank-%d-output.bin", rank)), output, 0600); e != nil {
		return e
	}
	if e := stallWrite(filepath.Join(dir, fmt.Sprintf("result-%d.json", rank)), receipt); e != nil {
		return e
	}
	if receipt.Comparisons != stallBytes/4 || receipt.Wrong != 0 {
		return errors.New("real rank numerical control failed")
	}
	var release bool
	if e := stallWait(waitCtx, filepath.Join(dir, "release.json"), &release); e != nil {
		return e
	}
	if !release {
		return errors.New("invalid release")
	}
	return nil
}

func stallOracle(output []byte) (count, wrong int) {
	if len(output) != stallBytes {
		return 0, 1
	}
	for i := 0; i < len(output); i += 4 {
		count++
		if binary.NativeEndian.Uint32(output[i:]) != 0x40400000 {
			wrong++
		}
	}
	return count, wrong
}

func stallLoadNCCL() (string, string, error) {
	path := os.Getenv("NCCLSTALL_NCCL_LIBRARY")
	want := os.Getenv("NCCLSTALL_NCCL_SHA256")
	if !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return "", "", errors.New("absolute canonical NCCL library required")
	}
	decoded, err := hex.DecodeString(want)
	if err != nil || len(decoded) != 32 {
		return "", "", errors.New("exact NCCL library digest required")
	}
	fd, err := unix.Open(path, unix.O_RDONLY|unix.O_NONBLOCK|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return "", "", err
	}
	file := os.NewFile(uintptr(fd), path)
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > 512<<20 {
		file.Close()
		return "", "", errors.New("bounded regular NCCL library required")
	}
	hash := sha256.New()
	n, readErr := io.Copy(hash, io.LimitReader(file, 512<<20+1))
	closeErr := file.Close()
	if readErr != nil || closeErr != nil || n != info.Size() || !bytes.Equal(hash.Sum(nil), decoded) {
		return "", "", errors.New("NCCL library digest differs")
	}
	if err := nccl.Load(path); err != nil {
		return "", "", err
	}
	if nccl.Path() != path {
		return "", "", errors.New("NCCL loaded path differs")
	}
	return path, want, nil
}

func stallNow() (int64, error) {
	var ts unix.Timespec
	err := unix.ClockGettime(unix.CLOCK_MONOTONIC, &ts)
	return ts.Nano(), err
}

func stallWrite(path string, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if len(data) > 16384 {
		return errors.New("rank metadata exceeds bound")
	}
	temporary := path + ".writing"
	file, err := os.OpenFile(temporary, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	n, writeErr := file.Write(data)
	closeErr := file.Close()
	if writeErr != nil || closeErr != nil {
		return errors.Join(writeErr, closeErr)
	}
	if n != len(data) {
		return errors.New("short rank metadata write")
	}
	return os.Rename(temporary, path)
}

func stallRead(path string, value any) error {
	fd, err := unix.Open(path, unix.O_RDONLY|unix.O_NONBLOCK|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return err
	}
	file := os.NewFile(uintptr(fd), path)
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() || info.Size() > 16384 {
		return errors.New("invalid rank metadata file")
	}
	data, err := io.ReadAll(io.LimitReader(file, 16385))
	if err != nil {
		return err
	}
	if len(data) > 16384 {
		return errors.New("rank metadata exceeds bound")
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(value); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return errors.New("trailing rank metadata")
	}
	return nil
}

func stallWait(ctx context.Context, path string, value any) error {
	return stallWaitChildren(ctx, path, value, nil)
}

func stallWaitChildren(ctx context.Context, path string, value any, children []*stallChild) error {
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()
	for {
		err := stallRead(path, value)
		if err == nil {
			return nil
		}
		if !os.IsNotExist(err) {
			return err
		}
		for rank, child := range children {
			select {
			case <-child.done:
				return fmt.Errorf("rank %d exited before %s: %v", rank, filepath.Base(path), child.err)
			default:
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

// TestNativeStallOracleControls exercises CPU-only controls on the Linux-built
// runner. It does not admit GPU work or qualify any physical collector.
func TestNativeStallOracleControls(t *testing.T) {
	output := make([]byte, stallBytes)
	for i := 0; i < len(output); i += 4 {
		binary.NativeEndian.PutUint32(output[i:], 0x40400000)
	}
	if count, wrong := stallOracle(output); count != stallBytes/4 || wrong != 0 {
		t.Fatal("positive oracle failed")
	}
	binary.NativeEndian.PutUint32(output, 0x7fc00000)
	if count, wrong := stallOracle(output); count != stallBytes/4 || wrong != 1 {
		t.Fatal("poison not detected")
	}
	ready := [2]stallRankReceipt{{PID: 11, Device: 0, NCCLVersion: 23203}, {PID: 22, Device: 1, NCCLVersion: 23203}}
	status := &ncclras.Status{Version: "2.32.3", Communicators: []ncclras.Communicator{{Hash: "0x1234567890abcdef", Size: 2, RanksCount: 2, Ranks: []ncclras.Rank{{Rank: 0, PID: 11, CUDADevice: 0, Counts: map[string]uint64{"AllReduce": 1}}, {Rank: 1, PID: 22, CUDADevice: 1, Counts: map[string]uint64{"AllReduce": 1}}}}}}
	if _, err := stallMembership(status, ready, 1); err != nil {
		t.Fatal(err)
	}
	status.Communicators[0].Ranks[1].PID = 11
	if _, err := stallMembership(status, ready, 1); err == nil {
		t.Fatal("swapped process identity admitted")
	}
	status.Communicators[0].Ranks[1].PID = 22
	status.Communicators[0].Ranks[1].Counts = nil
	if _, err := stallMembership(status, ready, 0); err == nil {
		t.Fatal("missing count treated as zero")
	}
}

func TestNativeStallLogBounds(t *testing.T) {
	var raw bytes.Buffer
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	w := &stallLog{w: &raw, remaining: 4, cancel: cancel}
	if n, err := w.Write([]byte("abc")); n != 3 || err != nil {
		t.Fatalf("first write: %d, %v", n, err)
	}
	if n, err := w.Write([]byte("def")); n != 1 || !errors.Is(err, errStallLogOverflow) {
		t.Fatalf("overflow: %d, %v", n, err)
	}
	if raw.String() != "abcd" || ctx.Err() == nil || !errors.Is(w.Err(), errStallLogOverflow) {
		t.Fatal("overflow did not retain prefix and cancel")
	}
	if n, err := w.Write([]byte("more")); n != 0 || !errors.Is(err, errStallLogOverflow) {
		t.Fatalf("after overflow: %d, %v", n, err)
	}
}

func TestNativeStallChildExit(t *testing.T) {
	if os.Getenv("NCCLSTALL_EXIT_CONTROL") == "1" {
		fmt.Fprintln(os.Stderr, "cpu-only rank initialization failure")
		os.Exit(3)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var raw bytes.Buffer
	log := &stallLog{w: &raw, remaining: 1 << 20, cancel: cancel}
	command := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestNativeStallChildExit$")
	command.Env = append(os.Environ(), "NCCLSTALL_EXIT_CONTROL=1")
	command.Stdout, command.Stderr = log, log
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	child := stallWatch(command, cancel)
	var ready bool
	err := stallWaitChildren(ctx, filepath.Join(t.TempDir(), "missing-ready.json"), &ready, []*stallChild{child})
	<-child.done
	if err == nil || errors.Is(err, context.DeadlineExceeded) || ctx.Err() == context.DeadlineExceeded {
		t.Fatalf("early exit became readiness timeout: %v", err)
	}
	var exit *exec.ExitError
	if !errors.As(child.err, &exit) || exit.ExitCode() != 3 {
		t.Fatalf("lost rank failure: %v", child.err)
	}
	if !strings.Contains(raw.String(), "cpu-only rank initialization failure") || log.Err() != nil {
		t.Fatalf("lost drained diagnostic: %q, %v", raw.String(), log.Err())
	}
}
