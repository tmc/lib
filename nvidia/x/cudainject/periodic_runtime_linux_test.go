//go:build linux

package cudainject_test

import (
	"bufio"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"

	"github.com/tmc/lib/nvidia/internal/nativereceipt"
	"golang.org/x/sys/unix"
)

const periodicRuntimeFileLimit = 256 << 20

type periodicRuntime struct {
	Process nativereceipt.Process
	RawMaps string
	Files   []periodicRuntimeFile
}

type periodicRuntimeFile struct {
	Path   string
	Bytes  int64
	SHA256 string
	Device uint64
	Inode  uint64
}

func periodicRuntimeBody(path string) (periodicRuntimeFile, error) {
	if !filepath.IsAbs(path) || strings.ContainsAny(path, "\n\r\t \\") {
		return periodicRuntimeFile{}, errors.New("periodic runtime: absolute canonical file required")
	}
	canonical, err := filepath.EvalSymlinks(path)
	if err != nil || canonical != path {
		return periodicRuntimeFile{}, errors.New("periodic runtime: noncanonical file")
	}
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_NONBLOCK|syscall.O_CLOEXEC|syscall.O_NOFOLLOW, 0)
	if err != nil {
		return periodicRuntimeFile{}, err
	}
	f := os.NewFile(uintptr(fd), path)
	defer f.Close()
	before, err := f.Stat()
	if err != nil {
		return periodicRuntimeFile{}, err
	}
	if !before.Mode().IsRegular() || before.Size() <= 0 || before.Size() > periodicRuntimeFileLimit {
		return periodicRuntimeFile{}, errors.New("periodic runtime: file extent refused")
	}
	h := sha256.New()
	n, err := io.Copy(h, io.LimitReader(f, periodicRuntimeFileLimit+1))
	if err != nil {
		return periodicRuntimeFile{}, err
	}
	after, err := f.Stat()
	if err != nil {
		return periodicRuntimeFile{}, err
	}
	a, aok := before.Sys().(*syscall.Stat_t)
	b, bok := after.Sys().(*syscall.Stat_t)
	if !aok || !bok || a.Dev != b.Dev || a.Ino != b.Ino || a.Size != b.Size || a.Mode != b.Mode || a.Ctim != b.Ctim || a.Mtim != b.Mtim || n != before.Size() {
		return periodicRuntimeFile{}, errors.New("periodic runtime: file changed")
	}
	return periodicRuntimeFile{path, n, fmt.Sprintf("%x", h.Sum(nil)), uint64(a.Dev), a.Ino}, nil
}

// periodicRuntimeInputs binds the four explicit files before starting the child.
// It does not infer libraries from names or prospective loader resolution.
func periodicRuntimeInputs(paths ...string) ([]periodicRuntimeFile, error) {
	if len(paths) != 4 {
		return nil, errors.New("periodic runtime: four explicit inputs required")
	}
	var files []periodicRuntimeFile
	seen := make(map[string]bool)
	var total int64
	for _, path := range paths {
		if seen[path] {
			return nil, errors.New("periodic runtime: duplicate input")
		}
		seen[path] = true
		f, err := periodicRuntimeBody(path)
		if err != nil {
			return nil, err
		}
		total += f.Bytes
		if total > 512<<20 {
			return nil, errors.New("periodic runtime: aggregate extent refused")
		}
		files = append(files, f)
	}
	return files, nil
}

func observePeriodicRuntime(pid int, inputs []periodicRuntimeFile) (periodicRuntime, error) {
	var result periodicRuntime
	if len(inputs) != 4 {
		return result, errors.New("periodic runtime: inputs missing")
	}
	p, err := nativereceipt.Identity(pid)
	if err != nil {
		return result, err
	}
	parent, err := nativereceipt.Identity(os.Getpid())
	if err != nil {
		return result, err
	}
	if p.Parent != parent.PID || p.Group != parent.Group || p.Executable != inputs[0].Path {
		return result, errors.New("periodic runtime: worker relation differs")
	}
	f, err := os.Open(fmt.Sprintf("/proc/%d/maps", pid))
	if err != nil {
		return result, err
	}
	raw, readErr := io.ReadAll(io.LimitReader(f, (1<<20)+1))
	closeErr := f.Close()
	if err := errors.Join(readErr, closeErr); err != nil {
		return result, err
	}
	if len(raw) == 0 || len(raw) > 1<<20 {
		return result, errors.New("periodic runtime: maps extent refused")
	}
	if err := periodicRuntimeMappings(raw, inputs); err != nil {
		return result, err
	}
	for _, want := range inputs {
		got, err := periodicRuntimeBody(want.Path)
		if err != nil {
			return result, err
		}
		if got != want {
			return result, errors.New("periodic runtime: observed mapped body differs")
		}
		result.Files = append(result.Files, got)
	}
	after, err := nativereceipt.Identity(pid)
	if err != nil || after != p {
		return result, errors.New("periodic runtime: worker identity changed")
	}
	result.Process, result.RawMaps = p, string(raw)
	result.Process.ExecutableSHA256 = result.Files[0].SHA256
	return result, nil
}

func periodicRuntimeMappings(raw []byte, inputs []periodicRuntimeFile) error {
	if len(raw) == 0 || len(raw) > 1<<20 {
		return errors.New("periodic runtime: maps extent refused")
	}
	wanted := make(map[string]periodicRuntimeFile)
	for _, f := range inputs {
		wanted[f.Path] = f
	}
	found := make(map[string]bool)
	s := bufio.NewScanner(strings.NewReader(string(raw)))
	s.Buffer(make([]byte, 4096), 64<<10)
	count := 0
	for s.Scan() {
		count++
		if count > 8192 {
			return errors.New("periodic runtime: too many mappings")
		}
		fields := strings.Fields(s.Text())
		if len(fields) < 5 {
			return errors.New("periodic runtime: malformed mapping")
		}
		if len(fields) == 5 {
			continue
		}
		path := strings.Join(fields[5:], " ")
		want, selected := wanted[path]
		if !selected {
			continue
		}
		ranges := strings.Split(fields[0], "-")
		if len(ranges) != 2 || len(fields) != 6 || len(fields[1]) != 4 {
			return errors.New("periodic runtime: malformed selected mapping")
		}
		begin, e4 := strconv.ParseUint(ranges[0], 16, 64)
		end, e5 := strconv.ParseUint(ranges[1], 16, 64)
		_, e6 := strconv.ParseUint(fields[2], 16, 64)
		if e4 != nil || e5 != nil || e6 != nil || begin >= end || !strings.ContainsRune("r-", rune(fields[1][0])) || !strings.ContainsRune("w-", rune(fields[1][1])) || !strings.ContainsRune("x-", rune(fields[1][2])) || !strings.ContainsRune("ps", rune(fields[1][3])) {
			return errors.New("periodic runtime: malformed selected range")
		}
		device := strings.Split(fields[3], ":")
		if len(device) != 2 {
			return errors.New("periodic runtime: malformed device")
		}
		major, e1 := strconv.ParseUint(device[0], 16, 32)
		minor, e2 := strconv.ParseUint(device[1], 16, 32)
		inode, e3 := strconv.ParseUint(fields[4], 10, 64)
		if e1 != nil || e2 != nil || e3 != nil || uint32(major) != unix.Major(want.Device) || uint32(minor) != unix.Minor(want.Device) || inode != want.Inode {
			return errors.New("periodic runtime: mapped descriptor differs")
		}
		found[path] = true
	}
	if err := s.Err(); err != nil {
		return err
	}
	for path := range wanted {
		if !found[path] {
			return errors.New("periodic runtime: required mapping absent")
		}
	}
	return nil
}
