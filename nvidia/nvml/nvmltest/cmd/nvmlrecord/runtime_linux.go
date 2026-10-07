//go:build linux

package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
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

const runtimeMapsLimit = 2 << 20
const runtimeBodyLimit = 128 << 20

type runtimeMapping struct {
	Begin, End, Offset uint64
	Permissions        string
	Major, Minor       uint32
	Inode              uint64
	Path               string
}
type runtimeBody struct {
	Path         string
	Major, Minor uint32
	Inode        uint64
	Bytes        int64
	SHA256       string
}
type runtimeEvidence struct {
	Version   int
	Process   nativereceipt.Process
	RawMaps   string
	Mappings  []runtimeMapping
	Libraries []runtimeBody
}

func runtimeProofSupported() error {
	path := os.Getenv("NVML_LIB_PATH")
	if !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return errors.New("native proof requires explicit canonical absolute NVML_LIB_PATH")
	}
	if err := runtimeOrdinaryPath(path); err != nil {
		return err
	}
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > runtimeBodyLimit {
		return errors.New("NVML input size or type refused")
	}
	return nil
}

func runtimeRead(path string, limit int64) ([]byte, error) {
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_CLOEXEC|syscall.O_NONBLOCK, 0)
	if err != nil {
		return nil, err
	}
	f := os.NewFile(uintptr(fd), path)
	b, err := io.ReadAll(io.LimitReader(f, limit+1))
	err = errors.Join(err, f.Close())
	if err != nil {
		return nil, err
	}
	if int64(len(b)) > limit {
		return nil, errors.New("runtime input exceeds bound")
	}
	return b, nil
}

func runtimeMaps(raw []byte) ([]runtimeMapping, error) {
	if len(raw) > runtimeMapsLimit {
		return nil, errors.New("runtime maps exceed bound")
	}
	scanner := bufio.NewScanner(strings.NewReader(string(raw)))
	scanner.Buffer(make([]byte, 4096), 64<<10)
	var out []runtimeMapping
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Fields(line)
		if len(fields) < 5 {
			return nil, errors.New("invalid runtime map")
		}
		span := strings.Split(fields[0], "-")
		dev := strings.Split(fields[3], ":")
		if len(span) != 2 || len(dev) != 2 {
			return nil, errors.New("invalid runtime map range")
		}
		begin, e1 := strconv.ParseUint(span[0], 16, 64)
		end, e2 := strconv.ParseUint(span[1], 16, 64)
		offset, e3 := strconv.ParseUint(fields[2], 16, 64)
		major, e4 := strconv.ParseUint(dev[0], 16, 32)
		minor, e5 := strconv.ParseUint(dev[1], 16, 32)
		inode, e6 := strconv.ParseUint(fields[4], 10, 64)
		perm := fields[1]
		if e1 != nil || e2 != nil || e3 != nil || e4 != nil || e5 != nil || e6 != nil || begin >= end || len(perm) != 4 {
			return nil, errors.New("invalid runtime map numbers")
		}
		if !strings.ContainsRune("r-", rune(perm[0])) || !strings.ContainsRune("w-", rune(perm[1])) || !strings.ContainsRune("x-", rune(perm[2])) || !strings.ContainsRune("ps", rune(perm[3])) {
			return nil, errors.New("invalid runtime map permissions")
		}
		rest := line
		for i := 0; i < 5; i++ {
			rest = strings.TrimLeft(rest, " \t")
			j := strings.IndexAny(rest, " \t")
			if j < 0 {
				rest = ""
				break
			}
			rest = rest[j:]
		}
		out = append(out, runtimeMapping{begin, end, offset, perm, uint32(major), uint32(minor), inode, strings.TrimLeft(rest, " \t")})
		if len(out) > 8192 {
			return nil, errors.New("runtime map count exceeds bound")
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, errors.New("empty runtime maps")
	}
	return out, nil
}

func runtimeOrdinaryPath(path string) error {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path || strings.Contains(path, "\\") || strings.HasSuffix(path, " (deleted)") {
		return errors.New("noncanonical runtime body path")
	}
	for p := path; p != "/"; p = filepath.Dir(p) {
		info, err := os.Lstat(p)
		if err != nil {
			return err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return errors.New("runtime body path contains symlink")
		}
		if p != path && !info.IsDir() {
			return errors.New("runtime body ancestor is not directory")
		}
	}
	return nil
}

func runtimeDigest(f *os.File, before os.FileInfo) (string, error) {
	if !before.Mode().IsRegular() || before.Size() <= 0 || before.Size() > runtimeBodyLimit {
		return "", errors.New("runtime body size or type refused")
	}
	h := sha256.New()
	n, err := io.Copy(h, io.LimitReader(f, runtimeBodyLimit+1))
	if err != nil {
		return "", err
	}
	after, err := f.Stat()
	if err != nil {
		return "", err
	}
	a, ok := before.Sys().(*syscall.Stat_t)
	b, ok2 := after.Sys().(*syscall.Stat_t)
	if !ok || !ok2 || !os.SameFile(before, after) || n != before.Size() || before.Size() != after.Size() || a.Mtim != b.Mtim || a.Ctim != b.Ctim {
		return "", errors.New("runtime descriptor body changed")
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func runtimeMappedBody(m runtimeMapping) (body runtimeBody, err error) {
	if err = runtimeOrdinaryPath(m.Path); err != nil {
		return body, err
	}
	fd, err := syscall.Open(m.Path, syscall.O_RDONLY|syscall.O_CLOEXEC|syscall.O_NOFOLLOW|syscall.O_NONBLOCK, 0)
	if err != nil {
		return body, err
	}
	f := os.NewFile(uintptr(fd), m.Path)
	defer func() { err = errors.Join(err, f.Close()) }()
	info, err := f.Stat()
	if err != nil {
		return body, err
	}
	st, ok := info.Sys().(*syscall.Stat_t)
	if !ok || m.Inode == 0 || st.Ino != m.Inode || unix.Major(uint64(st.Dev)) != m.Major || unix.Minor(uint64(st.Dev)) != m.Minor {
		return body, errors.New("runtime mapping descriptor differs")
	}
	digest, err := runtimeDigest(f, info)
	if err != nil {
		return body, err
	}
	current, err := os.Lstat(m.Path)
	if err != nil || !os.SameFile(info, current) {
		return body, errors.New("runtime body path changed")
	}
	return runtimeBody{m.Path, m.Major, m.Minor, m.Inode, info.Size(), digest}, nil
}

func captureRuntime() (json.RawMessage, error) {
	if err := runtimeProofSupported(); err != nil {
		return nil, err
	}
	p, err := nativereceipt.Identity(os.Getpid())
	if err != nil {
		return nil, err
	}
	// This descriptor belongs to this running executable, not a caller hash.
	f, err := os.Open("/proc/self/exe")
	if err != nil {
		return nil, err
	}
	info, err := f.Stat()
	var digest string
	if err == nil {
		digest, err = runtimeDigest(f, info)
	}
	err = errors.Join(err, f.Close())
	if err != nil {
		return nil, err
	}
	p.ExecutableSHA256 = digest
	raw, err := runtimeRead("/proc/self/maps", runtimeMapsLimit)
	if err != nil {
		return nil, err
	}
	maps, err := runtimeMaps(raw)
	if err != nil {
		return nil, err
	}
	exeInfo, err := os.Lstat(p.Executable)
	if err != nil || !os.SameFile(info, exeInfo) {
		return nil, errors.New("runtime executable path differs")
	}
	proof := runtimeEvidence{Version: 1, Process: p, RawMaps: string(raw), Mappings: maps}
	var total int64
	for _, m := range maps {
		base := filepath.Base(m.Path)
		if base != "libnvidia-ml.so" && !strings.HasPrefix(base, "libnvidia-ml.so.") {
			continue
		}
		if len(proof.Libraries) >= 128 {
			return nil, errors.New("runtime NVML mapping count exceeds bound")
		}
		if m.Path != os.Getenv("NVML_LIB_PATH") {
			return nil, errors.New("unexpected NVML mapping path")
		}
		body, e := runtimeMappedBody(m)
		if e != nil {
			return nil, fmt.Errorf("mapped NVML: %w", e)
		}
		total += body.Bytes
		if total > 256<<20 {
			return nil, errors.New("runtime NVML body aggregate exceeds bound")
		}
		proof.Libraries = append(proof.Libraries, body)
	}
	if len(proof.Libraries) == 0 {
		return nil, errors.New("actual NVML mapping not observed")
	}
	after, err := nativereceipt.Identity(os.Getpid())
	if err != nil {
		return nil, err
	}
	after.ExecutableSHA256 = digest
	if after != p {
		return nil, errors.New("runtime process identity changed")
	}
	return json.Marshal(proof)
}
