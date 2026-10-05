//go:build linux

package nativereceipt

import (
	"bufio"
	"errors"
	"fmt"
	"golang.org/x/sys/unix"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
)

func Identity(pid int) (Process, error) {
	if pid <= 1 {
		return Process{}, errors.New("native receipt: invalid process ID")
	}
	raw, err := readBounded("/proc/"+strconv.Itoa(pid)+"/stat", 64<<10)
	if err != nil {
		return Process{}, err
	}
	end := strings.LastIndexByte(string(raw), ')')
	if end < 0 {
		return Process{}, errors.New("native receipt: invalid process stat")
	}
	fields := strings.Fields(string(raw[end+1:]))
	if len(fields) < 20 {
		return Process{}, errors.New("native receipt: short process stat")
	}
	parent, e1 := strconv.Atoi(fields[1])
	group, e2 := strconv.Atoi(fields[2])
	start, e3 := strconv.ParseUint(fields[19], 10, 64)
	if e1 != nil || e2 != nil || e3 != nil || parent < 1 || group < 1 || start == 0 {
		return Process{}, errors.New("native receipt: invalid process identity")
	}
	exe, err := os.Readlink("/proc/" + strconv.Itoa(pid) + "/exe")
	if err != nil {
		return Process{}, err
	}
	if !filepath.IsAbs(exe) || strings.HasSuffix(exe, " (deleted)") {
		return Process{}, errors.New("native receipt: invalid executable path")
	}
	return Process{PID: pid, Parent: parent, Group: group, Start: start, Executable: exe}, nil
}
func newSession() (*Session, error) {
	p, err := Identity(os.Getpid())
	if err != nil {
		return nil, err
	}
	return &Session{receipt: Receipt{Process: p}}, nil
}
func readBounded(path string, limit int64) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(b)) > limit {
		return nil, errors.New("native receipt: process input exceeds bound")
	}
	return b, nil
}
func parseMaps(raw []byte) ([]Mapping, error) {
	if len(raw) > maxMaps {
		return nil, errors.New("native receipt: maps too large")
	}
	scanner := bufio.NewScanner(strings.NewReader(string(raw)))
	scanner.Buffer(make([]byte, 4096), 64<<10)
	var out []Mapping
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Fields(line)
		if len(fields) < 5 {
			return nil, errors.New("native receipt: invalid maps record")
		}
		ranges := strings.Split(fields[0], "-")
		device := strings.Split(fields[3], ":")
		if len(ranges) != 2 || len(device) != 2 {
			return nil, errors.New("native receipt: invalid maps range/device")
		}
		begin, e1 := strconv.ParseUint(ranges[0], 16, 64)
		end, e2 := strconv.ParseUint(ranges[1], 16, 64)
		offset, e3 := strconv.ParseUint(fields[2], 16, 64)
		major, e4 := strconv.ParseUint(device[0], 16, 32)
		minor, e5 := strconv.ParseUint(device[1], 16, 32)
		inode, e6 := strconv.ParseUint(fields[4], 10, 64)
		if e1 != nil || e2 != nil || e3 != nil || e4 != nil || e5 != nil || e6 != nil || begin >= end || len(fields[1]) != 4 {
			return nil, errors.New("native receipt: invalid maps numbers")
		}
		permissions := fields[1]
		if !strings.ContainsRune("r-", rune(permissions[0])) || !strings.ContainsRune("w-", rune(permissions[1])) || !strings.ContainsRune("x-", rune(permissions[2])) || !strings.ContainsRune("ps", rune(permissions[3])) {
			return nil, errors.New("native receipt: invalid maps permissions")
		}
		// Preserve spaces in mapped paths. Kernel newline escapes are refused for
		// selected file paths rather than guessing an ambiguous pathname.
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
		path := strings.TrimLeft(rest, " \t")
		out = append(out, Mapping{Begin: begin, End: end, Offset: offset, Permissions: permissions, Major: uint32(major), Minor: uint32(minor), Inode: inode, Path: path})
		if len(out) > 8192 {
			return nil, errors.New("native receipt: too many mappings")
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, errors.New("native receipt: empty mappings")
	}
	return out, nil
}
func library(path, name string) bool {
	base := filepath.Base(path)
	return base == name || strings.HasPrefix(base, name+".")
}
func selected(path, exe string) bool {
	return path == exe || filepath.IsAbs(path) && strings.Contains(filepath.Base(path), ".so")
}
func openMapped(m Mapping) (*body, error) {
	if !filepath.IsAbs(m.Path) || strings.Contains(m.Path, "\\") || strings.HasSuffix(m.Path, " (deleted)") || m.Inode == 0 {
		return nil, errors.New("native receipt: mapped path refused")
	}
	fd, err := syscall.Open(m.Path, syscall.O_RDONLY|syscall.O_CLOEXEC|syscall.O_NOFOLLOW|syscall.O_NONBLOCK, 0)
	if err != nil {
		return nil, fmt.Errorf("native receipt: open mapped body: %w", err)
	}
	f := os.NewFile(uintptr(fd), m.Path)
	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	st, ok := info.Sys().(*syscall.Stat_t)
	if !ok || !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > maxFileBytes || unix.Major(uint64(st.Dev)) != m.Major || unix.Minor(uint64(st.Dev)) != m.Minor || st.Ino != m.Inode {
		f.Close()
		return nil, errors.New("native receipt: mapped descriptor identity disagrees")
	}
	return &body{file: f, before: info, identity: File{Path: m.Path, Major: m.Major, Minor: m.Minor, Inode: m.Inode, Bytes: info.Size()}}, nil
}
func capture(s *Session, name, source, artifact string) (Stage, error) {
	p, err := Identity(os.Getpid())
	if err != nil {
		return Stage{}, err
	}
	old := s.receipt.Process
	if p.PID != old.PID || p.Parent != old.Parent || p.Group != old.Group || p.Start != old.Start || p.Executable != old.Executable {
		return Stage{}, errors.New("native receipt: process identity changed")
	}
	raw, err := readBounded("/proc/self/maps", maxMaps)
	if err != nil {
		return Stage{}, err
	}
	if s.rawBytes+len(raw) > maxRawMaps {
		return Stage{}, errors.New("native receipt: aggregate maps bound exceeded")
	}
	mappings, err := parseMaps(raw)
	if err != nil {
		return Stage{}, err
	}
	for _, m := range mappings {
		if !selected(m.Path, p.Executable) {
			continue
		}
		found := false
		for _, b := range s.bodies {
			if b.identity.Major == m.Major && b.identity.Minor == m.Minor && b.identity.Inode == m.Inode {
				if b.identity.Path != m.Path {
					return Stage{}, errors.New("native receipt: mapped inode has conflicting paths")
				}
				info, e := b.file.Stat()
				if e != nil || !sameBody(b.before, info) {
					return Stage{}, errors.New("native receipt: retained mapped body changed")
				}
				found = true
				break
			}
		}
		if !found {
			if len(s.bodies) >= maxFiles {
				return Stage{}, errors.New("native receipt: too many mapped bodies")
			}
			b, e := openMapped(m)
			if e != nil {
				return Stage{}, e
			}
			s.bodies = append(s.bodies, b)
			s.bytes += b.identity.Bytes
			if s.bytes > maxTotalBytes {
				return Stage{}, errors.New("native receipt: aggregate mapped bodies too large")
			}
		}
		s.receipt.DriverObserved = s.receipt.DriverObserved || library(m.Path, "libcuda.so")
		s.receipt.NVRTCObserved = s.receipt.NVRTCObserved || library(m.Path, "libnvrtc.so")
		s.receipt.BuiltinsObserved = s.receipt.BuiltinsObserved || library(m.Path, "libnvrtc-builtins.so")
	}
	s.rawBytes += len(raw)
	return Stage{Name: name, SourceSHA256: source, ArtifactSHA256: artifact, RawMaps: string(raw), Mappings: mappings}, nil
}

func metadataEqual(a, b os.FileInfo) bool {
	x, ok := a.Sys().(*syscall.Stat_t)
	y, ok2 := b.Sys().(*syscall.Stat_t)
	return ok && ok2 && x.Ctim == y.Ctim
}

func openReceipt(path string) (*os.File, error) {
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_CLOEXEC|syscall.O_NONBLOCK|syscall.O_NOFOLLOW, 0)
	if err != nil {
		return nil, err
	}
	return os.NewFile(uintptr(fd), path), nil
}
func validateProof(r Receipt) error {
	files := make(map[string]File)
	var total int64
	key := func(major, minor uint32, inode uint64) string { return fmt.Sprintf("%d:%d:%d", major, minor, inode) }
	for _, f := range r.Files {
		if !filepath.IsAbs(f.Path) || f.Inode == 0 {
			return errors.New("native receipt: invalid mapped identity")
		}
		k := key(f.Major, f.Minor, f.Inode)
		if _, ok := files[k]; ok {
			return errors.New("native receipt: duplicate mapped body")
		}
		files[k] = f
		total += f.Bytes
	}
	if total > maxTotalBytes {
		return errors.New("native receipt: aggregate bodies too large")
	}
	var driver, nvrtc, builtins, executable bool
	rawBytes := 0
	seen := make(map[string]bool)
	for _, s := range r.Stages {
		rawBytes += len(s.RawMaps)
		if rawBytes > maxRawMaps {
			return errors.New("native receipt: aggregate maps too large")
		}
		mappings, e := parseMaps([]byte(s.RawMaps))
		if e != nil {
			return e
		}
		if len(mappings) != len(s.Mappings) {
			return errors.New("native receipt: raw mapping count disagrees")
		}
		for i, m := range mappings {
			if m != s.Mappings[i] {
				return errors.New("native receipt: raw mapping identity disagrees")
			}
			if !selected(m.Path, r.Process.Executable) {
				continue
			}
			k := key(m.Major, m.Minor, m.Inode)
			f, ok := files[k]
			if !ok || f.Path != m.Path {
				return errors.New("native receipt: mapping descriptor proof absent")
			}
			seen[k] = true
			driver = driver || library(m.Path, "libcuda.so")
			nvrtc = nvrtc || library(m.Path, "libnvrtc.so")
			builtins = builtins || library(m.Path, "libnvrtc-builtins.so")
			executable = executable || m.Path == r.Process.Executable && f.SHA256 == r.Process.ExecutableSHA256
		}
	}
	if len(seen) != len(files) || !driver || !nvrtc || !executable || driver != r.DriverObserved || nvrtc != r.NVRTCObserved || builtins != r.BuiltinsObserved {
		return errors.New("native receipt: observed mapping proof disagrees")
	}
	if !builtins && len(r.Unobserved) == 0 {
		return errors.New("native receipt: absent builtins lack explicit limitation")
	}
	return nil
}
