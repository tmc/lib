//go:build linux

package main

import (
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
	"testing"

	"github.com/tmc/lib/nvidia/internal/nativereceipt"
	"golang.org/x/sys/unix"
)

type generatorProcessReceipt struct {
	Process  nativereceipt.Process `json:"process"`
	TestCode int                   `json:"test_code"`
	RawMaps  string                `json:"raw_maps"`
	Libclang []nativereceipt.File  `json:"libclang"`
}

// Optional full-command evidence belongs to this actual generator test worker.
// Ordinary invocation remains unchanged and does not require a receipt path.
func TestMain(m *testing.M) {
	dir := os.Getenv("CUDABIND_FULL_GENERATOR_RECEIPTS")
	if dir == "" {
		os.Exit(m.Run())
	}
	code := m.Run()
	proof, err := captureGeneratorProcess(code, os.Getenv("CUDABIND_LIBCLANG"), os.Getenv("CUDABIND_FULL_LIBCLANG_SHA256"))
	if err == nil {
		err = writeGeneratorProcess(dir, proof)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "full generator receipt:", err)
		code = 1
	}
	os.Exit(code)
}

func generatorBody(path string, mapping *nativereceipt.File, procExe bool) (nativereceipt.File, error) {
	flags := syscall.O_RDONLY | syscall.O_CLOEXEC | syscall.O_NONBLOCK
	if !procExe {
		flags |= syscall.O_NOFOLLOW
	}
	fd, err := syscall.Open(path, flags, 0)
	if err != nil {
		return nativereceipt.File{}, err
	}
	f := os.NewFile(uintptr(fd), path)
	defer f.Close()
	before, err := f.Stat()
	if err != nil {
		return nativereceipt.File{}, err
	}
	st, ok := before.Sys().(*syscall.Stat_t)
	if !ok || !before.Mode().IsRegular() || before.Size() <= 0 || before.Size() > 256<<20 {
		return nativereceipt.File{}, errors.New("ordinary bounded descriptor required")
	}
	row := nativereceipt.File{Path: path, Major: unix.Major(uint64(st.Dev)), Minor: unix.Minor(uint64(st.Dev)), Inode: st.Ino, Bytes: before.Size()}
	if mapping != nil && (mapping.Major != row.Major || mapping.Minor != row.Minor || mapping.Inode != row.Inode) {
		return nativereceipt.File{}, errors.New("mapped descriptor identity differs")
	}
	h := sha256.New()
	n, err := io.Copy(h, io.LimitReader(f, before.Size()+1))
	if err != nil {
		return nativereceipt.File{}, err
	}
	after, err := f.Stat()
	if err != nil {
		return nativereceipt.File{}, err
	}
	a, ok := after.Sys().(*syscall.Stat_t)
	if !ok || n != before.Size() || before.Size() != after.Size() || before.ModTime() != after.ModTime() || st.Dev != a.Dev || st.Ino != a.Ino {
		return nativereceipt.File{}, errors.New("descriptor changed during hash")
	}
	row.SHA256 = hex.EncodeToString(h.Sum(nil))
	return row, nil
}

func captureGeneratorProcess(code int, path, digest string) (generatorProcessReceipt, error) {
	var out generatorProcessReceipt
	if !filepath.IsAbs(path) || filepath.Clean(path) != path || len(digest) != 64 || strings.ToLower(digest) != digest {
		return out, errors.New("absolute libclang path and canonical digest required")
	}
	if _, err := hex.DecodeString(digest); err != nil {
		return out, err
	}
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		return out, err
	}
	out.Process, err = nativereceipt.Identity(os.Getpid())
	if err != nil {
		return out, err
	}
	exe, err := generatorBody("/proc/self/exe", nil, true)
	if err != nil {
		return out, err
	}
	out.Process.ExecutableSHA256 = exe.SHA256
	f, err := os.Open("/proc/self/maps")
	if err != nil {
		return out, err
	}
	raw, readErr := io.ReadAll(io.LimitReader(f, 8<<20+1))
	err = errors.Join(readErr, f.Close())
	if err != nil {
		return out, err
	}
	if len(raw) == 0 || len(raw) > 8<<20 {
		return out, errors.New("maps extent refused")
	}
	out.RawMaps, out.TestCode = string(raw), code
	seen := make(map[nativereceipt.File]bool)
	for _, line := range strings.Split(out.RawMaps, "\n") {
		fields := strings.Fields(line)
		if len(fields) < 6 {
			continue
		}
		p := strings.Join(fields[5:], " ")
		// Other loaded libraries are outside this selected library observation.
		// Every descriptor at the actual configured path must match its body.
		if p != resolved {
			continue
		}
		device := strings.Split(fields[3], ":")
		if len(device) != 2 {
			return out, errors.New("mapped device refused")
		}
		major, e1 := strconv.ParseUint(device[0], 16, 32)
		minor, e2 := strconv.ParseUint(device[1], 16, 32)
		inode, e3 := strconv.ParseUint(fields[4], 10, 64)
		if e1 != nil || e2 != nil || e3 != nil || inode == 0 {
			return out, errors.New("mapped identity refused")
		}
		mapping := nativereceipt.File{Major: uint32(major), Minor: uint32(minor), Inode: inode}
		body, err := generatorBody(p, &mapping, false)
		if err != nil {
			return out, err
		}
		if body.SHA256 != digest {
			return out, errors.New("observed libclang body differs")
		}
		if !seen[body] {
			out.Libclang = append(out.Libclang, body)
			seen[body] = true
		}
		if len(out.Libclang) > 32 {
			return out, errors.New("libclang descriptor count refused")
		}
	}
	if len(out.Libclang) == 0 {
		return out, errors.New("actual libclang mapping absent")
	}
	after, err := nativereceipt.Identity(os.Getpid())
	if err != nil || after != (nativereceipt.Process{PID: out.Process.PID, Parent: out.Process.Parent, Group: out.Process.Group, Start: out.Process.Start, Executable: out.Process.Executable}) {
		return out, errors.Join(err, errors.New("generator process identity changed"))
	}
	return out, nil
}

func writeGeneratorProcess(dir string, proof generatorProcessReceipt) error {
	return writeGeneratorProcessWithSync(dir, proof, (*os.File).Sync)
}

func writeGeneratorProcessWithSync(dir string, proof generatorProcessReceipt, syncDirectory func(*os.File) error) (retErr error) {
	if !filepath.IsAbs(dir) || filepath.Clean(dir) != dir {
		return errors.New("absolute ordinary receipt directory required")
	}
	for p := dir; ; p = filepath.Dir(p) {
		st, err := os.Lstat(p)
		if err != nil {
			return err
		}
		if !st.IsDir() || st.Mode()&os.ModeSymlink != 0 {
			return errors.New("ordinary receipt ancestor required")
		}
		if p == filepath.Dir(p) {
			break
		}
	}
	body, err := json.Marshal(proof)
	if err != nil {
		return err
	}
	if len(body) > 16<<20 {
		return errors.New("receipt output exceeds bound")
	}
	directoryFD, err := unix.Open(dir, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return err
	}
	directory := os.NewFile(uintptr(directoryFD), dir)
	defer func() { retErr = errors.Join(retErr, directory.Close()) }()
	held, err := directory.Stat()
	if err != nil {
		return err
	}
	checkDirectory := func() error {
		current, err := os.Lstat(dir)
		if err != nil {
			return err
		}
		if !current.IsDir() || current.Mode()&os.ModeSymlink != 0 || !os.SameFile(held, current) {
			return errors.New("receipt directory changed")
		}
		return nil
	}
	if err := checkDirectory(); err != nil {
		return err
	}
	fileFD, err := unix.Openat(directoryFD, "generator-process.json", unix.O_WRONLY|unix.O_CREAT|unix.O_EXCL|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0600)
	if err != nil {
		return err
	}
	f := os.NewFile(uintptr(fileFD), "generator-process.json")
	_, err = f.Write(body)
	err = errors.Join(err, f.Sync(), f.Close())
	if err != nil {
		return err
	}
	return errors.Join(syncDirectory(directory), checkDirectory())
}

func TestGeneratorReceiptDirectoryReplacementCPU(t *testing.T) {
	for _, symlink := range []bool{false, true} {
		root := t.TempDir()
		dir := filepath.Join(root, "receipt")
		other := filepath.Join(root, "other")
		for _, p := range []string{dir, other} {
			if err := os.Mkdir(p, 0700); err != nil {
				t.Fatal(err)
			}
		}
		err := writeGeneratorProcessWithSync(dir, generatorProcessReceipt{}, func(f *os.File) error {
			if err := os.Rename(dir, dir+"-old"); err != nil {
				t.Fatal(err)
			}
			if symlink {
				if err := os.Symlink(other, dir); err != nil {
					t.Fatal(err)
				}
			} else if err := os.Mkdir(dir, 0700); err != nil {
				t.Fatal(err)
			}
			return f.Sync()
		})
		if err == nil {
			t.Fatal("replaced receipt directory admitted")
		}
		if _, err := os.Stat(filepath.Join(dir+"-old", "generator-process.json")); err != nil {
			t.Fatal("original receipt missing", err)
		}
		if _, err := os.Stat(filepath.Join(dir, "generator-process.json")); !os.IsNotExist(err) {
			t.Fatal("replacement received a receipt", err)
		}
	}
}

func TestGeneratorReceiptDirectorySyncCPU(t *testing.T) {
	for _, fail := range []bool{false, true} {
		dir := t.TempDir()
		calls := 0
		want := errors.New("directory sync control")
		err := writeGeneratorProcessWithSync(dir, generatorProcessReceipt{}, func(f *os.File) error {
			calls++
			st, err := f.Stat()
			if err != nil || !st.IsDir() {
				t.Fatal("receipt sync target is not actual directory", err)
			}
			if fail {
				return want
			}
			return f.Sync()
		})
		if calls != 1 || fail && !errors.Is(err, want) || !fail && err != nil {
			t.Fatalf("directory sync calls=%d error=%v", calls, err)
		}
	}
}

// This is an ordinary mmap CPU fixture, not a loaded libclang qualification.
func TestGeneratorProcessMappedDescriptorCPU(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "libclang-fixture.so")
	body := []byte("ordinary mmap descriptor control\n")
	if err := os.WriteFile(path, body, 0600); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	mapped, err := unix.Mmap(int(f.Fd()), 0, len(body), unix.PROT_READ, unix.MAP_PRIVATE)
	if err != nil {
		f.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := errors.Join(unix.Munmap(mapped), f.Close()); err != nil {
			t.Error(err)
		}
	})
	sum := sha256.Sum256(body)
	digest := hex.EncodeToString(sum[:])
	proof, err := captureGeneratorProcess(0, path, digest)
	if err != nil {
		t.Fatal(err)
	}
	if proof.Process.PID != os.Getpid() || len(proof.Libclang) != 1 || proof.Libclang[0].SHA256 != digest || proof.Process.ExecutableSHA256 == "" {
		t.Fatal("actual own mapped descriptor relation absent")
	}
	for _, tt := range []struct{ name, path, digest string }{
		{"wrong-body", path, strings.Repeat("0", 64)},
		{"lexical-alias", filepath.Join(dir, "alias") + "/../libclang-fixture.so", digest},
		{"unobserved", filepath.Join(dir, "not-loaded.so"), digest},
		{"short-digest", path, "a"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := captureGeneratorProcess(0, tt.path, tt.digest); err == nil {
				t.Fatal("invalid observation admitted")
			}
		})
	}
	row := proof.Libclang[0]
	row.Inode++
	if _, err := generatorBody(path, &row, false); err == nil {
		t.Fatal("wrong mapped inode admitted")
	}
	if err := writeGeneratorProcess(dir, proof); err != nil {
		t.Fatal(err)
	}
	if err := writeGeneratorProcess(dir, proof); err == nil {
		t.Fatal("receipt replaced")
	}
}
