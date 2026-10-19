package cudacache

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// mapsFixture is /proc/self/maps text written by hand from the proc(5)
// format: NVRTC mapped as four segments, its builtins library beside it, an
// unlinked driver, a path with a space, and anonymous mappings.
const mapsFixture = `aaaab0000000-aaaab0001000 r-xp 00000000 103:02 1311 /usr/bin/prog
ffff80000000-ffff80400000 r--p 00000000 103:02 5001 /usr/local/cuda/lib64/libnvrtc.so.13.0.88
ffff80400000-ffff81000000 r-xp 00400000 103:02 5001 /usr/local/cuda/lib64/libnvrtc.so.13.0.88
ffff81000000-ffff81100000 r--p 00c00000 103:02 5001 /usr/local/cuda/lib64/libnvrtc.so.13.0.88
ffff81100000-ffff81200000 rw-p 00d00000 103:02 5001 /usr/local/cuda/lib64/libnvrtc.so.13.0.88
ffff82000000-ffff83000000 r-xp 00000000 103:02 5002 /usr/local/cuda/lib64/libnvrtc-builtins.so.13.0.88
ffff84000000-ffff85000000 r-xp 00000000 103:02 6001 /usr/lib/aarch64-linux-gnu/libcuda.so.580.95.05 (deleted)
ffff86000000-ffff86001000 r-xp 00000000 103:02 7001 /opt/my libs/libfoo.so.1
ffff87000000-ffff87001000 rw-p 00000000 00:00 0
ffff88000000-ffff88021000 rw-p 00000000 00:00 0                          [heap]
malformed line
`

func TestParseMappedLibrary(t *testing.T) {
	tests := []struct {
		name    string
		soname  string
		want    []mapping
		wantErr string
	}{
		{
			name:   "segments collapse, builtins excluded",
			soname: "libnvrtc.so.13",
			want:   []mapping{{path: "/usr/local/cuda/lib64/libnvrtc.so.13.0.88", major: 0x103, minor: 2, inode: 5001}},
		},
		{
			name:   "builtins by its own name",
			soname: "libnvrtc-builtins.so.13",
			want:   []mapping{{path: "/usr/local/cuda/lib64/libnvrtc-builtins.so.13.0.88", major: 0x103, minor: 2, inode: 5002}},
		},
		{
			name:   "soname links to a longer version, deleted",
			soname: "libcuda.so.1",
			want:   []mapping{{path: "/usr/lib/aarch64-linux-gnu/libcuda.so.580.95.05", major: 0x103, minor: 2, inode: 6001, deleted: true}},
		},
		{
			name:   "space in path",
			soname: "libfoo.so.1",
			want:   []mapping{{path: "/opt/my libs/libfoo.so.1", major: 0x103, minor: 2, inode: 7001}},
		},
		{
			name:   "any version of the same library",
			soname: "libnvrtc-builtins.so.12",
			want:   []mapping{{path: "/usr/local/cuda/lib64/libnvrtc-builtins.so.13.0.88", major: 0x103, minor: 2, inode: 5002}},
		},
		{
			name:    "not mapped",
			soname:  "libnvJitLink.so.13",
			wantErr: "not mapped",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := libraryMatches([]byte(mapsFixture), tt.soname)
			if tt.wantErr != "" {
				_, err := uniqueMapping(tt.soname, got)
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("uniqueMapping = %v, want error containing %q (mappings %+v)", err, tt.wantErr, got)
				}
				return
			}
			if len(got) != len(tt.want) {
				t.Fatalf("mappedFiles = %+v, want %+v", got, tt.want)
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("mapping %d = %+v, want %+v", i, got[i], tt.want[i])
				}
			}
		})
	}

	// Two different files answering to one soname cannot be told apart.
	two := mapsFixture + "ffff90000000-ffff90001000 r-xp 00000000 103:02 5003 /opt/other/libnvrtc.so.13.1.0\n"
	_, err := uniqueMapping("libnvrtc.so.13", libraryMatches([]byte(two), "libnvrtc.so.13"))
	if err == nil || !strings.Contains(err.Error(), "ambiguous") {
		t.Fatalf("two libnvrtc.so.13 files: %v, want ambiguous", err)
	}

	// Another major version beside it is not the library opened (review Q2 F1).
	cuda12 := mapsFixture + "ffff90000000-ffff90001000 r-xp 00000000 103:02 5004 /usr/local/cuda-12/lib64/libnvrtc.so.12.8.93\n"
	m, err := uniqueMapping("libnvrtc.so.13", libraryMatches([]byte(cuda12), "libnvrtc.so.13"))
	if err != nil || m.inode != 5001 {
		t.Fatalf("libnvrtc.so.13 beside libnvrtc.so.12 = %+v, %v; want inode 5001", m, err)
	}
}

func FuzzParseMappedLibrary(f *testing.F) {
	for _, line := range strings.Split(mapsFixture, "\n") {
		f.Add(line)
	}
	f.Fuzz(func(t *testing.T, line string) {
		m, ok := parseMapsLine(line)
		if !ok {
			return
		}
		if !strings.HasPrefix(m.path, "/") || m.inode == 0 {
			t.Fatalf("parseMapsLine(%q) = %+v: accepted a non-file mapping", line, m)
		}
		if !strings.Contains(line, m.path) {
			t.Fatalf("parseMapsLine(%q) = %+v: path not in line", line, m)
		}
	})
}

func TestLoadedLibraryPathHost(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "libstandin.so.1")
	if err := os.WriteFile(path, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := LoadedLibraryPath(path)
	if runtime.GOOS == "linux" {
		// The file exists but this process has not mapped it.
		if err == nil || !strings.Contains(err.Error(), "not mapped") {
			t.Fatalf("LoadedLibraryPath(unmapped) = %q, %v; want not mapped", got, err)
		}
		return
	}
	resolved, rerr := filepath.EvalSymlinks(path)
	if rerr != nil {
		t.Fatal(rerr)
	}
	if err != nil || got != resolved {
		t.Fatalf("LoadedLibraryPath(%s) = %q, %v; want %q", path, got, err, resolved)
	}
	if _, err := LoadedLibraryPath("libstandin.so.1"); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("LoadedLibraryPath(bare soname) on %s = %v, want ErrUnsupported", runtime.GOOS, err)
	}
}

func TestProbeToolchainConcurrent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "libnvrtc.so.13")
	if err := os.WriteFile(path, []byte("nvrtc"), 0o644); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	digests := make([]ToolDigest, 16)
	errs := make([]error, len(digests))
	for i := range digests {
		wg.Go(func() { digests[i], errs[i] = DigestTool("nvrtc", path) })
	}
	wg.Wait()
	for i := range digests {
		if errs[i] != nil || digests[i] != digests[0] {
			t.Fatalf("digest %d = %+v, %v; want %+v", i, digests[i], errs[i], digests[0])
		}
	}

	// A remembered digest must not survive a change of the file.
	if err := os.WriteFile(path, []byte("NVRTC"), 0o644); err != nil {
		t.Fatal(err)
	}
	d, err := DigestTool("nvrtc", path)
	if err != nil {
		t.Fatal(err)
	}
	if d.SHA256 == digests[0].SHA256 {
		t.Fatalf("digest after rewrite = %s, unchanged", d.SHA256)
	}

	if nvrtc.Loaded() || driver.Loaded() {
		return
	}
	var probes [8]ToolProvenance
	for i := range probes {
		wg.Go(func() {
			p, err := ProbeToolchain()
			if err != nil {
				t.Errorf("ProbeToolchain with nothing loaded: %v", err)
			}
			probes[i] = p
		})
	}
	wg.Wait()
	for _, p := range probes {
		if p.Recipe != RecipeVersion || len(p.Tools) != 0 || p.DriverVersion != 0 {
			t.Fatalf("ProbeToolchain with nothing loaded = %+v, want recipe only", p)
		}
	}
}
