package cudaprobe

import (
	"context"
	"encoding/binary"
	"errors"
	"go/parser"
	"go/token"
	"io"
	"os"
	"strconv"
	"strings"
	"testing"
)

type fixtureSource struct {
	records []Capture
	err     error
}

func (s *fixtureSource) Next(ctx context.Context) (Capture, error) {
	if err := ctx.Err(); err != nil {
		return Capture{}, err
	}
	if len(s.records) == 0 {
		if s.err != nil {
			return Capture{}, s.err
		}
		return Capture{}, io.EOF
	}
	c := s.records[0]
	s.records = s.records[1:]
	return c, nil
}

func TestTrackCaptureToLifetime(t *testing.T) {
	alloc := capture("cudaMalloc", 1, 0x1000, 64)
	alloc.Identity.AddressSpace = 1
	binary.LittleEndian.PutUint64(alloc.Data[120:], 0x8000)
	free := capture("cudaFree", 2, 0x8000)
	free.Identity.AddressSpace = 1
	for _, tt := range []struct {
		name       string
		records    []Capture
		err        error
		unmatched  int
		incomplete bool
	}{
		{"matched", []Capture{alloc, free}, nil, 0, false},
		{"exit-unmatched", []Capture{alloc}, nil, 1, false},
		{"lost-tail", []Capture{alloc}, io.ErrUnexpectedEOF, 1, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			tk, _ := NewTracker(alloc.Identity.Process, 1)
			err := Track(context.Background(), &fixtureSource{tt.records, tt.err}, tk)
			if !errors.Is(err, tt.err) {
				t.Fatalf("error=%v, want %v", err, tt.err)
			}
			got, incomplete := tk.Report()
			if len(got) != tt.unmatched || incomplete != tt.incomplete {
				t.Fatalf("report=%v incomplete=%t", got, incomplete)
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	tk, _ := NewTracker(alloc.Identity.Process, 1)
	src := &fixtureSource{records: []Capture{alloc}}
	if err := Track(ctx, src, tk); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if len(src.records) != 1 {
		t.Fatal("read after cancellation")
	}
}

func TestAllocationAddressSpaces(t *testing.T) {
	tk, _ := NewTracker(Process{PID: 7, Generation: 99}, 2)
	e := allocationEvent(1, "cudaMalloc", 8)
	if err := tk.Observe(e); err != nil {
		t.Fatal(err)
	}
	e.Sequence = 2
	e.Identity.AddressSpace = 2
	if err := tk.Observe(e); err != nil {
		t.Fatal(err)
	}
	e.Sequence = 3
	e.Identity.Symbol = "cudaFree"
	if err := tk.Observe(e); err != nil {
		t.Fatal(err)
	}
	got, incomplete := tk.Report()
	if len(got) != 1 || got[0].AddressSpace != 1 || incomplete {
		t.Fatalf("report=%v incomplete=%t", got, incomplete)
	}
	e.Sequence = 4
	e.Identity.AddressSpace = 0
	if err := tk.Observe(e); err != nil {
		t.Fatal(err)
	}
	got, incomplete = tk.Report()
	if len(got) != 1 || !incomplete {
		t.Fatalf("unknown context freed allocation: %v %t", got, incomplete)
	}
}

func TestDecodeLayerHasNoBPFDependency(t *testing.T) {
	pkgs, err := parser.ParseDir(token.NewFileSet(), ".", func(info os.FileInfo) bool { return !strings.HasSuffix(info.Name(), "_test.go") }, parser.ImportsOnly)
	if err != nil {
		t.Fatal(err)
	}
	seen := 0
	for _, pkg := range pkgs {
		for _, file := range pkg.Files {
			seen++
			for _, imp := range file.Imports {
				path, err := strconv.Unquote(imp.Path.Value)
				if err != nil {
					t.Fatal(err)
				}
				for _, bad := range []string{"ebpfhost", "cilium/ebpf", "libbpf", "debug/elf", "debug/dwarf"} {
					if strings.Contains(path, bad) {
						t.Errorf("forbidden import %q", path)
					}
				}
			}
		}
	}
	if seen == 0 {
		t.Fatal("checked no source files")
	}
}
