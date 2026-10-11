//go:build linux && native

package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalineinfo"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/egraph"
)

type sidecarStaticRow struct {
	Function         string
	PC               uint64
	File             string
	Line             int
	Category         string
	Symbol           string
	Class            *egraph.Id
	Operator, Origin string
}

type sidecarStaticReport struct {
	Scope, Status, Filename, SourceSHA256, MetadataSHA256, CUBINSHA256, PTXSHA256 string
	Image                                                                         sidecarImageRecord
	Rows                                                                          []sidecarStaticRow
}

func sidecarHash(body []byte) string { return fmt.Sprintf("%x", sha256.Sum256(body)) }

// This joins static instructions in the actual loaded image, not sampled PCs.
func sidecarAttribute(record sidecarImageRecord, image []byte, compiled sidecarImageSource, metadata []byte, dis *cudasass.Disassembly) (sidecarStaticReport, error) {
	r := sidecarStaticReport{Scope: "static loaded-image attribution; no sampled heat", Filename: compiled.Filename, SourceSHA256: sidecarHash([]byte(compiled.Code)), MetadataSHA256: sidecarHash(metadata), CUBINSHA256: sidecarHash(image), PTXSHA256: sidecarHash([]byte(compiled.PTX)), Image: record}
	if len(image) == 0 || len(image) > sidecarImageBytes || !filepath.IsAbs(compiled.Filename) || filepath.Clean(compiled.Filename) != compiled.Filename || len(compiled.Code) == 0 || len(compiled.Code) > 4<<20 || len(compiled.PTX) == 0 || len(compiled.PTX) > 8<<20 || record.Function == 0 || record.State.Generation == 0 || record.State.Version == 0 || record.State.Kind != "kernel" || record.SourceSHA256 != r.SourceSHA256 || record.State.SourceDigest != r.SourceSHA256 || record.CUBINSHA256 != r.CUBINSHA256 || record.Bytes != len(image) || record.CompilerKey != compiled.Key || record.State.CompilerKey != compiled.Key || record.Entry != compiled.Entry || record.ReferenceSidecarSHA256 != r.MetadataSHA256 {
		return r, errors.New("static attribution lacks exact held image/source identity")
	}
	source, err := egraph.ReadMetadata(strings.NewReader(string(metadata)), compiled.Code)
	if err != nil {
		return r, err
	}
	if len(source.Symbols) == 0 {
		r.Status = "edited source; original mapping unavailable"
		return r, nil
	}
	if record.ReferenceSourceSHA256 != r.SourceSHA256 || dis == nil {
		return r, errors.New("static attribution lacks unedited source or disassembly")
	}
	a := cudalineinfo.Attribute(cudalineinfo.Unit{Name: compiled.Entry, Source: compiled.Code, Kernels: []string{compiled.Entry}}, compiled.PTX, dis)
	seen := make(map[string]bool)
	mapped := 0
	for _, fn := range dis.Functions {
		for _, in := range fn.Instructions {
			if len(r.Rows) >= 65536 {
				return r, errors.New("static instruction bound exceeded")
			}
			key := fmt.Sprintf("%s:%x", fn.Name, in.Address)
			if seen[key] {
				return r, errors.New("duplicate static instruction identity")
			}
			seen[key] = true
			row := sidecarStaticRow{Function: fn.Name, PC: in.Address, File: in.SourceFile, Line: in.SourceLine}
			switch {
			case fn.Name != compiled.Entry || fn.SymbolAt(in.Address) != fn.Name:
				row.Category = "subroutine"
			case in.SourceFile == "":
				row.Category = "unlocated"
			case in.SourceFile == cudasass.PTXTextFile:
				row.Category = "compiler PTX"
			case in.SourceFile != compiled.Filename:
				switch strings.ToLower(filepath.Ext(in.SourceFile)) {
				case ".h", ".hpp", ".hxx", ".hh", ".cuh", ".inl":
					row.Category = "header"
				default:
					return r, errors.New("static instruction names foreign source file")
				}
			default:
				site, ok := a.SiteAt(fn.Name, in.Address)
				if !ok || site.File != compiled.Filename || site.Line != in.SourceLine || site.Line < 1 || site.Line > len(strings.Split(compiled.Code, "\n")) {
					return r, errors.New("static instruction site differs")
				}
				class, ok := source.ClassAt(site.Line)
				if !ok {
					row.Category = "outer source"
					break
				}
				for name, symbol := range source.Symbols {
					if site.Line >= symbol.FirstLine && site.Line <= symbol.LastLine {
						if row.Symbol != "" || symbol.Class != class {
							return r, errors.New("ambiguous static symbol")
						}
						row.Symbol, row.Operator, row.Origin = name, symbol.Operator, symbol.Origin
					}
				}
				if row.Symbol == "" {
					return r, errors.New("static class has no exact symbol")
				}
				row.Category, row.Class = "e-class", &class
				mapped++
			}
			r.Rows = append(r.Rows, row)
		}
	}
	if mapped == 0 {
		return r, errors.New("loaded image has no emitted e-class instructions")
	}
	r.Status = "attributed"
	return r, nil
}

func sidecarValidateStatic(got sidecarStaticReport, record sidecarImageRecord, image []byte, compiled sidecarImageSource, metadata []byte, dis *cudasass.Disassembly) error {
	want, err := sidecarAttribute(record, image, compiled, metadata, dis)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(got, want) {
		return errors.New("static attribution receipt differs from retained inputs")
	}
	return nil
}

func sidecarStaticAttribution(ctx context.Context, dir, name string, record sidecarImageRecord, image []byte, compiled sidecarImageSource, metadata []byte) error {
	if len(compiled.Code) == 0 || record.SourceSHA256 != sidecarHash([]byte(compiled.Code)) {
		return errors.New("static source retention absent")
	}
	if err := os.WriteFile(filepath.Join(dir, name+".source.cu"), []byte(compiled.Code), 0600); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, name+".ptx"), []byte(compiled.PTX), 0600); err != nil {
		return err
	}
	source, err := egraph.ReadMetadata(strings.NewReader(string(metadata)), compiled.Code)
	if err != nil {
		return err
	}
	var dis *cudasass.Disassembly
	if len(source.Symbols) != 0 {
		tool := os.Getenv("SIDECAR_NVDISASM")
		if !filepath.IsAbs(tool) || filepath.Clean(tool) != tool {
			return errors.New("absolute pinned SIDECAR_NVDISASM required")
		}
		toolBody, err := sidecarReadFile(tool)
		if err != nil {
			return err
		}
		if sidecarHash(toolBody) != os.Getenv("SIDECAR_NVDISASM_SHA256") {
			return errors.New("static disassembler body drift")
		}
		tmp := filepath.Join(dir, "disassembler-tmp")
		if err := os.MkdirAll(tmp, 0700); err != nil {
			return err
		}
		bounded, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()
		out, runErr := (cudasass.Disassembler{Path: tool, TempDir: tmp}).Disassemble(bounded, image, cudasass.Options{PrintCode: true, SourceLines: true})
		data, marshalErr := json.Marshal(out)
		writeErr := os.WriteFile(filepath.Join(dir, name+".disassembly.json"), data, 0600)
		if err := errors.Join(runErr, marshalErr, writeErr); err != nil {
			return err
		}
		if out.ExitCode != 0 || out.Truncated {
			return errors.New("static disassembler incomplete")
		}
		after, err := sidecarReadFile(tool)
		if err != nil || out.Path != tool || sidecarHash(after) != os.Getenv("SIDECAR_NVDISASM_SHA256") {
			return errors.Join(err, errors.New("static disassembler identity changed"))
		}
		dis, err = cudasass.ParseString(string(out.Stdout))
		if err != nil {
			return err
		}
	}
	report, err := sidecarAttribute(record, image, compiled, metadata, dis)
	if err != nil {
		return err
	}
	if err := sidecarValidateStatic(report, record, image, compiled, metadata, dis); err != nil {
		return err
	}
	data, err := json.Marshal(report)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, name+".attribution.json"), data, 0600)
}

func TestSidecarStaticAttributionHost(t *testing.T) {
	dir := filepath.Join("testdata", "sidecar", "recorded-bar")
	read := func(name string) []byte {
		t.Helper()
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	code, metadata, raw := read("bar.cu"), read("bar.egraph.json"), read("bar.sass")
	dis, err := cudasass.ParseString(string(raw))
	if err != nil {
		t.Fatal(err)
	}
	filename := ""
	for _, fn := range dis.Functions {
		if fn.Name == "chain" {
			for _, in := range fn.Instructions {
				if strings.HasSuffix(in.SourceFile, "/bar.cu") {
					filename = in.SourceFile
					break
				}
			}
		}
	}
	if !filepath.IsAbs(filename) {
		t.Fatal("recorded compiler filename absent")
	}
	// The image/PTX below are simulated owner inputs to the host join. The
	// SASS/source/metadata are the recorded compiler fixture, not device data.
	image := []byte("simulated retained image")
	compiled := sidecarImageSource{Digest: sidecarHash(code), Key: "host-key", Entry: "chain", Filename: filename, Code: string(code), PTX: "// simulated PTX"}
	record := sidecarImageRecord{Function: 123, SourceSHA256: compiled.Digest, CompilerKey: compiled.Key, Entry: compiled.Entry, CUBINSHA256: sidecarHash(image), ReferenceSourceSHA256: compiled.Digest, ReferenceSidecarSHA256: sidecarHash(metadata), Bytes: len(image)}
	record.State.Kind, record.State.Generation, record.State.Version = "kernel", 1, 1
	record.State.SourceDigest, record.State.CompilerKey = compiled.Digest, compiled.Key
	positive, err := sidecarAttribute(record, image, compiled, metadata, dis)
	if err != nil {
		t.Fatal(err)
	}
	if err := sidecarValidateStatic(positive, record, image, compiled, metadata, dis); err != nil {
		t.Fatal(err)
	}
	classes, outer, unlocated := 0, 0, 0
	for _, row := range positive.Rows {
		switch row.Category {
		case "e-class":
			classes++
		case "outer source":
			outer++
		case "unlocated":
			unlocated++
		}
	}
	if classes == 0 || outer == 0 {
		t.Fatal("recorded class/outer remainder missing")
	}
	t.Logf("recorded rows: class=%d outer=%d unlocated=%d", classes, outer, unlocated)
	t.Run("foreign-filename", func(t *testing.T) {
		c := compiled
		c.Filename = filepath.Join(filepath.Dir(filename), "foreign.cu")
		if _, err := sidecarAttribute(record, image, c, metadata, dis); err == nil {
			t.Fatal("foreign filename accepted")
		}
	})
	t.Run("foreign-instruction", func(t *testing.T) {
		other, err := cudasass.ParseString(string(raw))
		if err != nil {
			t.Fatal(err)
		}
		changed := false
		for _, fn := range other.Functions {
			if fn.Name != "chain" {
				continue
			}
			for i := range fn.Instructions {
				if fn.Instructions[i].SourceFile == filename {
					fn.Instructions[i].SourceFile = "/foreign.cu"
					changed = true
					break
				}
			}
			break
		}
		if !changed {
			t.Fatal("recorded instruction not found")
		}
		if _, err := sidecarAttribute(record, image, compiled, metadata, other); err == nil {
			t.Fatal("foreign instruction accepted")
		}
	})
	t.Run("foreign-image", func(t *testing.T) {
		if _, err := sidecarAttribute(record, []byte("another image"), compiled, metadata, dis); err == nil {
			t.Fatal("foreign image accepted")
		}
	})
	t.Run("stale-metadata", func(t *testing.T) {
		if _, err := sidecarAttribute(record, image, compiled, append(metadata, '\n'), dis); err == nil {
			t.Fatal("stale metadata accepted")
		}
	})
	for _, field := range []string{"pc", "class", "line", "file", "origin"} {
		t.Run(field, func(t *testing.T) {
			bad := positive
			bad.Rows = append([]sidecarStaticRow(nil), positive.Rows...)
			for i := range bad.Rows {
				row := &bad.Rows[i]
				if row.Class == nil {
					continue
				}
				switch field {
				case "pc":
					row.PC = ^uint64(0)
				case "class":
					c := *row.Class + 999
					row.Class = &c
				case "line":
					row.Line++
				case "file":
					row.File = "/foreign.cu"
				case "origin":
					row.Origin = "invented"
				}
				break
			}
			if sidecarValidateStatic(bad, record, image, compiled, metadata, dis) == nil {
				t.Fatal("contradictory static receipt accepted")
			}
		})
	}
	t.Run("edited-unavailable", func(t *testing.T) {
		c, r := compiled, record
		c.Code += "\n// edited\n"
		c.Digest = sidecarHash([]byte(c.Code))
		r.SourceSHA256, r.State.SourceDigest = c.Digest, c.Digest
		got, err := sidecarAttribute(r, image, c, metadata, nil)
		if err != nil || got.Status != "edited source; original mapping unavailable" || len(got.Rows) != 0 {
			t.Fatalf("edited attribution = %+v, %v", got, err)
		}
	})
}

// The record's owner fields are simulated here: this is compiler/disassembler
// preparation, not a successful Load, actual Pin, or GPU attribution receipt.
func TestSidecarStaticCompilerPreparationCPU(t *testing.T) {
	root := os.Getenv("SIDECAR_ATTRIBUTION_PREPARE")
	if root == "" {
		t.Skip("explicit compiler/disassembler preparation only")
	}
	if !filepath.IsAbs(root) {
		t.Fatal("absolute preparation root required")
	}
	for _, name := range []string{"bar", "fused_rmsnorm_gemv"} {
		t.Run(name, func(t *testing.T) {
			code, err := sidecarReadFile(filepath.Join(root, name+".cu"))
			if err != nil {
				t.Fatal(err)
			}
			metadata, err := sidecarReadFile(filepath.Join(root, name+".egraph.json"))
			if err != nil {
				t.Fatal(err)
			}
			ptx, err := sidecarReadFile(filepath.Join(root, name+".ptx"))
			if err != nil {
				t.Fatal(err)
			}
			image, err := sidecarReadFile(filepath.Join(root, name+".cubin"))
			if err != nil {
				t.Fatal(err)
			}
			c := sidecarImageSource{Digest: sidecarHash(code), Key: "simulated-compiler-owner", Entry: "chain", Filename: filepath.Join(root, name+".cu"), Code: string(code), PTX: string(ptx)}
			r := sidecarImageRecord{Function: 123, SourceSHA256: c.Digest, CompilerKey: c.Key, Entry: c.Entry, CUBINSHA256: sidecarHash(image), ReferenceSourceSHA256: c.Digest, ReferenceSidecarSHA256: sidecarHash(metadata), Bytes: len(image)}
			r.State.Kind, r.State.Generation, r.State.Version = "kernel", 1, 1
			r.State.SourceDigest, r.State.CompilerKey = c.Digest, c.Key
			dest := t.TempDir()
			if retained := os.Getenv("SIDECAR_ATTRIBUTION_OUTPUT"); retained != "" {
				if !filepath.IsAbs(retained) {
					t.Fatal("absolute CPU attribution output required")
				}
				dest = filepath.Join(retained, name)
				if err := os.MkdirAll(filepath.Dir(dest), 0700); err != nil {
					t.Fatal(err)
				}
				if err := os.Mkdir(dest, 0700); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(dest, "CPU-PREPARATION-NOT-LOADED.txt"), []byte("CPU compiler/disassembler preparation; owner fields simulated; no successful Load, Pin or GPU execution\n"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			if err := sidecarStaticAttribution(context.Background(), dest, name, r, image, c, metadata); err != nil {
				t.Fatal(err)
			}
		})
	}
}
