//go:build ignore

// Gen regenerates forward.sass, the disassembly this example prints.
//
// It compiles forward.cu and forward_bf16.cu the same way the program does,
// with NVRTC for a named physical architecture, and disassembles the resulting
// cubin with nvdisasm. Taking the listing from the compiler rather than writing it by
// hand is what makes the printed instructions, addresses and kernel headers
// the ones the hardware was actually given.
//
// Usage:
//
//	go run gen.go -o forward.sass
//
// It needs NVRTC and an nvdisasm that understands the target. It lives beside
// forward.cu rather than under testdata because go:embed cannot reach a
// parent directory, and embedding the source at build time is what lets the
// generator run on a remote GPU host from a machine with no CUDA toolchain:
//
//	GOOS=linux GOARCH=arm64 go run gen.go > forward.sass
//
// Only the compiled binary travels, so nothing on the remote host has to hold
// a copy of the tree.
package main

import (
	"context"
	_ "embed"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed forward.cu
var source string

//go:embed forward_bf16.cu
var bf16Source string

func main() {
	log.SetFlags(0)
	log.SetPrefix("gen: ")
	arch := flag.String("arch", "sm_121", "physical architecture to compile for")
	out := flag.String("o", "", "write the disassembly to this `file` instead of stdout")
	flag.Parse()

	if err := run(context.Background(), *arch, *out); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, arch, out string) error {
	mod, err := nvrtcx.Compile(source+"\n"+bf16Source,
		nvrtcx.WithName("llama_forward.cu"),
		nvrtcx.WithPhysicalArchTarget(arch),
	)
	if err != nil {
		// When NVRTC cannot destroy the program, the error holds it and
		// implements io.Closer to retry. gen.go builds alone, so this does
		// what the program's releaseErrorOwner does.
		var c io.Closer
		if errors.As(err, &c) {
			if cerr := c.Close(); cerr != nil {
				err = errors.Join(err, fmt.Errorf("release resources held by failed call: %w", cerr))
			}
		}
		return fmt.Errorf("compile for %s: %w", arch, err)
	}
	sass, err := cudasass.Disassembler{}.Disassemble(ctx, mod.CUBIN(), cudasass.Options{PrintCode: true})
	if err != nil {
		return fmt.Errorf("disassemble: %w", err)
	}
	if out == "" {
		_, err := os.Stdout.Write(sass.Stdout)
		return err
	}
	return os.WriteFile(out, sass.Stdout, 0o644)
}
