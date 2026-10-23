// Egvariants writes hopper kernel variants and shape dispatchers for
// offline tuning.
//
// Usage:
//
//	egvariants [-sweep] -o dir spec.json
//
// The spec lists kernel configurations by kind, with the field names of
// the hopper types, and dispatchers over the resulting variant names:
//
//	{
//	  "skinny_gemm": [{"Name": "qkv", "N": 2304, "K": 8192, "MaxM": 8, "Weight": "bf16", "Out": "bf16"}],
//	  "tile_gemm": [{"Name": "qkv", "N": 2304, "K": 8192, "Weight": "bf16", "Out": "bf16"}],
//	  "paged_attention": [{"Name": "attn", "Hq": 16, "Hkv": 2, "QLen": 1, "Window": 2048, "Page": 1, "Q": "e4m3", "Out": "f32"}],
//	  "dispatch": [{"name": "qkv_dispatch", "routes": [{"max": 8, "variant": "qkv"}, {"variant": "qkv"}]}]
//	}
//
// With -sweep, each configuration expands to every valid tile, stage and
// split setting (see hopper.SweepTileGEMM and friends), named after it.
// Egvariants writes dir/<variant>.cu (self-contained: prelude, kernel and
// launcher), dir/<dispatcher>.cu and dir/manifest.json, which records each
// variant's op, shape key, launch configuration, entry point, parameters
// and source SHA-256.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

type spec struct {
	SkinnyGEMM     []hopper.SkinnyGEMM     `json:"skinny_gemm"`
	TileGEMM       []hopper.TileGEMM       `json:"tile_gemm"`
	PagedAttention []hopper.PagedAttention `json:"paged_attention"`
	Dispatch       []struct {
		Name   string `json:"name"`
		Routes []struct {
			Max     int    `json:"max"`
			Variant string `json:"variant"`
		} `json:"routes"`
	} `json:"dispatch"`
}

func main() {
	os.Exit(run(os.Args[1:], os.Stderr))
}

func run(args []string, stderr io.Writer) int {
	fs := flag.NewFlagSet("egvariants", flag.ContinueOnError)
	fs.SetOutput(stderr)
	sweep := fs.Bool("sweep", false, "expand each configuration over its tunables")
	out := fs.String("o", "", "output directory")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "usage: egvariants [-sweep] -o dir spec.json\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 1 || *out == "" {
		fs.Usage()
		return 2
	}
	if err := write(fs.Arg(0), *out, *sweep); err != nil {
		fmt.Fprintf(stderr, "egvariants: %v\n", err)
		return 1
	}
	return 0
}

func write(specPath, dir string, sweep bool) error {
	data, err := os.ReadFile(specPath)
	if err != nil {
		return err
	}
	var s spec
	if err := json.Unmarshal(data, &s); err != nil {
		return fmt.Errorf("%s: %w", specPath, err)
	}
	vs, err := variants(s, sweep)
	if err != nil {
		return err
	}
	if err := hopper.WriteVariants(dir, vs); err != nil {
		return err
	}
	byName := make(map[string]hopper.Variant)
	for _, v := range vs {
		byName[v.Name] = v
	}
	for _, d := range s.Dispatch {
		var routes []hopper.Route
		for _, r := range d.Routes {
			v, ok := byName[r.Variant]
			if !ok {
				return fmt.Errorf("dispatch %s: no variant %q", d.Name, r.Variant)
			}
			routes = append(routes, hopper.Route{Max: r.Max, Variant: v})
		}
		src, err := hopper.Dispatcher(d.Name, routes)
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(dir, d.Name+".cu"), []byte(src), 0o644); err != nil {
			return err
		}
	}
	return nil
}

func variants(s spec, sweep bool) ([]hopper.Variant, error) {
	var vs []hopper.Variant
	add := func(more []hopper.Variant, err error) error {
		vs = append(vs, more...)
		return err
	}
	one := func(v hopper.Variant, err error) ([]hopper.Variant, error) { return []hopper.Variant{v}, err }
	for _, g := range s.SkinnyGEMM {
		f := func() ([]hopper.Variant, error) { return one(g.Variant()) }
		if sweep {
			f = func() ([]hopper.Variant, error) { return hopper.SweepSkinnyGEMM(g) }
		}
		if err := add(f()); err != nil {
			return nil, err
		}
	}
	for _, g := range s.TileGEMM {
		f := func() ([]hopper.Variant, error) { return one(g.Variant()) }
		if sweep {
			f = func() ([]hopper.Variant, error) { return hopper.SweepTileGEMM(g) }
		}
		if err := add(f()); err != nil {
			return nil, err
		}
	}
	for _, a := range s.PagedAttention {
		f := func() ([]hopper.Variant, error) { return one(a.Variant()) }
		if sweep {
			f = func() ([]hopper.Variant, error) { return hopper.SweepPagedAttention(a) }
		}
		if err := add(f()); err != nil {
			return nil, err
		}
	}
	return vs, nil
}
