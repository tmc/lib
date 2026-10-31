// Egvariants writes hopper kernel variants and shape dispatchers for
// offline tuning.
//
// Usage:
//
//	egvariants [-sweep] [-ptxas file.log] [-rank-device name] -o dir spec.json
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
// With -ptxas, variants failing the static ptxas gate (register spills,
// wgmma serialization, or occupancy) are filtered out. With -rank-device,
// surviving variants are ranked per shape bucket and emitted as a
// shape-routed dispatcher (<name>_dispatch.cu) alongside ranked.json.
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
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/qualify"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

type spec struct {
	SkinnyGEMM     []hopper.SkinnyGEMM     `json:"skinny_gemm"`
	TileGEMM       []hopper.TileGEMM       `json:"tile_gemm"`
	PagedAttention []hopper.PagedAttention `json:"paged_attention"`
	Dispatch       []struct {
		Name   string `json:"name"`
		Shapes []int  `json:"shapes,omitempty"`
		Routes []struct {
			Max     int    `json:"max"`
			Variant string `json:"variant"`
		} `json:"routes"`
	} `json:"dispatch"`
}

type rankedGroup struct {
	Key        string                        `json:"key"`
	Dispatcher string                        `json:"dispatcher"`
	Winners    map[int]string                `json:"winners"`
	Ranked     []qualify.RankedHopperVariant `json:"ranked"`
}

func main() {
	os.Exit(run(os.Args[1:], os.Stderr))
}

func run(args []string, stderr io.Writer) int {
	fs := flag.NewFlagSet("egvariants", flag.ContinueOnError)
	fs.SetOutput(stderr)
	sweep := fs.Bool("sweep", false, "expand each configuration over its tunables")
	ptxasFile := fs.String("ptxas", "", "filter variants against a ptxas -v log file")
	rankDevice := fs.String("rank-device", "", "rank variants for target device and emit shape dispatchers")
	out := fs.String("o", "", "output directory")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "usage: egvariants [-sweep] [-ptxas file.log] [-rank-device name] -o dir spec.json\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 1 || *out == "" {
		fs.Usage()
		return 2
	}
	if err := write(fs.Arg(0), *out, *sweep, *ptxasFile, *rankDevice); err != nil {
		fmt.Fprintf(stderr, "egvariants: %v\n", err)
		return 1
	}
	return 0
}

func write(specPath, dir string, sweep bool, ptxasFile, rankDevice string) error {
	data, err := os.ReadFile(specPath)
	if err != nil {
		return err
	}
	var s spec
	if err := json.Unmarshal(data, &s); err != nil {
		return fmt.Errorf("%s: %w", specPath, err)
	}
	vs, baseByKey, err := variantsWithBases(s, sweep)
	if err != nil {
		return err
	}
	if ptxasFile != "" {
		f, err := os.Open(ptxasFile)
		if err != nil {
			return err
		}
		rep, err := ptxasinfo.Parse(f)
		f.Close()
		if err != nil {
			return fmt.Errorf("%s: %w", ptxasFile, err)
		}
		if len(rep.Errors) > 0 {
			return fmt.Errorf("%s: ptxas error: %s", ptxasFile, rep.Errors[0])
		}
		vs, _ = qualify.FilterHopperVariantsByPTXAS(vs, *rep, "", qualify.StaticGate{NoSpills: true})
		if len(vs) == 0 {
			return fmt.Errorf("%s: all variants rejected by ptxas gate", ptxasFile)
		}
	}
	var cfg hopper.CostConfig
	if rankDevice != "" {
		dev, err := resolveRankDevice(rankDevice)
		if err != nil {
			return err
		}
		cfg = hopper.CostConfig{Device: dev}
	}
	if err := hopper.WriteVariants(dir, vs); err != nil {
		return err
	}
	byName := make(map[string]hopper.Variant, len(vs))
	var keyOrder []string
	byKey := make(map[string][]hopper.Variant)
	for _, v := range vs {
		byName[v.Name] = v
		if _, seen := byKey[v.Key]; !seen {
			keyOrder = append(keyOrder, v.Key)
		}
		byKey[v.Key] = append(byKey[v.Key], v)
	}

	if rankDevice != "" {
		var groups []rankedGroup
		for _, key := range keyOrder {
			gvs := byKey[key]
			shapes := []int{1, 8, 32, 128}
			if gvs[0].Op == "paged_attention" {
				shapes = []int{1, 8, 32}
			}
			winners := make(map[int]hopper.Variant, len(shapes))
			winNames := make(map[int]string, len(shapes))
			for _, sh := range shapes {
				rk := qualify.RankHopperVariants(gvs, qualify.TuningShape{M: sh, Batch: sh, MaxKeys: 80000}, cfg)
				winners[sh] = rk[0].Variant
				winNames[sh] = rk[0].Variant.Name
			}
			dispName := baseByKey[key]
			if dispName == "" {
				dispName = gvs[0].Name
			}
			if !strings.HasSuffix(dispName, "_dispatch") {
				dispName += "_dispatch"
			}
			src, err := qualify.BuildHopperShapeDispatcher(dispName, shapes, winners)
			if err != nil {
				return err
			}
			if err := os.WriteFile(filepath.Join(dir, dispName+".cu"), []byte(src), 0o644); err != nil {
				return err
			}
			repRanked := qualify.RankHopperVariants(gvs, qualify.TuningShape{M: shapes[len(shapes)-1], Batch: shapes[0], MaxKeys: 80000}, cfg)
			groups = append(groups, rankedGroup{
				Key:        key,
				Dispatcher: dispName,
				Winners:    winNames,
				Ranked:     repRanked,
			})
		}
		rdata, err := json.MarshalIndent(groups, "", "  ")
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(dir, "ranked.json"), append(rdata, '\n'), 0o644); err != nil {
			return err
		}
	}

	for _, d := range s.Dispatch {
		if len(d.Routes) == 0 && len(d.Shapes) > 0 && rankDevice != "" && len(keyOrder) > 0 {
			gvs := byKey[keyOrder[0]]
			winners := make(map[int]hopper.Variant, len(d.Shapes))
			for _, sh := range d.Shapes {
				m := max(sh, 1)
				rk := qualify.RankHopperVariants(gvs, qualify.TuningShape{M: m, Batch: m, MaxKeys: 80000}, cfg)
				winners[sh] = rk[0].Variant
			}
			src, err := qualify.BuildHopperShapeDispatcher(d.Name, d.Shapes, winners)
			if err != nil {
				return err
			}
			if err := os.WriteFile(filepath.Join(dir, d.Name+".cu"), []byte(src), 0o644); err != nil {
				return err
			}
			continue
		}
		var routes []hopper.Route
		for _, r := range d.Routes {
			v, ok := byName[r.Variant]
			if !ok && rankDevice != "" {
				v, ok = bestMatchingPrefix(vs, r.Variant, max(r.Max, 128), cfg)
			}
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

func resolveRankDevice(dev string) (string, error) {
	check := func(name string) bool {
		if _, err := cudaroofline.LookupDeviceCeiling(name, cudaroofline.DeviceResident); err == nil {
			return true
		}
		if _, err := cudaroofline.LookupCeiling(name, cudaroofline.DeviceResident); err == nil {
			return true
		}
		return false
	}
	if check(dev) {
		return dev, nil
	}
	if base := filepath.Base(dev); base != dev && check(base) {
		return base, nil
	}
	return "", fmt.Errorf("unknown rank device %q", dev)
}

func bestMatchingPrefix(vs []hopper.Variant, prefix string, m int, cfg hopper.CostConfig) (hopper.Variant, bool) {
	var cands []hopper.Variant
	for _, v := range vs {
		if v.Name == prefix || strings.HasPrefix(v.Name, prefix+"_") {
			cands = append(cands, v)
		}
	}
	if len(cands) == 0 {
		return hopper.Variant{}, false
	}
	rk := qualify.RankHopperVariants(cands, qualify.TuningShape{M: m, Batch: m, MaxKeys: 80000}, cfg)
	return rk[0].Variant, true
}

func variantsWithBases(s spec, sweep bool) ([]hopper.Variant, map[string]string, error) {
	var vs []hopper.Variant
	baseByKey := make(map[string]string)
	add := func(baseName string, more []hopper.Variant, err error) error {
		if err != nil {
			return err
		}
		for _, v := range more {
			if _, ok := baseByKey[v.Key]; !ok {
				baseByKey[v.Key] = trimKindSuffix(baseName)
			}
		}
		vs = append(vs, more...)
		return nil
	}
	one := func(v hopper.Variant, err error) ([]hopper.Variant, error) { return []hopper.Variant{v}, err }
	for _, g := range s.SkinnyGEMM {
		f := func() ([]hopper.Variant, error) { return one(g.Variant()) }
		if sweep {
			f = func() ([]hopper.Variant, error) { return hopper.SweepSkinnyGEMM(g) }
		}
		more, err := f()
		if err := add(g.Name, more, err); err != nil {
			return nil, nil, err
		}
	}
	for _, g := range s.TileGEMM {
		f := func() ([]hopper.Variant, error) { return one(g.Variant()) }
		if sweep {
			f = func() ([]hopper.Variant, error) { return hopper.SweepTileGEMM(g) }
		}
		more, err := f()
		if err := add(g.Name, more, err); err != nil {
			return nil, nil, err
		}
	}
	for _, a := range s.PagedAttention {
		f := func() ([]hopper.Variant, error) { return one(a.Variant()) }
		if sweep {
			f = func() ([]hopper.Variant, error) { return hopper.SweepPagedAttention(a) }
		}
		more, err := f()
		if err := add(a.Name, more, err); err != nil {
			return nil, nil, err
		}
	}
	return vs, baseByKey, nil
}

func trimKindSuffix(name string) string {
	for _, suf := range []string{"_skinny", "_tile", "_small", "_large"} {
		if trimmed := strings.TrimSuffix(name, suf); trimmed != name && trimmed != "" {
			return trimmed
		}
	}
	return name
}
