// Egvariants writes hopper kernel variants and shape dispatchers for
// offline tuning.
//
// Usage:
//
//	egvariants [-sweep] [-ptxas file.log] [-ptxas-refine] [-rank-device name] [-pgo-cache-dir dir] -o dir spec.json
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
// wgmma serialization, or occupancy) are filtered out. With -ptxas-refine
// (alongside -ptxas), each configuration step is also refined via
// qualify.ObservationFromPTXAS and qualify.RefineStep and summarized in
// dir/refined.json. With -rank-device, surviving variants are ranked per
// shape bucket and emitted as a shape-routed dispatcher (<name>_dispatch.cu)
// alongside ranked.json.
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
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
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

type refinedStepEntry struct {
	Name        string                    `json:"name"`
	Op          string                    `json:"op"`
	Observation qualify.KernelObservation `json:"observation"`
	Bottlenecks []qualify.BottleneckKind  `json:"bottlenecks,omitempty"`
	Actions     []string                  `json:"actions,omitempty"`
	Refined     bool                      `json:"refined"`
	Initial     hopper.PlanKernel         `json:"initial"`
	RefinedStep hopper.PlanKernel         `json:"refined_step"`
	Variant     *hopper.Variant           `json:"variant,omitempty"`
}

type refinedSummary struct {
	PTXASFile string             `json:"ptxas_file"`
	Device    string             `json:"device,omitempty"`
	DeviceSMs int                `json:"device_sms"`
	Steps     []refinedStepEntry `json:"steps"`
	Actions   []string           `json:"actions,omitempty"`
}

type variantOptions struct {
	specPath    string
	dir         string
	sweep       bool
	ptxasFile   string
	ptxasRefine bool
	rankDevice  string
	pgoCacheDir string
}

func main() {
	os.Exit(run(os.Args[1:], os.Stderr))
}

func run(args []string, stderr io.Writer) int {
	fs := flag.NewFlagSet("egvariants", flag.ContinueOnError)
	fs.SetOutput(stderr)
	sweep := fs.Bool("sweep", false, "expand each configuration over its tunables")
	ptxasFile := fs.String("ptxas", "", "filter variants against a ptxas -v log file")
	ptxasRefine := fs.Bool("ptxas-refine", false, "refine spec steps from ptxas -v observations and write refined.json")
	ptxasLoop := fs.Bool("ptxas-loop", false, "alias for -ptxas-refine")
	rankDevice := fs.String("rank-device", "", "rank variants for target device and emit shape dispatchers")
	pgoCacheDir := fs.String("pgo-cache-dir", "", "optional verified PGO cache directory to inspect")
	out := fs.String("o", "", "output directory")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "usage: egvariants [-sweep] [-ptxas file.log] [-ptxas-refine] [-rank-device name] [-pgo-cache-dir dir] -o dir spec.json\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 1 || *out == "" {
		fs.Usage()
		return 2
	}
	opts := variantOptions{
		specPath:    fs.Arg(0),
		dir:         *out,
		sweep:       *sweep,
		ptxasFile:   strings.TrimSpace(*ptxasFile),
		ptxasRefine: *ptxasRefine || *ptxasLoop,
		rankDevice:  strings.TrimSpace(*rankDevice),
		pgoCacheDir: strings.TrimSpace(*pgoCacheDir),
	}
	if err := writeWithOptions(opts); err != nil {
		fmt.Fprintf(stderr, "egvariants: %v\n", err)
		return 1
	}
	return 0
}

func write(specPath, dir string, sweep bool, ptxasFile, rankDevice string) error {
	return writeWithOptions(variantOptions{
		specPath:   specPath,
		dir:        dir,
		sweep:      sweep,
		ptxasFile:  ptxasFile,
		rankDevice: rankDevice,
	})
}

func writeWithOptions(opts variantOptions) error {
	if opts.ptxasRefine && opts.ptxasFile == "" {
		return fmt.Errorf("-ptxas-refine requires -ptxas <file.log>")
	}
	data, err := os.ReadFile(opts.specPath)
	if err != nil {
		return err
	}
	var s spec
	if err := json.Unmarshal(data, &s); err != nil {
		return fmt.Errorf("%s: %w", opts.specPath, err)
	}
	vs, baseByKey, err := variantsWithBases(s, opts.sweep)
	if err != nil {
		return err
	}
	var cfg hopper.CostConfig
	if opts.rankDevice != "" {
		dev, err := resolveRankDevice(opts.rankDevice)
		if err != nil {
			return err
		}
		cfg = hopper.CostConfig{Device: dev}
	}

	var refSummary *refinedSummary
	if opts.ptxasFile != "" {
		f, err := os.Open(opts.ptxasFile)
		if err != nil {
			return err
		}
		rep, err := ptxasinfo.Parse(f)
		f.Close()
		if err != nil {
			return fmt.Errorf("%s: %w", opts.ptxasFile, err)
		}
		if len(rep.Errors) > 0 {
			return fmt.Errorf("%s: ptxas error: %s", opts.ptxasFile, rep.Errors[0])
		}

		var refinedVariants []hopper.Variant
		if opts.ptxasRefine {
			refSummary, refinedVariants, err = refineSpecFromPTXAS(s, *rep, opts.ptxasFile, cfg.Device)
			if err != nil {
				return err
			}
		}

		filtered, _ := qualify.FilterHopperVariantsByPTXAS(vs, *rep, "", qualify.StaticGate{NoSpills: true})
		if opts.ptxasRefine && len(refinedVariants) > 0 {
			byNameFiltered := make(map[string]bool, len(filtered))
			for _, fv := range filtered {
				byNameFiltered[fv.Name] = true
			}
			for _, rv := range refinedVariants {
				if _, ok := baseByKey[rv.Key]; !ok {
					baseByKey[rv.Key] = trimKindSuffix(rv.Name)
				}
				if !byNameFiltered[rv.Name] {
					filtered = append(filtered, rv)
					byNameFiltered[rv.Name] = true
				} else {
					for i := range filtered {
						if filtered[i].Name == rv.Name {
							filtered[i] = rv
						}
					}
				}
			}
		}
		vs = filtered
		if len(vs) == 0 {
			return fmt.Errorf("%s: all variants rejected by ptxas gate", opts.ptxasFile)
		}
	}

	if err := hopper.WriteVariants(opts.dir, vs); err != nil {
		return err
	}
	if refSummary != nil {
		rdata, err := json.MarshalIndent(refSummary, "", "  ")
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(opts.dir, "refined.json"), append(rdata, '\n'), 0o644); err != nil {
			return err
		}
	}
	if opts.pgoCacheDir != "" {
		cache := autotune.NewVerifiedPGOCache(opts.pgoCacheDir)
		keys, err := cache.List()
		if err != nil {
			return err
		}
		var proofs []autotune.PGOPromotionProof
		for _, k := range keys {
			p, _, err := cache.LoadVerified(k)
			if err != nil {
				return err
			}
			proofs = append(proofs, p)
		}
		pdata, err := json.MarshalIndent(proofs, "", "  ")
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(opts.dir, "pgo_cache.json"), append(pdata, '\n'), 0o644); err != nil {
			return err
		}
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

	if opts.rankDevice != "" {
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
			if err := os.WriteFile(filepath.Join(opts.dir, dispName+".cu"), []byte(src), 0o644); err != nil {
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
		if err := os.WriteFile(filepath.Join(opts.dir, "ranked.json"), append(rdata, '\n'), 0o644); err != nil {
			return err
		}
	}

	for _, d := range s.Dispatch {
		if len(d.Routes) == 0 && len(d.Shapes) > 0 && opts.rankDevice != "" && len(keyOrder) > 0 {
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
			if err := os.WriteFile(filepath.Join(opts.dir, d.Name+".cu"), []byte(src), 0o644); err != nil {
				return err
			}
			continue
		}
		var routes []hopper.Route
		for _, r := range d.Routes {
			v, ok := byName[r.Variant]
			if !ok && opts.rankDevice != "" {
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
		if err := os.WriteFile(filepath.Join(opts.dir, d.Name+".cu"), []byte(src), 0o644); err != nil {
			return err
		}
	}
	return nil
}

func refineSpecFromPTXAS(s spec, rep ptxasinfo.Report, ptxasFile, device string) (*refinedSummary, []hopper.Variant, error) {
	deviceSMs := 132
	if device != "" {
		if normed, err := (hopper.CostConfig{Device: device}).Norm(); err == nil && normed.SMs > 0 {
			deviceSMs = normed.SMs
		}
	}
	allFns := rep.Functions()
	usedFns := make([]bool, len(allFns))

	var steps []hopper.PlanKernel
	var names, ops []string
	for _, g := range s.SkinnyGEMM {
		cp := g
		m := max(cp.MaxM, 1)
		steps = append(steps, hopper.PlanKernel{GEMM: &cp, M: m})
		names = append(names, cp.Name)
		ops = append(ops, "skinny_gemm")
	}
	for _, t := range s.TileGEMM {
		cp := t
		m := max(cp.BM, 128)
		steps = append(steps, hopper.PlanKernel{Tile: &cp, M: m})
		names = append(names, cp.Name)
		ops = append(ops, "tile_gemm")
	}
	for _, a := range s.PagedAttention {
		cp := a
		maxKeys := cp.Window
		if maxKeys <= 0 {
			maxKeys = 1024
		}
		steps = append(steps, hopper.PlanKernel{Attention: &cp, Batch: 1, MaxKeys: maxKeys})
		names = append(names, cp.Name)
		ops = append(ops, "paged_attention")
	}

	out := &refinedSummary{
		PTXASFile: ptxasFile,
		Device:    device,
		DeviceSMs: deviceSMs,
		Steps:     make([]refinedStepEntry, 0, len(steps)),
	}
	var refinedVars []hopper.Variant

	for i, step := range steps {
		name := names[i]
		fn := matchStepPTXASFunction(rep, name, i, allFns, usedFns)
		gridBlocks := computeStepGridBlocks(step)
		obs := qualify.ObservationFromPTXAS(fn, gridBlocks, deviceSMs)
		if name != "" {
			obs.Name = name
		}
		if obs.Registers > 168 && obs.AchievedOccupancy == 0 {
			obs.AchievedOccupancy = 0.15
		}
		bns := qualify.ClassifyBottlenecks(obs)
		refStep, actions := qualify.RefineStep(step, obs)
		if slices.Contains(bns, qualify.BottleneckWGMMASerial) && len(actions) == 0 {
			if refStep.Tile != nil && refStep.Tile.WGMMA && refStep.Tile.Weight != hopper.WeightE4M3 {
				cand := *refStep.Tile
				cand.WGMMA = false
				cand.WarpSpecialized = false
				cand.Repacked = false
				cand.DequantWords = 0
				if _, err := cand.Norm(); err == nil {
					refStep.Tile = &cand
					actions = append(actions, fmt.Sprintf("%s: disable WGMMA to eliminate WGMMA serialization", cand.Name))
				}
			}
		}

		var hvPtr *hopper.Variant
		if hv, err := variantFromStep(refStep); err == nil {
			hvPtr = &hv
			if len(actions) > 0 {
				refinedVars = append(refinedVars, hv)
			}
		}
		out.Actions = append(out.Actions, actions...)
		out.Steps = append(out.Steps, refinedStepEntry{
			Name:        name,
			Op:          ops[i],
			Observation: obs,
			Bottlenecks: bns,
			Actions:     actions,
			Refined:     len(actions) > 0,
			Initial:     step,
			RefinedStep: refStep,
			Variant:     hvPtr,
		})
	}
	return out, refinedVars, nil
}

func matchStepPTXASFunction(rep ptxasinfo.Report, name string, idx int, allFns []*ptxasinfo.Function, used []bool) ptxasinfo.Function {
	if name != "" {
		for j, fn := range allFns {
			if fn == nil {
				continue
			}
			if fn.Name == name || qualify.MatchesVariantFunction(fn.Name, name) || strings.Contains(fn.Name, name) {
				used[j] = true
				return *fn
			}
		}
	}
	if idx >= 0 && idx < len(allFns) && !used[idx] && allFns[idx] != nil {
		used[idx] = true
		return *allFns[idx]
	}
	return ptxasinfo.Function{Name: name, Entry: true}
}

func computeStepGridBlocks(k hopper.PlanKernel) int {
	switch {
	case k.Tile != nil:
		t := *k.Tile
		if normed, err := t.Norm(); err == nil {
			t = normed
		}
		grid, _ := t.Grid(max(k.M, 128), 1)
		return max(grid[0], 1) * max(grid[1], 1) * max(grid[2], 1)
	case k.GEMM != nil:
		g := *k.GEMM
		if g.MaxM <= 0 {
			g.MaxM = max(k.M, 1)
		}
		if normed, err := g.Norm(); err == nil {
			g = normed
		}
		if grid, _, err := g.Grid(); err == nil {
			return max(grid[0], 1) * max(grid[1], 1) * max(grid[2], 1)
		}
	case k.Attention != nil:
		a := *k.Attention
		if normed, err := a.Norm(); err == nil {
			a = normed
		}
		grid, _ := a.Grid(max(k.Batch, 1), max(k.MaxKeys, 1024))
		return max(grid[0], 1) * max(grid[1], 1) * max(grid[2], 1)
	}
	return 1
}

func variantFromStep(k hopper.PlanKernel) (hopper.Variant, error) {
	switch {
	case k.GEMM != nil:
		return k.GEMM.Variant()
	case k.Tile != nil:
		return k.Tile.Variant()
	case k.Attention != nil:
		return k.Attention.Variant()
	default:
		return hopper.Variant{}, fmt.Errorf("unsupported step kind")
	}
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
