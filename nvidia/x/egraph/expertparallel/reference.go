package expertparallel

import (
	"fmt"
	"math"
	"sort"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// Layout is one rank's routes ordered by destination rank, as the layout
// kernel writes them.
type Layout struct {
	SendOff   []int32 // [Ranks+1] first send position of each destination
	SendPos   []int32 // route -> send position
	SendRoute []int32 // send position -> route
}

// LayoutReference orders the routes ids (global expert ids) stably by
// destination rank, ids[i] / localExperts.
func LayoutReference(ids []int32, ranks, localExperts int) Layout {
	l := Layout{SendOff: make([]int32, ranks+1), SendPos: make([]int32, len(ids)), SendRoute: make([]int32, len(ids))}
	for _, e := range ids {
		l.SendOff[int(e)/localExperts+1]++
	}
	for d := range ranks {
		l.SendOff[d+1] += l.SendOff[d]
	}
	next := append([]int32(nil), l.SendOff[:ranks]...)
	for i, e := range ids {
		d := int(e) / localExperts
		l.SendPos[i] = next[d]
		l.SendRoute[next[d]] = int32(i)
		next[d]++
	}
	return l
}

// Reference runs the plan on the CPU, rank by rank and collective by
// collective, and returns out [Tokens, Hidden] rounded to BF16. env holds
// [Config.Inputs] values as [hopper.Eval] reads them. Route uses
// [hopper.RowTopK.Reference]; the expert stage is the optimized term
// evaluated by [hopper.Eval] (float64 GEMMs); combine accumulates in FP32
// with fused multiply-adds in increasing expert order.
func (p *Plan) Reference(env map[string]hopper.Value) ([]float32, error) {
	R, S, K, L, C, H := p.Ranks, p.TokensPerRank(), p.TopK, p.LocalExperts(), p.Capacity(), p.Hidden
	E, I := p.Experts, p.Inter
	for _, t := range p.Inputs() {
		v, ok := env[t.Name]
		if !ok || len(v.Data) != size(t.Shape) {
			return nil, fmt.Errorf("expertparallel reference: input %s missing or not %v", t.Name, t.Shape)
		}
	}
	f32 := func(v []float64) []float32 {
		out := make([]float32, len(v))
		for i, f := range v {
			out[i] = float32(f)
		}
		return out
	}
	var bias []float32
	if p.Route.Bias != 0 {
		bias = f32(env["bias"].Data)
	}
	logits, x := env["logits"].Data, env["x"].Data

	ids := make([][]int32, R)
	wts := make([][]float32, R)
	lay := make([]Layout, R)
	for r := range R {
		var err error
		ids[r], wts[r], err = p.Route.Reference(S, f32(logits[r*S*E:(r+1)*S*E]), bias)
		if err != nil {
			return nil, fmt.Errorf("expertparallel reference: %w", err)
		}
		lay[r] = LayoutReference(ids[r], R, L)
	}

	// base[d][s] is where source s's routes start in rank d's receive rows.
	base := make([][]int, R)
	y := make([][]float64, R)
	for d := range R {
		base[d] = make([]int, R+1)
		recvX := make([]float64, C*H)
		recvIDs := make([]float64, C)
		ones := make([]float64, C)
		for j := range C {
			recvIDs[j] = float64((d*L + L) % E)
			ones[j] = 1
		}
		for s := range R {
			lo, hi := int(lay[s].SendOff[d]), int(lay[s].SendOff[d+1])
			for j := lo; j < hi; j++ {
				i := int(lay[s].SendRoute[j])
				row := base[d][s] + j - lo
				t := s*S + i/K
				copy(recvX[row*H:(row+1)*H], x[t*H:(t+1)*H])
				recvIDs[row] = float64(ids[s][i])
			}
			base[d][s+1] = base[d][s] + hi - lo
		}
		denv := map[string]hopper.Value{
			"recv_x":      {Shape: []int{C, H}, Data: recvX},
			"recv_ids":    {Shape: []int{C, 1}, Data: recvIDs},
			"recv_w":      {Shape: []int{C, 1}, Data: ones},
			"local_start": {Shape: []int{1}, Data: []float64{float64(d * L)}},
			"post_w":      env["post_w"],
		}
		for _, w := range []struct {
			prefix string
			n, k   int
		}{{"w13", 2 * I, H}, {"w2", H, I}} {
			for _, t := range nvfp4Inputs(w.prefix, L, w.n, w.k) {
				per := size(t.Shape) / L
				denv[t.Name] = hopper.Value{Shape: t.Shape, Data: env[t.Name].Data[d*L*per : (d+1)*L*per]}
			}
		}
		v, err := hopper.Eval(p.expertExpr, denv)
		if err != nil {
			return nil, fmt.Errorf("expertparallel reference: expert stage rank %d: %w", d, err)
		}
		y[d] = v.Data
	}

	var shared []float64
	if p.Shared != nil {
		for r := range R {
			senv := map[string]hopper.Value{
				"x":              {Shape: []int{S, H}, Data: x[r*S*H : (r+1)*S*H]},
				"shared_offsets": {Shape: []int{2}, Data: []float64{0, float64(S)}},
			}
			for _, name := range []string{"s13_q", "s13_sf", "s13_gs", "s2_q", "s2_sf", "s2_gs"} {
				senv[name] = env[name]
			}
			v, err := hopper.Eval(p.sharedExpr, senv)
			if err != nil {
				return nil, fmt.Errorf("expertparallel reference: shared expert rank %d: %w", r, err)
			}
			shared = append(shared, v.Data...)
		}
	}

	out := make([]float32, p.Tokens*H)
	type route struct {
		e   int32
		row []float64
		w   float32
	}
	for r := range R {
		for t := range S {
			rs := make([]route, K)
			for k := range K {
				i := t*K + k
				e := ids[r][i]
				d := int(e) / L
				row := base[d][r] + int(lay[r].SendPos[i]-lay[r].SendOff[d])
				rs[k] = route{e, y[d][row*H : (row+1)*H], wts[r][i]}
			}
			sort.Slice(rs, func(a, b int) bool { return rs[a].e < rs[b].e })
			o := out[(r*S+t)*H : (r*S+t+1)*H]
			for c := range H {
				var acc float32
				for _, rt := range rs {
					acc = fma32(float32(rt.row[c]), rt.w, acc)
				}
				if shared != nil {
					acc += float32(shared[(r*S+t)*H+c])
				}
				o[c] = hopper.BF16.Round(acc)
			}
		}
	}
	return out, nil
}

// fma32 is a·b+c rounded once to float32: a·b is exact in float64, so only
// the sum rounds twice, which can differ from fmaf in rare halfway cases.
func fma32(a, b, c float32) float32 {
	return float32(math.FMA(float64(a), float64(b), float64(c)))
}
