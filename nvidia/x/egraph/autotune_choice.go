package egraph

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"hash/fnv"
	"math/rand"
	"reflect"
	"time"

	"github.com/tmc/lib/egraphcore"
)

type tuningChoice struct {
	choice    map[Id]int
	candidate TuningCandidate
}

type tuningChoiceSpace struct {
	graph        *EGraph
	root         Id
	revision     uint64
	classes      []Id
	alternatives map[Id][]ENode
	fingerprints *egraphcore.ChoiceFingerprinter[Id, ENode]
	materializer *egraphcore.ChoiceMaterializer[Id, ENode, *Expr]
}

func newTuningChoiceSpace(g *EGraph, root Id) (*tuningChoiceSpace, error) {
	if err := g.RebuildChecked(); err != nil {
		return nil, fmt.Errorf("rebuild graph: %w", err)
	}
	root = g.Find(root)
	if g.Class(root) == nil {
		return nil, fmt.Errorf("missing root class %d", root)
	}
	choices, err := egraphcore.NewChoiceSpaceFrom(root,
		func(id Id) (Id, error) { return g.Find(id), nil },
		func(id Id) ([]ENode, error) {
			class := g.Class(id)
			if class == nil {
				return nil, fmt.Errorf("missing reachable class %d", id)
			}
			return class.Nodes, nil
		},
		func(node ENode) (ENode, error) {
			copy := node
			copy.Children = append([]Id(nil), node.Children...)
			for i := range copy.Children {
				copy.Children[i] = g.Find(copy.Children[i])
			}
			return copy, nil
		},
		func(node ENode) []Id { return node.Children },
		func(node ENode) bool { return !node.Subsumed },
		func(node ENode) ENode { node.Children = append([]Id(nil), node.Children...); return node },
		func(left, right ENode) bool { return left.Key() < right.Key() },
	)
	if err != nil {
		return nil, fmt.Errorf("build choice space: %w", err)
	}
	space := &tuningChoiceSpace{graph: g, root: root, revision: g.revision, classes: choices.Classes(), alternatives: make(map[Id][]ENode), fingerprints: egraphcore.NewChoiceFingerprinter[Id, ENode]()}
	for _, class := range space.classes {
		space.alternatives[class] = choices.Alternatives(class)
	}
	space.materializer, err = egraphcore.NewChoiceMaterializer[Id, ENode, *Expr](space.classes)
	if err != nil {
		return nil, fmt.Errorf("create choice materializer: %w", err)
	}
	return space, nil
}

func (s *tuningChoiceSpace) reachable(choice map[Id]int) ([]Id, error) {
	return s.materializer.ReachableClasses(s.root, choice,
		func(id Id) []ENode { return s.alternatives[id] },
		func(node ENode) []Id { return node.Children })
}

func (s *tuningChoiceSpace) repair(choice map[Id]int, rng *rand.Rand) (map[Id]int, error) {
	return egraphcore.RepairChoiceCycles(s.root, choice,
		func(id Id) []ENode { return s.alternatives[id] },
		func(node ENode) []Id { return node.Children }, nil, rng.Intn, 128)
}

func (s *tuningChoiceSpace) materialize(choice map[Id]int) (*Expr, error) {
	return s.materializer.Materialize(s.root, choice,
		func(id Id) []ENode { return s.alternatives[id] },
		func(node ENode) []Id { return node.Children },
		func(id Id, node ENode, children []*Expr) (*Expr, error) {
			return &Expr{Class: id, Origin: s.graph.nodeOrigin(node), Op: node.Op, Data: cloneData(node.Data), Children: children}, nil
		})
}

func (s *tuningChoiceSpace) fingerprint(choice map[Id]int) (string, error) {
	sum, err := s.fingerprints.Fingerprint(s.root, choice,
		func(id Id) []ENode { return s.alternatives[id] },
		func(node ENode) []Id { return node.Children },
		func(_ Id, node ENode, childHashes []egraphcore.ChoiceFingerprint) egraphcore.ChoiceFingerprint {
			h := sha256.New()
			var length [8]byte
			writeField := func(value string) {
				binary.LittleEndian.PutUint64(length[:], uint64(len(value)))
				_, _ = h.Write(length[:])
				_, _ = h.Write([]byte(value))
			}
			writeField(node.Op)
			if node.Data == nil {
				writeField("")
			} else {
				if value, ok := egraphcore.ValueFingerprint(node.Data); ok {
					writeField("value:" + value)
				} else {
					// Type-only buckets preserve equal values even when the
					// payload contains pointers or other unsupported fields.
					writeField(fmt.Sprintf("type:%T", node.Data))
				}
			}
			binary.LittleEndian.PutUint64(length[:], uint64(len(childHashes)))
			_, _ = h.Write(length[:])
			for _, childHash := range childHashes {
				_, _ = h.Write(childHash[:])
			}
			var fingerprint egraphcore.ChoiceFingerprint
			copy(fingerprint[:], h.Sum(nil))
			return fingerprint
		}, nil)
	if err != nil {
		return "", err
	}
	return hex.EncodeToString(sum[:]), nil
}

func tuningChoicesEqual(space *tuningChoiceSpace, left, right map[Id]int) bool {
	if space == nil {
		return false
	}
	selectedRightNode := func(class Id) (ENode, bool) {
		nodes := space.alternatives[class]
		index, ok := right[class]
		if !ok || index < 0 || index >= len(nodes) {
			return ENode{}, false
		}
		return nodes[index], true
	}
	return egraphcore.SelectedChoiceMatches(
		space.root,
		left,
		func(class Id) []ENode { return space.alternatives[class] },
		func(node ENode) []Id { return node.Children },
		space.root,
		func(class Id) []Id {
			node, ok := selectedRightNode(class)
			if !ok {
				return nil
			}
			return node.Children
		},
		func(_ Id, leftNode ENode, rightClass Id) bool {
			rightNode, ok := selectedRightNode(rightClass)
			return ok && leftNode.Op == rightNode.Op && reflect.DeepEqual(leftNode.Data, rightNode.Data)
		},
	)
}

type expressionEvaluator func(context.Context, *Expr, time.Time) (TuningCandidate, time.Duration, error)

func cloneTuningExpr(expr *Expr) *Expr {
	if expr == nil {
		return nil
	}
	copy := &Expr{Class: expr.Class, Origin: expr.Origin, Op: expr.Op, Data: cloneData(expr.Data), Children: make([]*Expr, len(expr.Children))}
	for i, child := range expr.Children {
		copy.Children[i] = cloneTuningExpr(child)
	}
	return copy
}

func cloneTuningCandidate(candidate TuningCandidate) TuningCandidate {
	candidate.Expr = cloneTuningExpr(candidate.Expr)
	return candidate
}

func (t *Tuner) searchChoices(ctx context.Context, g *EGraph, root Id, deadline time.Time, searchID string, evaluate expressionEvaluator, ranked *[]RankedTuningResult) (TuningCandidate, time.Duration, error) {
	if err := ctx.Err(); err != nil {
		return TuningCandidate{}, 0, err
	}
	if !time.Now().Before(deadline) {
		return TuningCandidate{}, 0, fmt.Errorf("autotune budget expired before choice snapshot")
	}
	space, err := newTuningChoiceSpace(g, root)
	if err != nil {
		return TuningCandidate{}, 0, err
	}
	if err := ctx.Err(); err != nil {
		return TuningCandidate{}, 0, err
	}
	if !time.Now().Before(deadline) {
		return TuningCandidate{}, 0, fmt.Errorf("autotune budget expired during choice snapshot")
	}
	population, err := egraphcore.NewMeasuredSearch[tuningChoice](egraphcore.MeasuredSearchConfig{
		KeepBest: 3, GenerationSize: 8, RestartStagnation: 3,
	})
	if err != nil {
		return TuningCandidate{}, 0, err
	}
	h := fnv.New64a()
	_, _ = h.Write([]byte(searchID))
	rng := rand.New(rand.NewSource(int64(space.revision) ^ int64(space.root) ^ int64(h.Sum64())))
	seen := make(map[string][]map[Id]int)
	var lastErr error
	attempts, generationPosition := 0, 0
	const maxAttempts = 128
	for attempts < maxAttempts && time.Now().Before(deadline) {
		if err := ctx.Err(); err != nil {
			return TuningCandidate{}, 0, err
		}
		choice := make(map[Id]int, len(space.classes))
		if !population.Parents() || population.Resample() {
			for _, class := range space.classes {
				choice[class] = rng.Intn(len(space.alternatives[class]))
			}
		} else {
			parent, ok := population.ParentFor(generationPosition)
			if !ok {
				return TuningCandidate{}, 0, fmt.Errorf("measured search has no parent")
			}
			for class, index := range parent.Value.choice {
				choice[class] = index
			}
			reachable, reachErr := space.reachable(choice)
			if reachErr != nil {
				lastErr = reachErr
			} else {
				genome := make([]int, len(reachable))
				genes := make([]egraphcore.DiscreteChoiceGene, 0, len(reachable))
				for i, class := range reachable {
					genome[i] = choice[class]
					if len(space.alternatives[class]) > 1 {
						alleles := make([]int, len(space.alternatives[class]))
						for j := range alleles {
							alleles[j] = j
						}
						genes = append(genes, egraphcore.DiscreteChoiceGene{Index: i, Alleles: alleles})
					}
				}
				mutations := population.MutationLimit(10)
				if mutations > 1 {
					mutations = rng.Intn(mutations) + 1
				}
				genome, err = egraphcore.MutateDiscreteChoice(genome, genes, mutations, rng)
				if err != nil {
					return TuningCandidate{}, 0, err
				}
				for i, class := range reachable {
					choice[class] = genome[i]
				}
			}
		}
		attempts++
		generationPosition++
		choice, err = space.repair(choice, rng)
		if err != nil {
			lastErr = err
			if generationPosition >= 8 {
				population.CloseGeneration(maxAttempts - attempts)
				generationPosition = 0
			}
			continue
		}
		fingerprint, fingerprintErr := space.fingerprint(choice)
		if fingerprintErr != nil {
			lastErr = fingerprintErr
			if generationPosition >= 8 {
				population.CloseGeneration(maxAttempts - attempts)
				generationPosition = 0
			}
			continue
		}
		duplicate := false
		for _, old := range seen[fingerprint] {
			if tuningChoicesEqual(space, choice, old) {
				duplicate = true
				break
			}
		}
		if duplicate {
			if generationPosition >= 8 {
				population.CloseGeneration(maxAttempts - attempts)
				generationPosition = 0
			}
			continue
		}
		expr, err := space.materialize(choice)
		if err != nil {
			lastErr = err
			if generationPosition >= 8 {
				population.CloseGeneration(maxAttempts - attempts)
				generationPosition = 0
			}
			continue
		}
		seen[fingerprint] = append(seen[fingerprint], choice)
		candidate, latency, benchErr := evaluateChoice(ctx, evaluate, expr, deadline)
		if benchErr != nil {
			if errors.Is(benchErr, context.Canceled) || errors.Is(benchErr, context.DeadlineExceeded) {
				return TuningCandidate{}, 0, benchErr
			}
			lastErr = benchErr
		} else {
			id := fingerprint
			_, err = population.Record(egraphcore.MeasuredCandidate[tuningChoice]{ID: id, Score: float64(latency), Value: tuningChoice{choice: choice, candidate: candidate}})
			if err != nil {
				return TuningCandidate{}, 0, err
			}
			t.pgo.Record(candidate.Key(), latency)
		}
		if generationPosition >= 8 {
			population.CloseGeneration(maxAttempts - attempts)
			generationPosition = 0
		}
	}
	if generationPosition != 0 {
		population.CloseGeneration(maxAttempts - attempts)
	}
	bestMeasured, measured := population.Best()
	if !measured {
		if err := ctx.Err(); err != nil {
			return TuningCandidate{}, 0, err
		}
		if !time.Now().Before(deadline) {
			return TuningCandidate{}, 0, fmt.Errorf("autotune budget expired before a candidate completed benchmarking")
		}
		if lastErr == nil {
			lastErr = fmt.Errorf("autotune budget expired before a candidate completed benchmarking")
		}
		return TuningCandidate{}, 0, lastErr
	}
	if ranked != nil {
		for _, candidate := range population.Ranking() {
			*ranked = append(*ranked, RankedTuningResult{
				Candidate: cloneTuningCandidate(candidate.Value.candidate),
				Latency:   time.Duration(candidate.Score),
			})
		}
	}
	return cloneTuningCandidate(bestMeasured.Value.candidate), time.Duration(bestMeasured.Score), nil
}

func evaluateChoice(ctx context.Context, evaluate expressionEvaluator, expr *Expr, deadline time.Time) (candidate TuningCandidate, latency time.Duration, err error) {
	defer func() {
		if value := recover(); value != nil {
			candidate = TuningCandidate{}
			latency = 0
			err = fmt.Errorf("candidate evaluator panicked: %v", value)
		}
	}()
	return evaluate(ctx, expr, deadline)
}

func (t *Tuner) measureExpression(ctx context.Context, expr *Expr, deadline time.Time) (TuningCandidate, time.Duration, error) {
	var best TuningCandidate
	var bestDuration time.Duration
	benchmarked := false
	var lastErr error
	for _, blockSize := range autotuneBlockSizes {
		if err := ctx.Err(); err != nil {
			return TuningCandidate{}, 0, err
		}
		if !time.Now().Before(deadline) {
			break
		}
		cand := TuningCandidate{Expr: expr, BlockDim: blockSize, KernelName: fmt.Sprintf("Kernel_Auto_%s_Block%d", sanitizeName(expr.Op), blockSize)}
		plan, err := benchmarkPlan(cand)
		if err != nil {
			lastErr = err
			continue
		}
		cand.GridDim, cand.SharedMem = int(plan.grid(uint32(blockSize))), int(plan.shared)
		cand.CUDASource, err = GenerateCUDA(expr, CodegenOptions{KernelName: cand.KernelName, Precision: "float"})
		if err != nil {
			lastErr = err
			continue
		}
		candidateCtx, cancel := context.WithDeadline(ctx, deadline)
		latency, err := benchmarkCandidateContext(candidateCtx, t.dev, cand, 5)
		cancel()
		if err != nil {
			lastErr = err
			if ctx.Err() != nil {
				return TuningCandidate{}, 0, ctx.Err()
			}
			continue
		}
		if !benchmarked || latency < bestDuration {
			best, bestDuration, benchmarked = cand, latency, true
		}
	}
	if !benchmarked {
		if lastErr == nil {
			lastErr = fmt.Errorf("no supported candidate configuration completed")
		}
		return TuningCandidate{}, 0, lastErr
	}
	return best, bestDuration, nil
}
