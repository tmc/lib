package cudafuse

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// Failed stacks retain closures so unresolved native owners remain reachable.
// A retry resumes at the failed owner, never at an already completed teardown.
type attentionResidualCleanupStack struct {
	steps []func() error
}

var attentionResidualUnresolved []*attentionResidualCleanupStack

func (s *attentionResidualCleanupStack) add(f func() error) {
	s.steps = append(s.steps, f)
}

// Adopt nonnil error results before inspecting their error or asserting state.
func (s *attentionResidualCleanupStack) adopt(present bool, close func() error) {
	if present {
		s.add(close)
	}
}

func (s *attentionResidualCleanupStack) close() error {
	for len(s.steps) != 0 {
		i := len(s.steps) - 1
		if err := s.steps[i](); err != nil {
			return fmt.Errorf("attention residual cleanup has %d unresolved owners: %w", len(s.steps), err)
		}
		s.steps = s.steps[:i]
	}
	return nil
}

// Independent direct FP64 depth distribution, never a production fallback.
func attentionResidualReference(c AttentionResidualConfig, v, g, w []float32, mutant int) ([]float64, []float64) {
	scores := make([]float64, c.Candidates*c.Rows)
	norm := make([]float64, len(scores))
	for candidate := 0; candidate < c.Candidates; candidate++ {
		for row := 0; row < c.Rows; row++ {
			i := candidate*c.Rows + row
			var squares float64
			for d := 0; d < c.Width; d++ {
				x := float64(v[i*c.Width+d])
				squares += x * x
			}
			norm[i] = math.Sqrt(squares/float64(c.Width) + float64(c.Epsilon))
			for d := 0; d < c.Width; d++ {
				mult := float64(g[d])
				if mutant == 3 {
					mult--
				}
				scores[i] += float64(w[d]) * mult * float64(v[i*c.Width+d]) / norm[i]
			}
			if mutant == 4 {
				scores[i] /= math.Sqrt(float64(c.Width))
			}
		}
	}
	p := make([]float64, len(scores))
	count := c.Candidates
	if mutant == 5 && count > 1 {
		count--
	}
	for row := 0; row < c.Rows; row++ {
		max := scores[row]
		for k := 1; k < count; k++ {
			max = math.Max(max, scores[k*c.Rows+row])
		}
		var sum float64
		for k := 0; k < count; k++ {
			p[k*c.Rows+row] = math.Exp(scores[k*c.Rows+row] - max)
			sum += p[k*c.Rows+row]
		}
		for k := 0; k < count; k++ {
			p[k*c.Rows+row] /= sum
		}
	}
	if mutant == 2 {
		for k := 0; k < c.Candidates; k++ {
			max := scores[k*c.Rows]
			for r := 1; r < c.Rows; r++ {
				max = math.Max(max, scores[k*c.Rows+r])
			}
			var sum float64
			for r := 0; r < c.Rows; r++ {
				p[k*c.Rows+r] = math.Exp(scores[k*c.Rows+r] - max)
				sum += p[k*c.Rows+r]
			}
			for r := 0; r < c.Rows; r++ {
				p[k*c.Rows+r] /= sum
			}
		}
	}
	out := make([]float64, c.Rows*c.Width)
	for row := 0; row < c.Rows; row++ {
		for d := 0; d < c.Width; d++ {
			for k := 0; k < c.Candidates; k++ {
				i := k*c.Rows + row
				value := float64(v[i*c.Width+d])
				if mutant == 1 {
					value /= norm[i]
				}
				out[row*c.Width+d] += p[i] * value
			}
		}
	}
	return out, p
}

func attentionResidualFixture(c AttentionResidualConfig) (v, g, w, offset []float32) {
	v = make([]float32, c.Candidates*c.Rows*c.Width)
	g = make([]float32, c.Width)
	w = make([]float32, c.Width)
	offset = make([]float32, c.Width)
	for d := range g {
		offset[d] = float32((d*7)%11-5) * 0.037
		g[d] = 1 + offset[d]
		w[d] = float32((d*5)%9-4)*0.071 + 0.023
	}
	for k := 0; k < c.Candidates; k++ {
		for r := 0; r < c.Rows; r++ {
			for d := 0; d < c.Width; d++ {
				v[(k*c.Rows+r)*c.Width+d] = float32((d*3+k*11+r*17)%23-11)*float32(k+1)*0.13 + float32((k+r)%3)*0.19
			}
		}
	}
	return
}

func attentionResidualCompare(got []float32, want []float64) error {
	if len(got) != len(want) {
		return fmt.Errorf("output extent differs")
	}
	for i, x := range got {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || math.Abs(float64(x)-want[i]) > 2e-5+2e-4*math.Abs(want[i]) {
			return fmt.Errorf("output[%d]=%g want %g", i, x, want[i])
		}
	}
	return nil
}

// Test-local FP32 arithmetic model, including the existing collective's
// explicit 32-lane order. It is host math evidence, not simulated GPU evidence.
func attentionResidualFP32(c AttentionResidualConfig, v, g, w []float32) []float32 {
	if c.Candidates == 1 {
		return append([]float32(nil), v...)
	}
	reduce := func(x [32]float32) float32 {
		for offset := 16; offset > 0; offset /= 2 {
			old := x
			for lane := 0; lane+offset < 32; lane++ {
				x[lane] = old[lane] + old[lane+offset]
			}
		}
		return x[0]
	}
	s := make([]float32, c.Candidates*c.Rows)
	for i := range s {
		var lanes [32]float32
		for d := 0; d < c.Width; d++ {
			x := v[i*c.Width+d]
			lanes[d%32] += x * x
		}
		rms := float32(math.Sqrt(float64(reduce(lanes)/float32(c.Width) + c.Epsilon)))
		lanes = [32]float32{}
		for d := 0; d < c.Width; d++ {
			lanes[d%32] += w[d] * g[d] * (v[i*c.Width+d] / rms)
		}
		s[i] = reduce(lanes)
	}
	p := make([]float32, len(s))
	for row := 0; row < c.Rows; row++ {
		max := s[row]
		for k := 1; k < c.Candidates; k++ {
			if s[k*c.Rows+row] > max {
				max = s[k*c.Rows+row]
			}
		}
		var sum float32
		for k := 0; k < c.Candidates; k++ {
			p[k*c.Rows+row] = float32(math.Exp(float64(s[k*c.Rows+row] - max)))
			sum += p[k*c.Rows+row]
		}
		for k := 0; k < c.Candidates; k++ {
			p[k*c.Rows+row] /= sum
		}
	}
	out := make([]float32, c.Rows*c.Width)
	for row := 0; row < c.Rows; row++ {
		for d := 0; d < c.Width; d++ {
			for k := 0; k < c.Candidates; k++ {
				out[row*c.Width+d] += p[k*c.Rows+row] * v[(k*c.Rows+row)*c.Width+d]
			}
		}
	}
	return out
}

// The dispatch packet owns an adjacent source tree. go test runs from the
// package directory, so qualification must not resolve pins against its cwd.
func attentionResidualSourceBytes(manifestPath, name string) ([]byte, error) {
	if !filepath.IsAbs(manifestPath) || !strings.HasPrefix(name, "nvidia/x/cudafuse/attention_residual") || filepath.ToSlash(filepath.Clean(name)) != name || strings.Contains(name, "\\") || strings.Contains(name, "..") || !strings.HasSuffix(name, ".go") {
		return nil, fmt.Errorf("invalid qualification source path")
	}
	root := filepath.Join(filepath.Dir(manifestPath), "source")
	path := root
	parts := append([]string{""}, strings.Split(name, "/")...)
	for i, part := range parts {
		if part != "" {
			path = filepath.Join(path, part)
		}
		info, err := os.Lstat(path)
		if err != nil {
			return nil, err
		}
		if i == len(parts)-1 {
			if !info.Mode().IsRegular() {
				return nil, fmt.Errorf("qualification source is not a regular file")
			}
		} else if !info.IsDir() {
			return nil, fmt.Errorf("qualification source parent is not a directory")
		}
	}
	return os.ReadFile(path)
}
