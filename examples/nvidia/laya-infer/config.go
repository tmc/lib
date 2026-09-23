package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// agentConfig mirrors the fields laya.c reads from rl_agent_config.json.
type agentConfig struct {
	MaxLen      int                `json:"max_len"`
	HeadMaxLen  int                `json:"head_max_len"`
	HeadLayers  int                `json:"head_layers"`
	Temperature []float64          `json:"temperature"`
	TempByOpts  map[string]float64 `json:"temperature_by_options"`
}

// encoderConfig mirrors the ModernBERT hyper-parameters from encoder config.json.
type encoderConfig struct {
	HiddenSize       int      `json:"hidden_size"`
	NumHiddenLayers  int      `json:"num_hidden_layers"`
	NumAttnHeads     int      `json:"num_attention_heads"`
	IntermediateSize int      `json:"intermediate_size"`
	VocabSize        int      `json:"vocab_size"`
	LocalAttention   int      `json:"local_attention"`
	GlobalEveryN     int      `json:"global_attn_every_n_layers"`
	NormEps          *float64 `json:"norm_eps"`
	LayerNormEps     *float64 `json:"layer_norm_eps"`
	HiddenActivation string   `json:"hidden_activation"`
	LayerTypes       []string `json:"layer_types"`
	RopeParameters   struct {
		FullAttention struct {
			RopeTheta float64 `json:"rope_theta"`
		} `json:"full_attention"`
		SlidingAttention struct {
			RopeTheta float64 `json:"rope_theta"`
		} `json:"sliding_attention"`
	} `json:"rope_parameters"`
	GlobalRopeTheta *float64 `json:"global_rope_theta"`
	LocalRopeTheta  *float64 `json:"local_rope_theta"`
}

// config is the resolved geometry the rest of the program uses.
type config struct {
	MaxLen, HeadMaxLen int
	HeadLayers         int
	Temperature        [3]float64
	TempByOpts         map[string]float64

	D, NumLayers, NumHeads, DFF, Vocab int
	Window                             int
	Eps                                float32
	ThetaGlobal, ThetaLocal            float64
	Local                              []bool

	HeadHeads, HeadFF int
	ActHidden, NumAct int
}

// HeadDim returns the per-head width of the encoder attention.
func (c *config) HeadDim() int { return c.D / c.NumHeads }

// loadConfig reads rl_agent_config.json and config.json from dir, applying the
// same defaults laya.c uses when a field is absent.
func loadConfig(dir string) (*config, error) {
	var ac agentConfig
	if err := readJSON(filepath.Join(dir, "rl_agent_config.json"), &ac); err != nil {
		return nil, err
	}
	c := &config{
		MaxLen:     orDefault(ac.MaxLen, 512),
		HeadMaxLen: orDefault(ac.HeadMaxLen, 192),
		HeadLayers: orDefault(ac.HeadLayers, 2),
		TempByOpts: ac.TempByOpts,
	}
	for i := range c.Temperature {
		c.Temperature[i] = 1.0
		if i < len(ac.Temperature) {
			c.Temperature[i] = ac.Temperature[i]
		}
	}

	var ec encoderConfig
	path := filepath.Join(dir, "config.json")
	if err := readJSON(path, &ec); err != nil {
		if !os.IsNotExist(err) {
			return nil, err
		}
		fmt.Fprintf(os.Stderr, "laya: %s missing, assuming ModernBERT-large hyper-parameters\n", path)
	}
	c.D = orDefault(ec.HiddenSize, 1024)
	c.NumLayers = orDefault(ec.NumHiddenLayers, 28)
	c.NumHeads = orDefault(ec.NumAttnHeads, 16)
	c.DFF = orDefault(ec.IntermediateSize, 2624)
	c.Vocab = orDefault(ec.VocabSize, 50368)
	c.Window = orDefault(ec.LocalAttention, 128) / 2
	globalEvery := orDefault(ec.GlobalEveryN, 3)

	c.Eps = 1e-5
	if ec.LayerNormEps != nil {
		c.Eps = float32(*ec.LayerNormEps)
	}
	if ec.NormEps != nil {
		c.Eps = float32(*ec.NormEps)
	}

	c.ThetaGlobal, c.ThetaLocal = 160000.0, 10000.0
	if ec.GlobalRopeTheta != nil {
		c.ThetaGlobal = *ec.GlobalRopeTheta
	}
	if ec.LocalRopeTheta != nil {
		c.ThetaLocal = *ec.LocalRopeTheta
	}
	if v := ec.RopeParameters.FullAttention.RopeTheta; v != 0 {
		c.ThetaGlobal = v
	}
	if v := ec.RopeParameters.SlidingAttention.RopeTheta; v != 0 {
		c.ThetaLocal = v
	}

	if ec.HiddenActivation != "" && ec.HiddenActivation != "gelu" {
		return nil, fmt.Errorf("hidden_activation %q is not supported (only gelu)", ec.HiddenActivation)
	}
	if c.D%c.NumHeads != 0 || (c.D/c.NumHeads)%2 != 0 {
		return nil, fmt.Errorf("bad head geometry in encoder config")
	}
	if c.D/c.NumHeads > 256 {
		return nil, fmt.Errorf("head_dim > 256 is not supported")
	}

	c.Local = make([]bool, c.NumLayers)
	for i := range c.Local {
		if i < len(ec.LayerTypes) {
			c.Local[i] = ec.LayerTypes[i] == "sliding_attention"
		} else {
			c.Local[i] = i%globalEvery != 0
		}
	}

	c.HeadHeads = max(c.D/64, 1)
	return c, nil
}

// tempFor selects the calibration temperature for a question type and option count.
func (c *config) tempFor(qtype, k int) float64 {
	var size string
	switch {
	case k <= 2:
		size = "2"
	case k <= 5:
		size = "3-5"
	case k <= 10:
		size = "6-10"
	default:
		size = "11+"
	}
	if t, ok := c.TempByOpts[qNames[qtype]+":"+size]; ok {
		return t
	}
	return c.Temperature[qtype]
}

func readJSON(path string, dst any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(b, dst); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}

func orDefault(v, dflt int) int {
	if v == 0 {
		return dflt
	}
	return v
}
