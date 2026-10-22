package main

import (
	"fmt"
	"reflect"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/gguf"
)

// proConfig returns the Pro shape without sharing mutable schedule storage.
func proConfig() Config {
	c := Flash
	c.Variant, c.Layers, c.Embed, c.Heads = "pro", 61, 7168, 128
	c.QLoRA, c.OutGroups, c.Experts = 1536, 16, 384
	c.ExpertFF, c.IndexerTopK, c.ExpertWeightScale = 3072, 1024, 2.5
	c.CompressRatio = make([]int, c.Layers)
	c.SwiGLUClamp = make([]float32, c.Layers)
	s, _ := dsv4shape.For(dsv4shape.Pro)
	for i := range c.CompressRatio {
		l, _ := s.Layer(i)
		c.CompressRatio[i], c.SwiGLUClamp[i] = l.CompressRatio, 10
	}
	return c
}

// ConfigFor recognizes Flash or Pro metadata. Current backends still admit
// only Flash through ReadConfig, Validate and Supports.
func ConfigFor(g *gguf.File) (*Config, error) {
	if g == nil {
		return nil, fmt.Errorf("read config: missing GGUF")
	}
	md := make(map[string]uint64)
	for k, v := range g.Metadata {
		if n, ok := v.Int(); ok && n >= 0 {
			md[k] = uint64(n)
		}
	}
	s, err := dsv4shape.Select(md)
	if err != nil {
		return nil, err
	}
	c := Flash
	if s.Variant == dsv4shape.Pro {
		c = proConfig()
	}
	return readConfig(g, c)
}

// ValidateShape validates the metadata shape, not backend kernel support.
func (c *Config) ValidateShape() error {
	if c == nil {
		return fmt.Errorf("config: missing configuration")
	}
	if c.Variant == "flash" {
		return c.Validate()
	}
	if c.Variant != "pro" {
		return fmt.Errorf("config: unknown variant %q", c.Variant)
	}
	w := proConfig()
	if !reflect.DeepEqual(*c, w) {
		return fmt.Errorf("config: invalid Pro shape")
	}
	return nil
}
