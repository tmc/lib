package main

import (
	"io"

	"github.com/tmc/lib/examples/ds4-min/internal/dsparkstate"
)

type dsparkSupportEnvelope struct {
	model          [32]byte
	context, block int
	targets        []int
	journal        *dsparkFeatureJournal
	history        []float32
	capture        *dsparkSupportCapture
}

type dsparkSupportCapture struct {
	generation      uint64
	position, token int
	published       bool
	captured        []bool
	status          uint32
	features        []float32
}

func (v *dsparkSupportEnvelope) wire() *dsparkstate.Envelope {
	if v == nil {
		return nil
	}
	w := &dsparkstate.Envelope{Model: v.model, Context: v.context, Block: v.block, Targets: v.targets, History: v.history, Platform: 2}
	if j := v.journal; j != nil {
		w.Journal = &dsparkstate.Journal{Frontier: j.frontier, Epoch: j.epoch, Prefix: j.prefix, BasePrefix: j.basePrefix, Rows: make([]dsparkstate.Row, len(j.rows))}
		for i, row := range j.rows {
			w.Journal.Rows[i] = dsparkstate.Row{Position: row.position, Token: row.token, Prefix: row.prefix, Epoch: row.epoch}
		}
	}
	if c := v.capture; c != nil {
		w.Platform = 1
		w.Capture = &dsparkstate.Capture{Generation: c.generation, Position: c.position, Token: c.token, Published: c.published, Captured: c.captured, Status: c.status, Features: c.features}
	}
	return w
}

func supportFromWire(w *dsparkstate.Envelope) *dsparkSupportEnvelope {
	v := &dsparkSupportEnvelope{model: w.Model, context: w.Context, block: w.Block, targets: w.Targets, history: w.History}
	j := w.Journal
	v.journal = &dsparkFeatureJournal{frontier: j.Frontier, epoch: j.Epoch, prefix: j.Prefix, basePrefix: j.BasePrefix, rows: make([]dsparkFeatureRow, len(j.Rows))}
	for i, row := range j.Rows {
		v.journal.rows[i] = dsparkFeatureRow{position: row.Position, token: row.Token, prefix: row.Prefix, epoch: row.Epoch}
	}
	if c := w.Capture; c != nil {
		v.capture = &dsparkSupportCapture{generation: c.Generation, position: c.Position, token: c.Token, published: c.Published, captured: c.Captured, status: c.Status, features: c.Features}
	}
	return v
}

func (v *dsparkSupportEnvelope) validate(frontier int, generation uint64) error {
	return v.wire().Validate(frontier, generation)
}

func writeDSparkSupport(dst io.Writer, v *dsparkSupportEnvelope, frontier int, generation uint64) error {
	return dsparkstate.Write(dst, v.wire(), frontier, generation)
}

func readDSparkSupport(data []byte, expected *dsparkSupportEnvelope, frontier int, generation uint64) (*dsparkSupportEnvelope, error) {
	w, err := dsparkstate.Read(data, expected.wire(), frontier, generation)
	if err != nil {
		return nil, err
	}
	return supportFromWire(w), nil
}
