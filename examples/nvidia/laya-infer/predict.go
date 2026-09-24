package main

// Request handling: parse a request, run every question in one batch, and
// render the reply byte-for-byte like laya.c does.

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

// Predict answers every question in request against its state.
func (m *model) Predict(request []byte) ([]byte, error) {
	c := m.cfg
	debug := os.Getenv("LAYA_DEBUG") != ""

	r, err := parseRequest(request)
	if err != nil {
		return nil, err
	}
	if debug {
		fmt.Fprintf(os.Stderr, "[state] %s\n", r.state)
	}

	b, ntok, err := m.batch(r)
	if err != nil {
		return nil, err
	}
	rows, err := m.forward(b)
	if err != nil {
		return nil, err
	}

	var out strings.Builder
	out.WriteString(`{"model": "laya-rl-agent", "answers": {`)
	for i, q := range r.questions {
		k := len(q.opts)
		logits, actProb := m.score(rows[i], k)
		if debug {
			parts := make([]string, k)
			for j, v := range logits {
				parts[j] = fmt.Sprintf("%.6f", v)
			}
			fmt.Fprintf(os.Stderr, "[%s] logits: %s\n[%s] act_prob(raw): %.6f\n",
				r.ids[i], strings.Join(parts, " "), r.ids[i], actProb)
		}

		p, conf := c.calibrate(q.typ, logits)

		if i > 0 {
			out.WriteString(", ")
		}
		writeJSONString(&out, r.ids[i], false)
		fmt.Fprintf(&out, `: {"type": "%s", `, qNames[q.typ])
		switch q.typ {
		case 0:
			am := 0
			for j := range k {
				if p[j] > p[am] {
					am = j
				}
			}
			out.WriteString(`"choice": `)
			writeJSONString(&out, q.labels[am], false)
			out.WriteString(`, "probabilities": {`)
			for j := range k {
				if j > 0 {
					out.WriteString(", ")
				}
				writeJSONString(&out, q.labels[j], false)
				out.WriteString(": ")
				out.WriteString(putNum(p[j]))
			}
			out.WriteString(`}, "confidence": `)
			out.WriteString(putNum(conf))
		case 1:
			out.WriteString(`"score": `)
			out.WriteString(putNum(expectedScore(p)))
			out.WriteString(`, "legend": {`)
			for j := range k {
				if j > 0 {
					out.WriteString(", ")
				}
				fmt.Fprintf(&out, `"%d": `, j)
				q.crit.arr[j].dumps(&out, false)
			}
			out.WriteString(`}, "probabilities": {`)
			for j := range k {
				if j > 0 {
					out.WriteString(", ")
				}
				fmt.Fprintf(&out, `"%d": `, j)
				out.WriteString(putNum(p[j]))
			}
			out.WriteString(`}, "confidence": `)
			out.WriteString(putNum(conf))
		default:
			out.WriteString(`"noul": `)
			out.WriteString(putNum(p[1]))
			out.WriteString(`, "confidence": `)
			out.WriteString(putNum(conf))
		}
		out.WriteString(`, "action": {"act_probability": `)
		out.WriteString(putNum(float64(actProb)))
		out.WriteString("}}")
	}
	fmt.Fprintf(&out, `}, "usage": {"input_tokens": %d, "output_tokens": 0}}`, ntok)
	return []byte(out.String()), nil
}

// batch tokenizes every question in r against its state and pads them into
// one batch. It also reports the number of real tokens, which the reply's
// usage counts.
func (m *model) batch(r *request) (*batch, int, error) {
	c := m.cfg
	debug := os.Getenv("LAYA_DEBUG") != ""
	n := len(r.questions)
	seqs := make([][]int32, n)
	markers := make([][]int32, n)
	types := make([]int32, n)
	ntok := 0
	for i, q := range r.questions {
		qid := r.ids[i]
		ids, mk, err := buildSequence(c, m.tok, r.state, q)
		if err != nil {
			return nil, 0, err
		}
		if len(mk) != len(q.opts) {
			return nil, 0, fmt.Errorf("question %q options exceed head_max_len=%d", qid, c.HeadMaxLen)
		}
		if debug {
			fmt.Fprintf(os.Stderr, "[%s] T=%d markers: %s\n[%s] ids: %s\n",
				qid, len(ids), joinInts(mk), qid, joinInts(ids))
		}
		seqs[i], markers[i], types[i] = ids, mk, int32(q.typ)
		ntok += len(ids)
	}
	return newBatch(seqs, types, markers), ntok, nil
}

// putNum prints round(x, 4) the way Python's repr does: at least one decimal
// place, no trailing zeros beyond it.
func putNum(x float64) string {
	s := strconv.FormatFloat(x, 'f', 4, 64)
	for len(s) > 2 && s[len(s)-1] == '0' && s[len(s)-2] != '.' {
		s = s[:len(s)-1]
	}
	if s == "-0.0" {
		return "0.0"
	}
	return s
}

func joinInts(v []int32) string {
	var b strings.Builder
	for _, x := range v {
		fmt.Fprintf(&b, " %d", x)
	}
	return b.String()
}

// calibrate turns one question's logits into the probabilities and the
// confidence its answer reports: a temperature per (type, option count), then
// a softmax, then one minus the normalized entropy, except that a noul
// question's confidence is the probability of the side it comes down on.
func (c *config) calibrate(typ int, logits []float32) (p []float64, conf float64) {
	k := len(logits)
	ts := max(c.tempFor(typ, k), 1e-3)
	scaled := make([]float32, k)
	for j, v := range logits {
		scaled[j] = float32(float64(v) / ts)
	}
	p = softmax64(scaled)
	if typ != 0 && typ != 1 {
		return p, max(p[1], 1.0-p[1])
	}
	conf = 1.0
	if k >= 2 {
		var ent float64
		for _, v := range p {
			ent -= v * math.Log(max(v, 1e-12))
		}
		conf = min(max(1.0-ent/math.Log(float64(k)), 0), 1)
	}
	return p, conf
}

// expectedScore is a score question's answer: the expected option index.
func expectedScore(p []float64) float64 {
	var es float64
	for j, v := range p {
		es += float64(j) * v
	}
	return es
}
