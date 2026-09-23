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

	root, err := parseJSON(request)
	if err != nil {
		return nil, err
	}
	state, qs := root.get("state"), root.get("questions")
	if state == nil || qs == nil || qs.kind != jObject {
		return nil, fmt.Errorf("request must be an object with 'state' and 'questions'")
	}
	st := state.str
	if state.kind != jString {
		st = state.dumpString(false) // serialize_state: json.dumps(state, ensure_ascii=False)
	}
	if debug {
		fmt.Fprintf(os.Stderr, "[state] %s\n", st)
	}

	n := qs.length()
	questions := make([]*question, n)
	seqs := make([][]int32, n)
	markers := make([][]int32, n)
	types := make([]int32, n)
	ntok := 0
	for i := range n {
		qid := qs.keys[i]
		q, err := parseQuestion(qid, qs.vals[i])
		if err != nil {
			return nil, err
		}
		ids, mk, err := buildSequence(c, m.tok, st, q)
		if err != nil {
			return nil, err
		}
		if len(mk) != len(q.opts) {
			return nil, fmt.Errorf("question %q options exceed head_max_len=%d", qid, c.HeadMaxLen)
		}
		if debug {
			fmt.Fprintf(os.Stderr, "[%s] T=%d markers: %s\n[%s] ids: %s\n",
				qid, len(ids), joinInts(mk), qid, joinInts(ids))
		}
		questions[i], seqs[i], markers[i], types[i] = q, ids, mk, int32(q.typ)
		ntok += len(ids)
	}

	rows, err := m.forward(newBatch(seqs, types, markers))
	if err != nil {
		return nil, err
	}

	var out strings.Builder
	out.WriteString(`{"model": "laya-rl-agent", "answers": {`)
	for i, q := range questions {
		k := len(q.opts)
		logits, actProb := m.score(rows[i], k)
		if debug {
			parts := make([]string, k)
			for j, v := range logits {
				parts[j] = fmt.Sprintf("%.6f", v)
			}
			fmt.Fprintf(os.Stderr, "[%s] logits: %s\n[%s] act_prob(raw): %.6f\n",
				qs.keys[i], strings.Join(parts, " "), qs.keys[i], actProb)
		}

		// Calibration: temperature per (type, option count), then softmax and
		// an entropy-based confidence.
		ts := max(c.tempFor(q.typ, k), 1e-3)
		scaled := make([]float32, k)
		for j, v := range logits {
			scaled[j] = float32(float64(v) / ts)
		}
		p := softmax64(scaled)
		conf := 1.0
		if k >= 2 {
			var ent float64
			for _, v := range p {
				ent -= v * math.Log(max(v, 1e-12))
			}
			conf = min(max(1.0-ent/math.Log(float64(k)), 0), 1)
		}

		if i > 0 {
			out.WriteString(", ")
		}
		writeJSONString(&out, qs.keys[i], false)
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
			writeJSONString(&out, q.label(am), false)
			out.WriteString(`, "probabilities": {`)
			for j := range k {
				if j > 0 {
					out.WriteString(", ")
				}
				writeJSONString(&out, q.label(j), false)
				out.WriteString(": ")
				out.WriteString(putNum(p[j]))
			}
			out.WriteString(`}, "confidence": `)
			out.WriteString(putNum(conf))
		case 1:
			var es float64
			for j, v := range p {
				es += float64(j) * v
			}
			out.WriteString(`"score": `)
			out.WriteString(putNum(es))
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
			out.WriteString(putNum(max(p[1], 1.0-p[1])))
		}
		out.WriteString(`, "action": {"act_probability": `)
		out.WriteString(putNum(float64(actProb)))
		out.WriteString("}}")
	}
	fmt.Fprintf(&out, `}, "usage": {"input_tokens": %d, "output_tokens": 0}}`, ntok)
	return []byte(out.String()), nil
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
