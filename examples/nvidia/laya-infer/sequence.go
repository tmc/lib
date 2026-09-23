package main

// Question parsing and input assembly, ported from laya.c's q_parse and
// build_sequence (the reference Agent._to_internal, render_options and
// common.build_sequence).

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/tmc/lib/hf/hftokenizer"
)

// qNames are the question types, indexed by question.typ.
var qNames = [3]string{"choice", "score", "noul"}

// tokenizer wraps the Hugging Face tokenizer with the special-token ids
// build_sequence needs.
type tokenizer struct {
	tk             *hftokenizer.Tokenizer
	cls, sep, mask int32
	maskStr        string
}

func loadTokenizer(dir string) (*tokenizer, error) {
	path := filepath.Join(dir, "tokenizer.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	tk, err := hftokenizer.New(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	var doc struct {
		AddedTokens []struct {
			ID      int32  `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	t := &tokenizer{tk: tk, cls: -1, sep: -1, mask: -1, maskStr: "[MASK]"}
	for _, a := range doc.AddedTokens {
		switch a.Content {
		case "[CLS]":
			t.cls = a.ID
		case "[SEP]":
			t.sep = a.ID
		case "[MASK]":
			t.mask = a.ID
			t.maskStr = a.Content
		}
	}
	if t.cls < 0 || t.sep < 0 || t.mask < 0 {
		return nil, fmt.Errorf("%s: [CLS]/[SEP]/[MASK] not found in added_tokens", path)
	}
	return t, nil
}

// encode tokenizes text without the post-processor's [CLS]/[SEP] wrapper and
// keeps at most limit ids; a negative limit keeps all of them.
func (t *tokenizer) encode(text string, limit int) ([]int32, error) {
	ids, err := t.tk.EncodeRaw(text)
	if err != nil {
		return nil, err
	}
	if limit >= 0 && len(ids) > limit {
		ids = ids[:limit]
	}
	return ids, nil
}

// question is one parsed request entry: the rendered instructions and the
// option strings the scorer ranks.
type question struct {
	typ  int // index into qNames
	ins  string
	opts []string
	crit *jsonValue
}

// label returns the name reported for option i: the criteria key for a choice
// object, the criteria string for a choice list.
func (q *question) label(i int) string {
	if q.crit.kind == jObject {
		return q.crit.keys[i]
	}
	return q.crit.arr[i].str
}

func parseQuestion(qid string, qd *jsonValue) (*question, error) {
	fail := func(format string, a ...any) (*question, error) {
		return nil, fmt.Errorf("question %q: "+format, append([]any{qid}, a...)...)
	}
	if qd.kind != jObject {
		return nil, fmt.Errorf("question %q must be an object", qid)
	}
	ty, ins, crit := qd.get("type"), qd.get("instructions"), qd.get("criteria")
	if ty == nil || ty.kind != jString {
		return fail("missing 'type'")
	}
	q := &question{typ: -1, crit: crit}
	for i, n := range qNames {
		if ty.str == n {
			q.typ = i
		}
	}
	if q.typ < 0 {
		return fail("unknown type %q (choice|score|noul)", ty.str)
	}
	if ins == nil {
		return fail("missing 'instructions'")
	}
	if ins.kind == jString {
		q.ins = ins.str
	} else {
		q.ins = ins.dumpString(true) // json.dumps(ins)
	}

	switch q.typ {
	case 0:
		if crit == nil || (crit.kind != jObject && crit.kind != jArray) || crit.length() == 0 {
			return fail("choice needs non-empty 'criteria'")
		}
		for i := range crit.length() {
			if crit.kind == jArray {
				if crit.arr[i].kind != jString {
					return fail("list criteria must be strings")
				}
				q.opts = append(q.opts, crit.arr[i].str)
				continue
			}
			v := crit.vals[i]
			if absent(v) {
				q.opts = append(q.opts, crit.keys[i])
			} else {
				q.opts = append(q.opts, crit.keys[i]+": "+renderCriterion(v))
			}
		}
	case 1:
		if crit == nil || crit.kind != jArray || len(crit.arr) == 0 {
			return fail("score needs a non-empty 'criteria' list")
		}
		for i, v := range crit.arr {
			q.opts = append(q.opts, fmt.Sprintf("level %d: %s", i, renderCriterion(v)))
		}
	default:
		if crit != nil && crit.kind != jObject && !absent(crit) {
			return fail("noul 'criteria' must be an object")
		}
		fc, tc := crit.get("false"), crit.get("true")
		fd, td := "no, the statement does not hold", "yes, the statement holds"
		if !absent(fc) {
			fd = renderCriterion(fc)
		}
		if !absent(tc) {
			td = renderCriterion(tc)
		}
		q.opts = []string{"false: " + fd, "true: " + td}
	}
	return q, nil
}

// absent reports whether a criterion is missing, null or the empty string.
func absent(v *jsonValue) bool {
	return v == nil || v.kind == jNull || (v.kind == jString && v.str == "")
}

// renderCriterion yields a string criterion verbatim and any other value as
// json.dumps with ensure_ascii=False.
func renderCriterion(v *jsonValue) string {
	if v.kind == jString {
		return v.str
	}
	return v.dumpString(false)
}

// replaceMask blanks out any literal mask token in user text so it cannot be
// confused with the markers build_sequence inserts.
func replaceMask(s, mask string) string {
	return strings.ReplaceAll(s, mask, " ")
}

// buildSequence assembles the model input
//
//	[CLS] <type> question: ins [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
//
// and returns it with the position of each option's [MASK] marker. Options are
// capped at 48 ids each, and the instructions are trimmed to whatever the
// options leave inside head_max_len.
func buildSequence(c *config, t *tokenizer, state string, q *question) (ids, markers []int32, err error) {
	head, err := t.encode(qNames[q.typ]+" question: "+replaceMask(q.ins, t.maskStr), c.HeadMaxLen)
	if err != nil {
		return nil, nil, err
	}

	opts := make([][]int32, len(q.opts))
	total := 0
	for i, o := range q.opts {
		body, err := t.encode(" "+replaceMask(o, t.maskStr), 48)
		if err != nil {
			return nil, nil, err
		}
		opts[i] = append([]int32{t.mask}, body...)
		total += len(opts[i])
	}

	// When the options alone crowd out the instructions, give each option an
	// equal share instead and reserve at least 16 ids for the instructions.
	budget := c.HeadMaxLen - total
	if budget < 16 {
		per := max((c.HeadMaxLen-16)/max(len(q.opts), 1), 4)
		total = 0
		for i := range opts {
			if len(opts[i]) > per {
				opts[i] = opts[i][:per]
			}
			total += len(opts[i])
		}
		budget = c.HeadMaxLen - total
	}
	if take := max(budget, 8); len(head) > take {
		head = head[:take]
	}

	ids = append(ids, t.cls)
	ids = append(ids, head...)
	ids = append(ids, t.sep)
	for _, o := range opts {
		markers = append(markers, int32(len(ids)))
		ids = append(ids, o...)
	}
	ids = append(ids, t.sep)

	room := max(c.MaxLen-len(ids)-1, 0)
	body, err := t.encode(replaceMask(state, t.maskStr), room)
	if err != nil {
		return nil, nil, err
	}
	ids = append(ids, body...)
	ids = append(ids, t.sep)
	if len(ids) > c.MaxLen {
		ids = ids[:c.MaxLen]
	}
	keep := markers[:0]
	for _, m := range markers {
		if int(m) < c.MaxLen {
			keep = append(keep, m)
		}
	}
	return ids, keep, nil
}
