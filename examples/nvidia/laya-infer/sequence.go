package main

// Question parsing and input assembly, ported from laya.c's q_parse and
// build_sequence (the reference Agent._to_internal, render_options and
// common.build_sequence).

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
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
	path := modelFile(dir, "tokenizer", "tokenizer.json")
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

// question is one parsed request entry: the rendered instructions, the
// option strings the scorer ranks, and the label each option is reported
// under.
type question struct {
	typ    int // index into qNames
	ins    string
	opts   []string
	labels []string // choice only
	crit   *jsonValue
}

// A request is a parsed request body: the serialized state and the questions
// in the order the request gave them, which is the order the reply keeps.
type request struct {
	state     string
	ids       []string
	questions []*question
}

// parseRequest parses a request body. Questions may be an object keyed by id,
// as laya.c takes them, or an array of objects each carrying its own "id", as
// mlx-go-sys1 also accepts; a JSON array is the only way some clients can
// promise an order. Either way an id may appear only once, since the reply
// is an object keyed by id and a repeat would silently shadow an answer.
func parseRequest(data []byte) (*request, error) {
	root, err := parseJSON(data)
	if err != nil {
		return nil, err
	}
	state, qs := root.get("state"), root.get("questions")
	if state == nil || qs == nil || (qs.kind != jObject && qs.kind != jArray) {
		return nil, fmt.Errorf("request must be an object with 'state' and 'questions'")
	}
	r := &request{state: state.str}
	if state.kind != jString {
		r.state = state.dumpString(false) // serialize_state: json.dumps(state, ensure_ascii=False)
	}
	seen := make(map[string]bool)
	for i := range qs.length() {
		qd := qs.at(i)
		var qid string
		if qs.kind == jObject {
			qid = qs.keys[i]
		} else {
			id := qd.get("id")
			if id == nil || id.kind != jString || id.str == "" {
				return nil, fmt.Errorf("question %d: missing 'id'", i)
			}
			qid = id.str
		}
		if seen[qid] {
			return nil, fmt.Errorf("duplicate question id %q", qid)
		}
		seen[qid] = true
		q, err := parseQuestion(qid, qd)
		if err != nil {
			return nil, err
		}
		r.ids = append(r.ids, qid)
		r.questions = append(r.questions, q)
	}
	return r, nil
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
		seen := make(map[string]bool)
		for i := range crit.length() {
			label, desc, err := choiceOption(crit, i)
			if err != nil {
				return fail("%v", err)
			}
			if label == "" {
				return fail("choice label cannot be empty")
			}
			if seen[label] {
				return fail("choice label %q appears twice", label)
			}
			seen[label] = true
			q.labels = append(q.labels, label)
			if absent(desc) {
				q.opts = append(q.opts, label)
			} else {
				q.opts = append(q.opts, label+": "+renderCriterion(desc))
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

// choiceOption returns the label and description of choice option i. The
// criteria may be an object from label to description, a list of bare
// labels, or a list of {"label": ..., "description": ...} objects, which is
// how mlx-go-sys1 writes a described option list; a list may mix the two.
// A missing description is nil, which renders as the bare label.
func choiceOption(crit *jsonValue, i int) (label string, desc *jsonValue, err error) {
	if crit.kind == jObject {
		return crit.keys[i], crit.vals[i], nil
	}
	switch v := crit.arr[i]; v.kind {
	case jString:
		return v.str, nil, nil
	case jObject:
		l := v.get("label")
		if l == nil || l.kind != jString {
			return "", nil, fmt.Errorf("choice option %d needs a string 'label'", i)
		}
		return l.str, v.get("description"), nil
	}
	return "", nil, fmt.Errorf("list criteria must be strings or {label, description} objects")
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
