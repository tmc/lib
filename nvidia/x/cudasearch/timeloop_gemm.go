package cudasearch

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/cudatensor"
)

const (
	// maxTimeloopBytes bounds TimeloopGEMMOptions.MaxBytes.
	maxTimeloopBytes = 1 << 20
	// maxTimeloopDepth bounds object and array nesting. The exported schema
	// nests at most seven levels.
	maxTimeloopDepth = 8
)

// TimeloopGEMMOptions configures one selected temporal analytical search.
//
// Einsum names the operation to search; none is chosen implicitly. MaxBytes
// bounds the bytes read and retained from the document and must be positive
// and at most 1 MiB. It does not bound the time a Read takes. ElementBytes maps
// exactly the selected operation's three external tensor names, as written in
// the document, to positive element sizes in bytes.
//
// The document carries no schedule, element types or hardware limits. The
// descriptors and Constraints are passed unchanged to [NewLinearPrunedSearch]
// in the B/M/N/K coordinates derived from the document; nil Permutations
// selects its default order. ElementBytes, the descriptors and Constraints are
// caller assertions, not evidence of tensor dtypes or hardware. Options are
// borrowed for the call and must not be modified concurrently.
type TimeloopGEMMOptions struct {
	Einsum         string
	MaxBytes       int64
	ElementBytes   map[string]int
	Factorizations []cudalayout.IndexFactorization
	Permutations   []cudalayout.LoopPermutation
	Bypasses       []cudalayout.DatatypeBypass
	Constraints    cudalayout.HardwareConstraints
}

// TimeloopGEMMResult keeps document identities beside the actual search result.
//
// ShapeBounds holds every Shape bound of the document verbatim. RankNames maps
// the selected einsum's ranks to B, M, N or K, and TensorNames maps roles to
// external tensor names. The document does not record pre-canonical rank
// names. Stats is the [LinearPrunedSearch.Search] result for the selected
// einsum alone, unfused; it is not a whole-graph or fused estimate. All maps
// and Stats.Best belong to the caller.
type TimeloopGEMMResult struct {
	Einsum      string
	ShapeBounds map[string]string
	RankNames   map[cudashape.DimToken]string
	TensorNames map[cudalayout.TensorRole]string
	Stats       MapSpaceSearchStats
}

// SearchTimeloopGEMM reads a workload written by
// [cudatensor.ExportTimeloopWorkload], maps the selected einsum with
// [cudatensor.MapCanonicalGEMM] and runs [LinearPrunedSearch.Search] on it
// with the identity spatial temporal model.
//
// It reads only the exported JSON schema. Duplicate, unknown or case-folded
// keys, nulls, wrong types, missing fields, trailing values and nesting deeper
// than 8 are refused, as are default or per-einsum renames other than those
// the exporter writes. Every rank must be one upper-case ASCII letter followed
// by optional digits, and the projected ranks must be exactly the Shape ranks.
// Conformance does not prove that the exporter or a canonical graph produced
// the document.
//
// The selected einsum must be a rank-2 MK,KN->MN or batched rank-3
// BMK,BKN->BMN contraction of three distinct tensors whose Shape bounds are
// positive decimal integers. Symbolic, interval and other bounds are refused,
// not resolved. Other einsums are checked for structure only; their bounds are
// not evaluated and they contribute no cost.
//
// A refused document or option returns a zero result. If Search fails, the
// result holds the document identities and Search's partial Stats, and the
// error wraps Search's error. ctx is checked before and after reading; it
// cannot interrupt a blocking Read.
func SearchTimeloopGEMM(ctx context.Context, r io.Reader, options TimeloopGEMMOptions) (TimeloopGEMMResult, error) {
	if ctx == nil {
		return TimeloopGEMMResult{}, errors.New("cudasearch: nil context")
	}
	if r == nil {
		return TimeloopGEMMResult{}, errors.New("cudasearch: nil Timeloop reader")
	}
	if options.Einsum == "" {
		return TimeloopGEMMResult{}, errors.New("cudasearch: no Timeloop einsum selected")
	}
	if options.MaxBytes <= 0 || options.MaxBytes > maxTimeloopBytes {
		return TimeloopGEMMResult{}, fmt.Errorf("cudasearch: Timeloop MaxBytes %d is not in [1, %d]", options.MaxBytes, maxTimeloopBytes)
	}
	if err := ctx.Err(); err != nil {
		return TimeloopGEMMResult{}, fmt.Errorf("cudasearch: Timeloop search: %w", err)
	}
	data, err := io.ReadAll(io.LimitReader(r, options.MaxBytes+1))
	if err != nil {
		return TimeloopGEMMResult{}, fmt.Errorf("cudasearch: reading Timeloop workload: %w", err)
	}
	if int64(len(data)) > options.MaxBytes {
		return TimeloopGEMMResult{}, fmt.Errorf("cudasearch: Timeloop workload exceeds %d bytes", options.MaxBytes)
	}
	if err := ctx.Err(); err != nil {
		return TimeloopGEMMResult{}, fmt.Errorf("cudasearch: Timeloop search: %w", err)
	}
	doc, err := decodeTimeloop(data)
	if err != nil {
		return TimeloopGEMMResult{}, err
	}
	gemm, err := doc.selectGEMM(options.Einsum)
	if err != nil {
		return TimeloopGEMMResult{}, err
	}
	dtypes, err := gemm.dtypes(options.Einsum, options.ElementBytes)
	if err != nil {
		return TimeloopGEMMResult{}, err
	}
	search := NewLinearPrunedSearch(options.Factorizations, options.Permutations, options.Bypasses, options.Constraints, dtypes, gemm.problem)
	result := TimeloopGEMMResult{
		Einsum:      options.Einsum,
		ShapeBounds: maps.Clone(doc.shape),
		RankNames:   maps.Clone(gemm.rankNames),
		TensorNames: map[cudalayout.TensorRole]string{
			cudalayout.TensorA: gemm.names[0],
			cudalayout.TensorB: gemm.names[1],
			cudalayout.TensorC: gemm.names[2],
		},
	}
	result.Stats, err = search.Search(ctx)
	if err != nil {
		return result, fmt.Errorf("cudasearch: Timeloop einsum %q: %w", options.Einsum, err)
	}
	return result, nil
}

// timeloopDoc is the admitted subset of an exported workload.
type timeloopDoc struct {
	shape    map[string]string
	einsums  []timeloopEinsum
	defaults []timeloopRuleSet
}

type timeloopEinsum struct {
	name     string
	accesses []timeloopAccess
	renames  timeloopRenames
}

type timeloopAccess struct {
	name       string
	projection []string
	output     bool
}

type timeloopRenames struct {
	input, weight, output string
}

type timeloopRuleSet struct {
	name  string
	rules []timeloopRule
}

type timeloopRule struct {
	name, source string
	count        int
}

// timeloopDefaultRules are the default tensor-role renames the exporter writes.
var timeloopDefaultRules = map[string]string{
	"input":  "Inputs() & Intermediates()",
	"output": "Outputs()",
	"weight": "~(input | output)",
}

func decodeTimeloop(data []byte) (*timeloopDoc, error) {
	if !utf8.Valid(data) {
		return nil, errors.New("cudasearch: Timeloop workload: not valid UTF-8")
	}
	if err := checkTimeloopDepth(data); err != nil {
		return nil, fmt.Errorf("cudasearch: Timeloop workload: %w", err)
	}
	d := timeloopDecoder{json.NewDecoder(bytes.NewReader(data))}
	d.dec.UseNumber()
	doc := new(timeloopDoc)
	err := d.object("document", []string{"workload", "renames"}, func(key string) error {
		var err error
		switch key {
		case "workload":
			err = d.workload(doc)
		case "renames":
			doc.defaults, err = d.renames("renames")
		default:
			err = timeloopUnknown("document", key)
		}
		return err
	})
	if err == nil {
		if _, terr := d.dec.Token(); terr != io.EOF {
			err = errors.New("trailing data after workload")
		}
	}
	if err != nil {
		return nil, fmt.Errorf("cudasearch: Timeloop workload: %w", err)
	}
	if err := doc.validate(); err != nil {
		return nil, err
	}
	return doc, nil
}

// checkTimeloopDepth rejects nesting deeper than maxTimeloopDepth without
// decoding. Malformed input is left to the decoder.
func checkTimeloopDepth(data []byte) error {
	depth := 0
	inString, escaped := false, false
	for _, c := range data {
		if inString {
			switch {
			case escaped:
				escaped = false
			case c == '\\':
				escaped = true
			case c == '"':
				inString = false
			}
			continue
		}
		switch c {
		case '"':
			inString = true
		case '{', '[':
			depth++
			if depth > maxTimeloopDepth {
				return fmt.Errorf("nesting deeper than %d", maxTimeloopDepth)
			}
		case '}', ']':
			depth--
		}
	}
	return nil
}

// timeloopDecoder decodes the exported schema token by token, so that keys
// match exactly and every value has its schema type.
type timeloopDecoder struct {
	dec *json.Decoder
}

func (d timeloopDecoder) token(what string) (json.Token, error) {
	tok, err := d.dec.Token()
	if err != nil {
		return nil, fmt.Errorf("%s: %w", what, err)
	}
	return tok, nil
}

func (d timeloopDecoder) delim(what string, want json.Delim) error {
	tok, err := d.token(what)
	if err != nil {
		return err
	}
	if tok != want {
		return fmt.Errorf("%s: got %s, want %s", what, timeloopKind(tok), timeloopKind(want))
	}
	return nil
}

// object decodes an object, calling field to decode the value of each key.
// Keys may not repeat, and every key in required must be present.
func (d timeloopDecoder) object(what string, required []string, field func(key string) error) error {
	if err := d.delim(what, '{'); err != nil {
		return err
	}
	seen := make(map[string]bool)
	for d.dec.More() {
		tok, err := d.token(what)
		if err != nil {
			return err
		}
		key, ok := tok.(string)
		if !ok {
			return fmt.Errorf("%s: got %s, want key", what, timeloopKind(tok))
		}
		if seen[key] {
			return fmt.Errorf("%s: duplicate key %q", what, key)
		}
		seen[key] = true
		if err := field(key); err != nil {
			return err
		}
	}
	if err := d.delim(what, '}'); err != nil {
		return err
	}
	for _, key := range required {
		if !seen[key] {
			return fmt.Errorf("%s: missing key %q", what, key)
		}
	}
	return nil
}

func (d timeloopDecoder) array(what string, elem func(i int) error) error {
	if err := d.delim(what, '['); err != nil {
		return err
	}
	for i := 0; d.dec.More(); i++ {
		if err := elem(i); err != nil {
			return err
		}
	}
	return d.delim(what, ']')
}

func (d timeloopDecoder) str(what string) (string, error) {
	tok, err := d.token(what)
	if err != nil {
		return "", err
	}
	s, ok := tok.(string)
	if !ok {
		return "", fmt.Errorf("%s: got %s, want string", what, timeloopKind(tok))
	}
	return s, nil
}

func (d timeloopDecoder) integer(what string) (int, error) {
	tok, err := d.token(what)
	if err != nil {
		return 0, err
	}
	n, ok := tok.(json.Number)
	if !ok {
		return 0, fmt.Errorf("%s: got %s, want integer", what, timeloopKind(tok))
	}
	i, err := strconv.Atoi(string(n))
	if err != nil {
		return 0, fmt.Errorf("%s: %s is not an integer", what, n)
	}
	return i, nil
}

func (d timeloopDecoder) boolean(what string) (bool, error) {
	tok, err := d.token(what)
	if err != nil {
		return false, err
	}
	b, ok := tok.(bool)
	if !ok {
		return false, fmt.Errorf("%s: got %s, want boolean", what, timeloopKind(tok))
	}
	return b, nil
}

func (d timeloopDecoder) workload(doc *timeloopDoc) error {
	return d.object("workload", []string{"version", "shape", "einsums"}, func(key string) error {
		switch key {
		case "version":
			v, err := d.str("workload.version")
			if err != nil {
				return err
			}
			if v != "0.5" {
				return fmt.Errorf("workload.version %q is not 0.5", v)
			}
			return nil
		case "shape":
			doc.shape = make(map[string]string)
			return d.object("workload.shape", nil, func(rank string) error {
				bound, err := d.str(fmt.Sprintf("workload.shape[%q]", rank))
				doc.shape[rank] = bound
				return err
			})
		case "einsums":
			return d.array("workload.einsums", func(i int) error {
				e, err := d.einsum(fmt.Sprintf("workload.einsums[%d]", i))
				doc.einsums = append(doc.einsums, e)
				return err
			})
		}
		return timeloopUnknown("workload", key)
	})
}

func (d timeloopDecoder) einsum(what string) (timeloopEinsum, error) {
	var e timeloopEinsum
	err := d.object(what, []string{"name", "tensor_accesses", "renames"}, func(key string) error {
		var err error
		switch key {
		case "name":
			e.name, err = d.str(what + ".name")
		case "tensor_accesses":
			err = d.array(what+".tensor_accesses", func(i int) error {
				a, err := d.access(fmt.Sprintf("%s.tensor_accesses[%d]", what, i))
				e.accesses = append(e.accesses, a)
				return err
			})
		case "renames":
			e.renames, err = d.einsumRenames(what + ".renames")
		default:
			err = timeloopUnknown(what, key)
		}
		return err
	})
	return e, err
}

func (d timeloopDecoder) access(what string) (timeloopAccess, error) {
	var a timeloopAccess
	err := d.object(what, []string{"name", "projection"}, func(key string) error {
		var err error
		switch key {
		case "name":
			a.name, err = d.str(what + ".name")
		case "projection":
			err = d.array(what+".projection", func(i int) error {
				rank, err := d.str(fmt.Sprintf("%s.projection[%d]", what, i))
				a.projection = append(a.projection, rank)
				return err
			})
		case "output":
			a.output, err = d.boolean(what + ".output")
		default:
			err = timeloopUnknown(what, key)
		}
		return err
	})
	return a, err
}

func (d timeloopDecoder) einsumRenames(what string) (timeloopRenames, error) {
	var r timeloopRenames
	err := d.object(what, []string{"output", "input", "weight"}, func(key string) error {
		var err error
		switch key {
		case "output":
			r.output, err = d.str(what + ".output")
		case "input":
			r.input, err = d.str(what + ".input")
		case "weight":
			r.weight, err = d.str(what + ".weight")
		default:
			err = timeloopUnknown(what, key)
		}
		return err
	})
	return r, err
}

func (d timeloopDecoder) renames(what string) ([]timeloopRuleSet, error) {
	var sets []timeloopRuleSet
	err := d.object(what, []string{"einsums"}, func(key string) error {
		if key != "einsums" {
			return timeloopUnknown(what, key)
		}
		return d.array(what+".einsums", func(i int) error {
			set, err := d.ruleSet(fmt.Sprintf("%s.einsums[%d]", what, i))
			sets = append(sets, set)
			return err
		})
	})
	return sets, err
}

func (d timeloopDecoder) ruleSet(what string) (timeloopRuleSet, error) {
	var s timeloopRuleSet
	err := d.object(what, []string{"name", "tensor_accesses"}, func(key string) error {
		var err error
		switch key {
		case "name":
			s.name, err = d.str(what + ".name")
		case "tensor_accesses":
			err = d.array(what+".tensor_accesses", func(i int) error {
				r, err := d.rule(fmt.Sprintf("%s.tensor_accesses[%d]", what, i))
				s.rules = append(s.rules, r)
				return err
			})
		default:
			err = timeloopUnknown(what, key)
		}
		return err
	})
	return s, err
}

func (d timeloopDecoder) rule(what string) (timeloopRule, error) {
	var r timeloopRule
	err := d.object(what, []string{"name", "source", "expected_count"}, func(key string) error {
		var err error
		switch key {
		case "name":
			r.name, err = d.str(what + ".name")
		case "source":
			r.source, err = d.str(what + ".source")
		case "expected_count":
			r.count, err = d.integer(what + ".expected_count")
		default:
			err = timeloopUnknown(what, key)
		}
		return err
	})
	return r, err
}

func timeloopUnknown(what, key string) error {
	return fmt.Errorf("%s: unknown key %q", what, key)
}

func timeloopKind(tok json.Token) string {
	switch tok := tok.(type) {
	case nil:
		return "null"
	case bool:
		return "boolean"
	case json.Number:
		return "number"
	case string:
		return "string"
	case json.Delim:
		switch tok {
		case '{':
			return "object"
		case '[':
			return "array"
		}
		return strconv.Quote(tok.String())
	}
	return fmt.Sprintf("%T", tok)
}

// validate checks the structure the exporter guarantees for every einsum.
func (doc *timeloopDoc) validate() error {
	if err := validateTimeloopDefaults(doc.defaults); err != nil {
		return err
	}
	names := make(map[string]bool, len(doc.einsums))
	projected := make(map[string]bool, len(doc.shape))
	for i, e := range doc.einsums {
		if e.name == "" {
			return fmt.Errorf("cudasearch: Timeloop einsum %d has no name", i)
		}
		if names[e.name] {
			return fmt.Errorf("cudasearch: duplicate Timeloop einsum %q", e.name)
		}
		names[e.name] = true
		var inputs, outputs []string
		for j, a := range e.accesses {
			what := fmt.Sprintf("cudasearch: Timeloop einsum %q access %d", e.name, j)
			if a.name == "" {
				return fmt.Errorf("%s has no tensor name", what)
			}
			if len(a.projection) == 0 {
				return fmt.Errorf("%s has no projection", what)
			}
			ranks := make(map[string]bool, len(a.projection))
			for _, rank := range a.projection {
				if !timeloopRank(rank) {
					return fmt.Errorf("%s rank %q is not a canonical rank", what, rank)
				}
				if ranks[rank] {
					return fmt.Errorf("%s repeats rank %q", what, rank)
				}
				if _, ok := doc.shape[rank]; !ok {
					return fmt.Errorf("%s rank %q has no shape bound", what, rank)
				}
				ranks[rank] = true
				projected[rank] = true
			}
			if a.output {
				outputs = append(outputs, a.name)
				continue
			}
			if len(outputs) > 0 {
				return fmt.Errorf("%s is an input after an output", what)
			}
			inputs = append(inputs, a.name)
		}
		if len(inputs) == 0 || len(outputs) == 0 {
			return fmt.Errorf("cudasearch: Timeloop einsum %q needs input and output accesses", e.name)
		}
		weight := "Nothing()"
		if len(inputs) > 1 {
			weight = inputs[1]
		}
		for _, r := range []struct{ field, got, want string }{
			{"input", e.renames.input, inputs[0]},
			{"output", e.renames.output, outputs[0]},
			{"weight", e.renames.weight, weight},
		} {
			if r.got != r.want {
				return fmt.Errorf("cudasearch: Timeloop einsum %q renames %s = %q, want %q", e.name, r.field, r.got, r.want)
			}
		}
	}
	for _, rank := range slices.Sorted(maps.Keys(doc.shape)) {
		if !projected[rank] {
			return fmt.Errorf("cudasearch: Timeloop shape rank %q is not projected by any einsum", rank)
		}
	}
	return nil
}

func validateTimeloopDefaults(sets []timeloopRuleSet) error {
	if len(sets) != 1 || sets[0].name != "default" {
		return errors.New(`cudasearch: Timeloop default renames are not one "default" rule set`)
	}
	rules := sets[0].rules
	if len(rules) != len(timeloopDefaultRules) {
		return fmt.Errorf("cudasearch: Timeloop default renames have %d rules, want %d", len(rules), len(timeloopDefaultRules))
	}
	seen := make(map[string]bool, len(rules))
	for _, r := range rules {
		source, ok := timeloopDefaultRules[r.name]
		if !ok || seen[r.name] || r.source != source || r.count != 1 {
			return fmt.Errorf("cudasearch: Timeloop default renames rule %q (source %q, expected_count %d) is not an exporter rule", r.name, r.source, r.count)
		}
		seen[r.name] = true
	}
	return nil
}

// timeloopRank reports whether s is one ASCII upper-case letter followed by
// optional ASCII digits, which cudashape.ParseDimTokens keeps as one token.
func timeloopRank(s string) bool {
	if s == "" || s[0] < 'A' || s[0] > 'Z' {
		return false
	}
	for i := 1; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

// timeloopExtent parses a positive decimal integer bound.
func timeloopExtent(bound string) (int, error) {
	if bound == "" || bound[0] < '1' || bound[0] > '9' {
		return 0, errors.New("not a positive decimal integer")
	}
	for i := 1; i < len(bound); i++ {
		if bound[i] < '0' || bound[i] > '9' {
			return 0, errors.New("not a positive decimal integer")
		}
	}
	return strconv.Atoi(bound)
}

// timeloopGEMM is a selected einsum mapped to B/M/N/K and A/B/C.
type timeloopGEMM struct {
	names     [3]string // external names of A, B and C
	problem   map[string]int
	rankNames map[cudashape.DimToken]string
}

func (doc *timeloopDoc) selectGEMM(name string) (timeloopGEMM, error) {
	i := slices.IndexFunc(doc.einsums, func(e timeloopEinsum) bool { return e.name == name })
	if i < 0 {
		return timeloopGEMM{}, fmt.Errorf("cudasearch: Timeloop einsum %q not found", name)
	}
	e := doc.einsums[i]
	var inputs, outputs []timeloopAccess
	for _, a := range e.accesses {
		if a.output {
			outputs = append(outputs, a)
		} else {
			inputs = append(inputs, a)
		}
	}
	if len(inputs) != 2 || len(outputs) != 1 {
		return timeloopGEMM{}, fmt.Errorf("cudasearch: Timeloop einsum %q has %d input and %d output accesses, want 2 and 1", name, len(inputs), len(outputs))
	}
	g := timeloopGEMM{names: [3]string{e.renames.input, e.renames.weight, e.renames.output}}
	if g.names[0] == g.names[1] || g.names[0] == g.names[2] || g.names[1] == g.names[2] {
		return timeloopGEMM{}, fmt.Errorf("cudasearch: Timeloop einsum %q tensor names %q, %q and %q are not distinct", name, g.names[0], g.names[1], g.names[2])
	}
	extents := make(map[cudashape.DimToken]int)
	var operands [3]string
	for j, a := range []timeloopAccess{inputs[0], inputs[1], outputs[0]} {
		for _, rank := range a.projection {
			extent, err := timeloopExtent(doc.shape[rank])
			if err != nil {
				return timeloopGEMM{}, fmt.Errorf("cudasearch: Timeloop einsum %q rank %q bound %q: %w", name, rank, doc.shape[rank], err)
			}
			extents[cudashape.DimToken(rank)] = extent
		}
		operands[j] = strings.Join(a.projection, "")
	}
	w, err := cudatensor.MapCanonicalGEMM(operands[0]+","+operands[1]+"->"+operands[2], extents)
	if err != nil {
		return timeloopGEMM{}, fmt.Errorf("cudasearch: Timeloop einsum %q: %w", name, err)
	}
	if !distinctTimeloopGEMM(w, len(extents)) {
		return timeloopGEMM{}, fmt.Errorf("cudasearch: Timeloop einsum %q ranks %v do not map to distinct B/M/N/K", name, w.RankNames)
	}
	g.problem, g.rankNames = w.Problem, w.RankNames
	return g, nil
}

// distinctTimeloopGEMM reports whether w maps distinct ranks to M, N
// and K, plus B when batched, with a positive B/M/N/K problem.
func distinctTimeloopGEMM(w cudatensor.GEMMWorkload, ranks int) bool {
	want := []string{"M", "N", "K"}
	switch ranks {
	case 3:
	case 4:
		want = append(want, "B")
	default:
		return false
	}
	if len(w.RankNames) != ranks {
		return false
	}
	roles := make(map[string]bool, ranks)
	for _, role := range w.RankNames {
		roles[role] = true
	}
	for _, role := range want {
		if !roles[role] {
			return false
		}
	}
	if len(w.Problem) != 4 || (ranks == 3 && w.Problem["B"] != 1) {
		return false
	}
	for _, dim := range []string{"B", "M", "N", "K"} {
		if w.Problem[dim] <= 0 {
			return false
		}
	}
	return true
}

// dtypes maps sizes, keyed by external tensor name, to roles.
func (g timeloopGEMM) dtypes(einsum string, sizes map[string]int) (map[cudalayout.TensorRole]int, error) {
	if len(sizes) != len(g.names) {
		return nil, fmt.Errorf("cudasearch: Timeloop einsum %q ElementBytes has %d names, want %q, %q and %q", einsum, len(sizes), g.names[0], g.names[1], g.names[2])
	}
	dtypes := make(map[cudalayout.TensorRole]int, len(g.names))
	for i, role := range []cudalayout.TensorRole{cudalayout.TensorA, cudalayout.TensorB, cudalayout.TensorC} {
		size, ok := sizes[g.names[i]]
		if !ok || size <= 0 {
			return nil, fmt.Errorf("cudasearch: Timeloop einsum %q ElementBytes[%q] is not a positive size", einsum, g.names[i])
		}
		dtypes[role] = size
	}
	return dtypes, nil
}
