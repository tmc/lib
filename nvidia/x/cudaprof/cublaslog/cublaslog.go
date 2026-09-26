package cublaslog

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// A Matmul is one logged matmul call.
//
// Sizes follow cuBLAS's column-major convention: D is M×N, op(A) is M×K and
// op(B) is K×N. A caller that computes a row-major product through cuBLAS
// (PyTorch does) appears here with its operands swapped.
type Matmul struct {
	M, N, K int64
	// Batch is the batch count, 1 for an unbatched call.
	Batch int64

	// TransA and TransB are "N", "T" or "C".
	TransA, TransB string

	// Types are the matrix data types as logged, such as "R_16BF".
	A, B, C, D Layout

	Compute string // such as "COMPUTE_32F"
	Scale   string // such as "R_32F"
	// Epilogue is the epilogue without its EPILOGUE_ prefix, "DEFAULT"
	// when none was set.
	Epilogue string

	// Beta is beta as logged. Alpha is not logged.
	Beta string

	Algo Algo

	Stream string

	// Fields holds every top-level field as logged, a group without its
	// brackets.
	Fields map[string]string
}

// A Layout is one matrix operand.
type Layout struct {
	Type       string
	Rows, Cols int64
	LD         int64
	// Order is the element order without its CUBLASLT_ORDER_ prefix,
	// "COL" when none was set.
	Order string
}

// An Algo is the algorithm the call ran with.
type Algo struct {
	ID int
	// Tile and Stages drop the MATMUL_TILE_ and MATMUL_STAGES_ prefixes.
	Tile, Stages string
	// SplitK is the split-K factor, 1 when none was set.
	SplitK int
	// Fields holds the algorithm's fields as logged.
	Fields map[string]string
}

// String returns a compact description of the algorithm, such as
// "21 16x16/32x1" or "6 64x64/16x5 splitK=4".
func (a Algo) String() string {
	s := strconv.Itoa(a.ID)
	if a.Tile != "" || a.Stages != "" {
		s += " " + a.Tile + "/" + a.Stages
	}
	if a.SplitK > 1 {
		s += " splitK=" + strconv.Itoa(a.SplitK)
	}
	for _, k := range []string{"reductionScheme", "swizzle", "customOption"} {
		if v, ok := a.Fields[k]; ok && v != "0" && v != "CUBLASLT_REDUCTION_SCHEME_NONE" {
			s += " " + k + "=" + v
		}
	}
	return s
}

// Types returns the operand types, as "A,B->D" when C and D agree and
// "A,B,C->D" otherwise.
func (m Matmul) Types() string {
	if m.C.Type == m.D.Type {
		return m.A.Type + "," + m.B.Type + "->" + m.D.Type
	}
	return m.A.Type + "," + m.B.Type + "," + m.C.Type + "->" + m.D.Type
}

// Parse parses the text of one logged matmul call, the message the logger
// callback receives for a function such as cublasLtMatmul.
func Parse(line string) (Matmul, error) {
	fields, err := split(line)
	if err != nil {
		return Matmul{}, err
	}
	m := Matmul{Batch: 1, TransA: "N", TransB: "N", Epilogue: "DEFAULT", Fields: fields}
	for _, x := range []struct {
		name string
		l    *Layout
	}{{"Adesc", &m.A}, {"Bdesc", &m.B}, {"Cdesc", &m.C}, {"Ddesc", &m.D}} {
		g, ok := fields[x.name]
		if !ok {
			if x.name == "Cdesc" || x.name == "Ddesc" {
				continue
			}
			return Matmul{}, fmt.Errorf("no %s", x.name)
		}
		l, err := layout(g)
		if err != nil {
			return Matmul{}, fmt.Errorf("%s: %w", x.name, err)
		}
		*x.l = l
	}
	// cuBLASLt requires C and D to agree unless D has its own descriptor;
	// when one is missing it is the other.
	switch {
	case m.D.Type == "" && m.C.Type == "":
		return Matmul{}, errors.New("no Cdesc or Ddesc")
	case m.D.Type == "":
		m.D = m.C
	case m.C.Type == "":
		m.C = m.D
	}
	if b, ok := group(fields["Adesc"])["batchCount"]; ok {
		n, err := strconv.ParseInt(b, 10, 64)
		if err != nil {
			return Matmul{}, fmt.Errorf("Adesc batchCount %q", b)
		}
		m.Batch = n
	}
	cd := group(fields["computeDesc"])
	m.Compute = cd["computeType"]
	m.Scale = cd["scaleType"]
	if v, ok := cd["transa"]; ok {
		m.TransA = strings.TrimPrefix(v, "OP_")
	}
	if v, ok := cd["transb"]; ok {
		m.TransB = strings.TrimPrefix(v, "OP_")
	}
	if v, ok := cd["epilogue"]; ok {
		m.Epilogue = strings.TrimPrefix(v, "EPILOGUE_")
	}
	m.M, m.N = m.D.Rows, m.D.Cols
	if m.TransA == "N" {
		m.K = m.A.Cols
	} else {
		m.K = m.A.Rows
	}
	ag := group(fields["algo"])
	m.Algo = Algo{SplitK: 1, Fields: ag}
	if v, ok := ag["algoId"]; ok {
		id, err := strconv.Atoi(v)
		if err != nil {
			return Matmul{}, fmt.Errorf("algoId %q", v)
		}
		m.Algo.ID = id
	}
	m.Algo.Tile = strings.TrimPrefix(ag["tile"], "MATMUL_TILE_")
	m.Algo.Stages = strings.TrimPrefix(ag["stages"], "MATMUL_STAGES_")
	if v, ok := ag["splitK"]; ok {
		n, err := strconv.Atoi(v)
		if err != nil {
			return Matmul{}, fmt.Errorf("splitK %q", v)
		}
		m.Algo.SplitK = n
	}
	m.Beta = fields["beta"]
	m.Stream = fields["stream"]
	return m, nil
}

func layout(g string) (Layout, error) {
	f := group(g)
	l := Layout{Type: f["type"], Order: "COL"}
	if l.Type == "" {
		return Layout{}, errors.New("no type")
	}
	for _, x := range []struct {
		name string
		v    *int64
	}{{"rows", &l.Rows}, {"cols", &l.Cols}, {"ld", &l.LD}} {
		s, ok := f[x.name]
		if !ok {
			return Layout{}, fmt.Errorf("no %s", x.name)
		}
		n, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			return Layout{}, fmt.Errorf("%s %q", x.name, s)
		}
		*x.v = n
	}
	if v, ok := f["order"]; ok {
		l.Order = strings.TrimPrefix(v, "ORDER_")
	}
	return l, nil
}

// group splits the inside of a bracketed group, returning nil for an
// empty or malformed one.
func group(s string) map[string]string {
	if s == "" {
		return nil
	}
	f, err := split(s)
	if err != nil {
		return nil
	}
	return f
}

// split splits space-separated key=value fields. A value that opens a
// bracket runs to the matching close, and is returned without the brackets.
func split(s string) (map[string]string, error) {
	f := make(map[string]string)
	for {
		s = strings.TrimLeft(s, " ")
		if s == "" {
			return f, nil
		}
		eq := strings.IndexByte(s, '=')
		if eq <= 0 || strings.ContainsAny(s[:eq], " []") {
			return nil, fmt.Errorf("malformed field at %q", clip(s))
		}
		key := s[:eq]
		s = s[eq+1:]
		var val string
		if strings.HasPrefix(s, "[") {
			depth, end := 0, -1
			for i := 0; i < len(s) && end < 0; i++ {
				switch s[i] {
				case '[':
					depth++
				case ']':
					depth--
					if depth == 0 {
						end = i
					}
				}
			}
			if end < 0 {
				return nil, fmt.Errorf("unclosed group at %q", clip(key+"="+s))
			}
			val, s = s[1:end], s[end+1:]
		} else {
			sp := strings.IndexByte(s, ' ')
			if sp < 0 {
				sp = len(s)
			}
			val, s = s[:sp], s[sp:]
		}
		f[key] = val
	}
}

func clip(s string) string {
	if len(s) > 40 {
		return s[:40] + "..."
	}
	return s
}
