package treesitter

import (
	"fmt"

	"github.com/tmc/lib/treesitter/internal/ts"
)

// Point is a position in source text: a zero-based row and a zero-based
// byte column within that row.
type Point struct {
	Row    int
	Column int
}

// String returns the point as row:column.
func (p Point) String() string {
	return fmt.Sprintf("%d:%d", p.Row, p.Column)
}

// Range is a span of source text, in both bytes and points.
type Range struct {
	StartPoint Point
	EndPoint   Point
	StartByte  int
	EndByte    int
}

// InputEdit describes a change to a source text. StartByte and StartPoint
// must not exceed the old end byte and point.
type InputEdit struct {
	StartByte   int
	OldEndByte  int
	NewEndByte  int
	StartPoint  Point
	OldEndPoint Point
	NewEndPoint Point
}

// Symbol identifies a node type within a language.
type Symbol uint16

// FieldID identifies a field within a language.
type FieldID uint16

// StateID identifies a parse state within a language.
type StateID uint16

// SymbolType classifies a language's symbols.
type SymbolType int32

// Symbol types.
const (
	SymbolRegular   SymbolType = SymbolType(ts.SymbolRegular)
	SymbolAnonymous SymbolType = SymbolType(ts.SymbolAnonymous)
	SymbolSupertype SymbolType = SymbolType(ts.SymbolSupertype)
	SymbolAuxiliary SymbolType = SymbolType(ts.SymbolAuxiliary)
)

// String returns the name of the symbol type.
func (t SymbolType) String() string {
	switch t {
	case SymbolRegular:
		return "regular"
	case SymbolAnonymous:
		return "anonymous"
	case SymbolSupertype:
		return "supertype"
	case SymbolAuxiliary:
		return "auxiliary"
	}
	return fmt.Sprintf("SymbolType(%d)", int32(t))
}

// InputEncoding is the text encoding of parser input.
type InputEncoding int32

// Input encodings. Custom requires a decode function and is not supported
// by this package.
const (
	UTF8    InputEncoding = InputEncoding(ts.InputEncodingUTF8)
	UTF16LE InputEncoding = InputEncoding(ts.InputEncodingUTF16LE)
	UTF16BE InputEncoding = InputEncoding(ts.InputEncodingUTF16BE)
)

// LogType classifies parser log messages.
type LogType int32

// Log types.
const (
	LogParse LogType = LogType(ts.LogParse)
	LogLex   LogType = LogType(ts.LogLex)
)

// String returns the name of the log type.
func (t LogType) String() string {
	switch t {
	case LogParse:
		return "parse"
	case LogLex:
		return "lex"
	}
	return fmt.Sprintf("LogType(%d)", int32(t))
}

// Quantifier describes how many times a query capture may occur.
type Quantifier int32

// Quantifiers.
const (
	QuantifierZero       Quantifier = Quantifier(ts.QuantifierZero)
	QuantifierZeroOrOne  Quantifier = Quantifier(ts.QuantifierZeroOrOne)
	QuantifierZeroOrMore Quantifier = Quantifier(ts.QuantifierZeroOrMore)
	QuantifierOne        Quantifier = Quantifier(ts.QuantifierOne)
	QuantifierOneOrMore  Quantifier = Quantifier(ts.QuantifierOneOrMore)
)

// String returns the quantifier in query syntax.
func (q Quantifier) String() string {
	switch q {
	case QuantifierZero:
		return ""
	case QuantifierZeroOrOne:
		return "?"
	case QuantifierZeroOrMore:
		return "*"
	case QuantifierOne:
		return "1"
	case QuantifierOneOrMore:
		return "+"
	}
	return fmt.Sprintf("Quantifier(%d)", int32(q))
}

// u32 converts a Go count or offset to the uint32 the C API uses,
// clamping a negative value to zero.
func u32(n int) uint32 {
	if n < 0 {
		return 0
	}
	return uint32(n)
}

func fromPoint(p ts.Point) Point { return Point{Row: int(p.Row), Column: int(p.Column)} }
func toPoint(p Point) ts.Point   { return ts.Point{Row: u32(p.Row), Column: u32(p.Column)} }
func fromRange(r ts.Range) Range {
	return Range{
		StartPoint: fromPoint(r.StartPoint),
		EndPoint:   fromPoint(r.EndPoint),
		StartByte:  int(r.StartByte),
		EndByte:    int(r.EndByte),
	}
}
func toRange(r Range) ts.Range {
	return ts.Range{
		StartPoint: toPoint(r.StartPoint),
		EndPoint:   toPoint(r.EndPoint),
		StartByte:  u32(r.StartByte),
		EndByte:    u32(r.EndByte),
	}
}
func toInputEdit(e InputEdit) ts.InputEdit {
	return ts.InputEdit{
		StartByte:   u32(e.StartByte),
		OldEndByte:  u32(e.OldEndByte),
		NewEndByte:  u32(e.NewEndByte),
		StartPoint:  toPoint(e.StartPoint),
		OldEndPoint: toPoint(e.OldEndPoint),
		NewEndPoint: toPoint(e.NewEndPoint),
	}
}
