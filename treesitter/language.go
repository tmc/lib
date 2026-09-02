package treesitter

import (
	"fmt"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/tmc/lib/treesitter/internal/ts"
)

// Language is a grammar: the set of node types, fields, and parse states a
// [Parser] uses. Languages live for the life of the process and need not be
// closed.
type Language struct {
	l ts.Language
}

// LoadLanguage opens the grammar shared library at path and returns the
// language exported by its tree_sitter_<name> function. The library stays
// open for the life of the process.
//
// LoadLanguage rejects a grammar whose ABI version this library cannot
// parse with.
func LoadLanguage(path, name string) (*Language, error) {
	if err := load(); err != nil {
		return nil, err
	}
	h, err := purego.Dlopen(path, purego.RTLD_LAZY|purego.RTLD_LOCAL)
	if err != nil {
		return nil, fmt.Errorf("treesitter: open grammar %s: %w", path, err)
	}
	sym, err := purego.Dlsym(h, "tree_sitter_"+name)
	if err != nil {
		return nil, fmt.Errorf("treesitter: grammar %s: %w", path, err)
	}
	var fn func() uintptr
	purego.RegisterFunc(&fn, sym)
	ptr := fn()
	if ptr == 0 {
		return nil, fmt.Errorf("treesitter: grammar %s: tree_sitter_%s returned NULL", path, name)
	}
	lang := &Language{l: ts.Language(ptr)}
	v := lang.ABIVersion()
	if v < MinCompatibleLanguageVersion || v > LanguageVersion {
		return nil, fmt.Errorf("treesitter: grammar %s: ABI version %d is outside the supported range %d-%d",
			path, v, MinCompatibleLanguageVersion, LanguageVersion)
	}
	return lang, nil
}

// NewLanguage wraps a TSLanguage pointer obtained elsewhere, such as from a
// grammar linked into the program. It returns nil for a zero pointer. The
// library must already be loaded or loadable; use [Load] to check.
func NewLanguage(ptr uintptr) *Language {
	if ptr == 0 {
		return nil
	}
	return &Language{l: ts.Language(ptr)}
}

// Pointer returns the underlying TSLanguage pointer.
func (l *Language) Pointer() uintptr {
	if l == nil {
		return 0
	}
	return uintptr(l.l)
}

// Name returns the language's name, or "" for a grammar generated before
// names were recorded.
func (l *Language) Name() string {
	return ts.LanguageName(l.l)
}

// ABIVersion returns the ABI version the grammar was generated with.
func (l *Language) ABIVersion() int {
	return int(ts.LanguageAbiVersion(l.l))
}

// Version returns the grammar's semantic version. ok is false when the
// grammar records no version.
func (l *Language) Version() (major, minor, patch uint8, ok bool) {
	p := ts.LanguageMetadata(l.l)
	if p == 0 {
		return 0, 0, 0, false
	}
	m := (*ts.LanguageMetadataStruct)(cptr(p))
	return m.MajorVersion, m.MinorVersion, m.PatchVersion, true
}

// SymbolCount returns the number of distinct node types.
func (l *Language) SymbolCount() int {
	return int(ts.LanguageSymbolCount(l.l))
}

// StateCount returns the number of valid parse states.
func (l *Language) StateCount() int {
	return int(ts.LanguageStateCount(l.l))
}

// FieldCount returns the number of distinct field names.
func (l *Language) FieldCount() int {
	return int(ts.LanguageFieldCount(l.l))
}

// SymbolName returns the name of a node type.
func (l *Language) SymbolName(s Symbol) string {
	return ts.LanguageSymbolName(l.l, ts.Symbol(s))
}

// SymbolForName returns the symbol for a node type name. Named selects
// between named and anonymous node types of the same spelling. It returns 0
// when the language has no such type.
func (l *Language) SymbolForName(name string, named bool) Symbol {
	return Symbol(ts.LanguageSymbolForName(l.l, stringPtr(name), uint32(len(name)), named))
}

// SymbolType classifies a node type.
func (l *Language) SymbolType(s Symbol) SymbolType {
	return SymbolType(ts.LanguageSymbolType(l.l, ts.Symbol(s)))
}

// FieldName returns the name of a field, or "" for an unknown id.
func (l *Language) FieldName(id FieldID) string {
	return ts.LanguageFieldNameForId(l.l, ts.FieldId(id))
}

// FieldID returns the id of a field name, or 0 when the language has no
// such field.
func (l *Language) FieldID(name string) FieldID {
	return FieldID(ts.LanguageFieldIdForName(l.l, stringPtr(name), uint32(len(name))))
}

// Supertypes returns the language's supertype symbols.
func (l *Language) Supertypes() []Symbol {
	var n uint32
	p := ts.LanguageSupertypes(l.l, &n)
	return symbolSlice(p, n)
}

// Subtypes returns the symbols a supertype can expand to.
func (l *Language) Subtypes(supertype Symbol) []Symbol {
	var n uint32
	p := ts.LanguageSubtypes(l.l, ts.Symbol(supertype), &n)
	return symbolSlice(p, n)
}

// NextState returns the parse state reached from state by shifting symbol.
// Together with [LookaheadIterator] it drives error recovery and
// completion.
func (l *Language) NextState(state StateID, symbol Symbol) StateID {
	return StateID(ts.LanguageNextState(l.l, ts.StateId(state), ts.Symbol(symbol)))
}

// symbolSlice copies n symbols out of language-owned memory.
func symbolSlice(p uintptr, n uint32) []Symbol {
	if p == 0 || n == 0 {
		return nil
	}
	src := unsafe.Slice((*ts.Symbol)(cptr(p)), n)
	out := make([]Symbol, n)
	for i, s := range src {
		out[i] = Symbol(s)
	}
	return out
}
