// Package elfinspect reads ELF headers, sections, segments, and symbol tables.
// It preserves file metadata without loading code or interpreting symbol names
// as an admission policy. In particular, GNU-unique bindings remain raw value
// 10, and TLS, undefined, absolute, and common symbols remain distinct.
//
// Read reports absent symbol tables explicitly. Neither a present table nor
// any section or PT_LOAD mapping proves runtime storage or loaded addresses.
package elfinspect
