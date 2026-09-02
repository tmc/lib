package treesitter

import (
	"fmt"
	"io"
	"os"
)

// Parse parses src with lang. It is shorthand for creating a parser,
// setting its language, and closing it; use a [Parser] directly to parse
// more than once, to parse incrementally, or to set options.
func Parse(lang *Language, src []byte) (*Tree, error) {
	p, err := NewParser()
	if err != nil {
		return nil, err
	}
	defer p.Close()
	if err := p.SetLanguage(lang); err != nil {
		return nil, err
	}
	return p.Parse(src, nil)
}

// ParseFile reads the file at path, chooses a grammar from its name with
// [DetectLanguage], finds it with [FindLanguage], and parses it. The
// resulting tree carries the file's contents as its source.
func ParseFile(path string) (*Tree, error) {
	name := DetectLanguage(path)
	if name == "" {
		return nil, fmt.Errorf("treesitter: no known language for %s", path)
	}
	lang, err := FindLanguage(name)
	if err != nil {
		return nil, err
	}
	src, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(lang, src)
}

// readAtChunk is the size of the window ParseReaderAt keeps in memory.
const readAtChunk = 32 * 1024

// ParseReaderAt parses the first size bytes of r.
//
// tree-sitter reads its input by byte offset and revisits earlier offsets,
// so the input must support random access; that is why this takes an
// [io.ReaderAt] rather than an [io.Reader]. A *bytes.Reader, *strings.Reader,
// *os.File, or *io.SectionReader satisfies it directly.
//
// Only a window of the document is held in memory, so the tree has no
// source and [Node.Text] returns nil for its nodes. Read the file into
// memory and use [Parser.Parse] when the text is wanted afterwards.
func (p *Parser) ParseReaderAt(r io.ReaderAt, size int64, old *Tree) (*Tree, error) {
	if size < 0 {
		return nil, fmt.Errorf("treesitter: ParseReaderAt: negative size %d", size)
	}
	if size > int64(^uint32(0)) {
		return nil, fmt.Errorf("treesitter: ParseReaderAt: size %d exceeds the 4 GiB tree-sitter limit", size)
	}
	buf := make([]byte, readAtChunk)
	var readErr error
	t, err := p.ParseInput(func(byteIndex int, _ Point) []byte {
		off := int64(byteIndex)
		if off >= size || readErr != nil {
			return nil
		}
		n := int64(len(buf))
		if rem := size - off; rem < n {
			n = rem
		}
		got, err := r.ReadAt(buf[:n], off)
		if err != nil && err != io.EOF {
			// A short read past the end of the input is reported as
			// io.EOF and ends the parse; anything else is a failure.
			readErr = err
			return nil
		}
		return buf[:got]
	}, UTF8, old)
	if readErr != nil {
		if t != nil {
			t.Close()
		}
		return nil, fmt.Errorf("treesitter: ParseReaderAt: %w", readErr)
	}
	return t, err
}
