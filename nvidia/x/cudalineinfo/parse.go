package cudalineinfo

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudair"
)

// ParseUnit reads generated source annotated with #line directives, as
// [Unit.SourceWithLineDirectives] writes it, back into a Unit named name.
// The directives are removed from Source and their coordinates recorded in
// LineMap; a directive naming the generated file itself ends the mapping.
// Only the file and line survive the round trip. The function and callers
// of each site are not in the source, so a parsed unit rolls up no further
// than its sites.
func ParseUnit(name, annotated string) Unit {
	u := Unit{Name: name, LineMap: map[int]cudair.SourceLocation{}}
	generated := u.GeneratedName()
	var (
		b    strings.Builder
		file string
		line int
		n    int
	)
	for _, text := range strings.Split(annotated, "\n") {
		if f, l, ok := parseLineDirective(text); ok {
			file, line = f, l
			if f == generated || filepath.Base(f) == generated {
				file = ""
			}
			continue
		}
		n++
		b.WriteString(text)
		b.WriteByte('\n')
		if file != "" {
			u.LineMap[n] = cudair.SourceLocation{File: file, Line: line}
			line++
		}
	}
	u.Source = strings.TrimSuffix(b.String(), "\n")
	return u
}

// parseLineDirective parses `#line N "file"`. The file is a quoted string as
// [Unit.SourceWithLineDirectives] writes it, and may hold spaces.
func parseLineDirective(text string) (file string, line int, ok bool) {
	rest, ok := strings.CutPrefix(strings.TrimSpace(text), "#line")
	if !ok {
		return "", 0, false
	}
	rest = strings.TrimLeft(rest, " \t")
	i := strings.IndexAny(rest, " \t")
	if i < 0 {
		return "", 0, false
	}
	num, name := rest[:i], strings.TrimLeft(rest[i:], " \t")
	line, err := strconv.Atoi(num)
	if err != nil || line <= 0 {
		return "", 0, false
	}
	q, err := strconv.QuotedPrefix(name)
	if err != nil || strings.TrimSpace(name[len(q):]) != "" {
		return "", 0, false
	}
	file, err = strconv.Unquote(q)
	if err != nil {
		return "", 0, false
	}
	return file, line, true
}

// String returns the unit's name for messages.
func (u Unit) String() string {
	return fmt.Sprintf("unit %s (%d mapped lines)", u.Name, len(u.LineMap))
}
