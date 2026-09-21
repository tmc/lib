package cudalineinfo

import "strings"

// stripDirectives removes #line directives, recovering the source a unit
// holds from the annotated file that was written to disk.
func stripDirectives(src string) string {
	var out []string
	for _, l := range strings.Split(src, "\n") {
		if strings.HasPrefix(l, "#line ") {
			continue
		}
		out = append(out, l)
	}
	return strings.Join(out, "\n")
}
