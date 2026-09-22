package cudasass

import (
	"fmt"
	"strings"
	"unicode"
)

type dotGraphSpan struct {
	name  string
	start int
	end   int
	// cluster reports that the span is a "cluster_<name>" subgraph of an
	// enclosing digraph rather than a digraph itself, so rendering it needs a
	// wrapper; see [dotGraphSpan.render].
	cluster bool
}

// render returns the span as a standalone digraph. A cluster is rewrapped
// under the function's own name: nvdisasm puts every node and edge of a
// function inside its cluster and nothing else, so the body stands alone.
func (s dotGraphSpan) render(dot string) string {
	text := strings.TrimSpace(dot[s.start:s.end])
	if !s.cluster {
		return text + "\n"
	}
	open := openBrace(text)
	if open < 0 {
		return "digraph " + dotID(s.name) + " {\n}\n"
	}
	body := text[open+1:]
	body = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(body), "}"))
	return "digraph " + dotID(s.name) + " {\n" + body + "\n}\n"
}

// openBrace returns the index of the "{" opening a graph body, or -1. The
// quoted name before it may hold a brace of its own.
func openBrace(text string) int {
	for i := 0; i < len(text); i++ {
		switch text[i] {
		case '"':
			for i++; i < len(text) && text[i] != '"'; i++ {
				if text[i] == '\\' {
					i++
				}
			}
		case '{':
			return i
		}
	}
	return -1
}

// skipBlockComment returns the index just past the "/*" comment opening at
// dot[i]. An unterminated comment runs to the end: everything after it is
// comment, including a "}" that would otherwise close a graph.
func skipBlockComment(dot string, i int) int {
	if j := strings.Index(dot[i+2:], "*/"); j >= 0 {
		return i + 2 + j + len("*/")
	}
	return len(dot)
}

// isDOTNameByte reports whether c can appear in a bare DOT identifier.
func isDOTNameByte(c byte) bool {
	return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_'
}

// dotUnquote returns the text a DOT identifier denotes: a quoted name loses
// its quotes and its \" and \\ escapes, a bare name is itself. It is the
// inverse of [dotID], so a name survives being extracted and read back.
func dotUnquote(name string) string {
	if len(name) < 2 || name[0] != '"' || name[len(name)-1] != '"' {
		return name
	}
	inner := name[1 : len(name)-1]
	if !strings.Contains(inner, `\`) {
		return inner
	}
	var b strings.Builder
	for i := 0; i < len(inner); i++ {
		if inner[i] == '\\' && i+1 < len(inner) && (inner[i+1] == '"' || inner[i+1] == '\\') {
			i++
		}
		b.WriteByte(inner[i])
	}
	return b.String()
}

// dotID returns name as a DOT identifier, quoted unless it is already a bare
// one. The symbols a disassembler prints are not all bare identifiers: an
// internal function such as $__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath
// emitted unquoted would head a graph no renderer reads. DOT admits bytes above
// 0x7f in a bare name too, but the reader here compares each byte to
// unicode.IsSpace, for which 0x85 and 0xa0 are spaces, so only ASCII is left
// bare.
func dotID(name string) string {
	bare := name != ""
	for i := 0; i < len(name) && bare; i++ {
		switch c := name[i]; {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c == '_':
		case c >= '0' && c <= '9' && i > 0:
		default:
			bare = false
		}
	}
	if bare {
		return name
	}
	return `"` + strings.NewReplacer(`\`, `\\`, `"`, `\"`).Replace(name) + `"`
}

// ListDOTGraphs returns the names of the function graphs in Graphviz DOT text.
//
// nvdisasm prints a whole module as one digraph named "f" and puts each
// function in a "cluster_<function>" subgraph of it, so the top-level name
// identifies nothing a caller can ask for. Where those clusters are present
// they are the graphs, and the names returned are the functions'. A digraph
// with no clusters is reported under its own name.
//
// Every name returned is one [ExtractDOTGraph] will extract. A name that
// unquotes to nothing is not reported: DOT permits an anonymous graph, and an
// empty name is how ExtractDOTGraph is asked for the whole document, so such a
// graph names nothing a caller can ask for.
func ListDOTGraphs(dot string) []string {
	var names []string
	for _, s := range parseDOTGraphs(dot) {
		if dotUnquote(s.name) != "" {
			names = append(names, s.name)
		}
	}
	return names
}

// ExtractDOTGraph extracts the digraph definition for the named kernel from Graphviz DOT text.
// If name is empty once unquoted, it names no kernel and the entire dot text is returned.
// Matching checks exact name, trimmed quoted name, and substring match against mangled symbols.
func ExtractDOTGraph(dot string, name string) (string, error) {
	cleanName := dotUnquote(name)
	if cleanName == "" {
		if strings.TrimSpace(dot) == "" {
			return "", fmt.Errorf("empty control flow graph")
		}
		return dot, nil
	}
	spans := parseDOTGraphs(dot)
	if len(spans) == 0 {
		return "", fmt.Errorf("no control flow graphs found")
	}

	// 1. Exact match, on the name as given and then as it unquotes, so that
	// a name [ListDOTGraphs] reported is found whichever form it is in.
	for _, s := range spans {
		if s.name == name {
			return s.render(dot), nil
		}
	}
	for _, s := range spans {
		if s.name == cleanName {
			return s.render(dot), nil
		}
	}

	// 2. Case-insensitive exact match
	for _, s := range spans {
		if strings.EqualFold(s.name, cleanName) {
			return s.render(dot), nil
		}
	}

	// 3. Substring match
	var matched []dotGraphSpan
	for _, s := range spans {
		if strings.Contains(s.name, cleanName) {
			matched = append(matched, s)
		}
	}
	if len(matched) == 1 {
		return matched[0].render(dot), nil
	}
	if len(matched) > 1 {
		names := make([]string, len(matched))
		for i, m := range matched {
			names[i] = m.name
		}
		return "", fmt.Errorf("ambiguous kernel name %q (matches: %s)", name, strings.Join(names, ", "))
	}

	available := make([]string, len(spans))
	for i, s := range spans {
		available[i] = s.name
	}
	return "", fmt.Errorf("kernel %q not found in control flow graph (available: %s)", name, strings.Join(available, ", "))
}

func parseDOTGraphs(dot string) []dotGraphSpan {
	var spans []dotGraphSpan
	n := len(dot)
	i := 0

	for i < n {
		// Skip whitespace
		if unicode.IsSpace(rune(dot[i])) {
			i++
			continue
		}

		// Skip comments
		if dot[i] == '/' && i+1 < n && dot[i+1] == '/' {
			i += 2
			for i < n && dot[i] != '\n' {
				i++
			}
			continue
		}
		if dot[i] == '/' && i+1 < n && dot[i+1] == '*' {
			i = skipBlockComment(dot, i)
			continue
		}
		if dot[i] == '#' {
			i++
			for i < n && dot[i] != '\n' {
				i++
			}
			continue
		}

		// Check for digraph keyword
		if strings.HasPrefix(dot[i:], "digraph") {
			after := i + len("digraph")
			if after == n || unicode.IsSpace(rune(dot[after])) || dot[after] == '{' || dot[after] == '/' || dot[after] == '"' {
				graphStart := i
				i = after

				// Parse graph name (optional in DOT, but nvdisasm always names it)
				var graphName string
				for i < n {
					if unicode.IsSpace(rune(dot[i])) {
						i++
						continue
					}
					// Skip comments between digraph and name
					if dot[i] == '/' && i+1 < n && dot[i+1] == '/' {
						i += 2
						for i < n && dot[i] != '\n' {
							i++
						}
						continue
					}
					if dot[i] == '/' && i+1 < n && dot[i+1] == '*' {
						i = skipBlockComment(dot, i)
						continue
					}
					break
				}

				if i < n && dot[i] == '"' {
					nameStart := i
					i++
					for i < n && dot[i] != '"' {
						if dot[i] == '\\' && i+1 < n {
							i += 2
						} else {
							i++
						}
					}
					if i < n && dot[i] == '"' {
						i++
					}
					graphName = dot[nameStart:i]
				} else if i < n && dot[i] != '{' {
					nameStart := i
					for i < n && !unicode.IsSpace(rune(dot[i])) && dot[i] != '{' && dot[i] != '/' && dot[i] != '"' {
						i++
					}
					graphName = dot[nameStart:i]
				}

				// Find opening brace '{', which a quoted name or a stray
				// string before it may hold a copy of.
				if rel := openBrace(dot[i:]); rel >= 0 {
					i += rel
				} else {
					i = n
				}

				if i < n && dot[i] == '{' {
					depth := 1
					i++
					for i < n && depth > 0 {
						if dot[i] == '"' {
							i++
							for i < n && dot[i] != '"' {
								if dot[i] == '\\' && i+1 < n {
									i += 2
								} else {
									i++
								}
							}
							if i < n && dot[i] == '"' {
								i++
							}
							continue
						}
						if dot[i] == '/' && i+1 < n && dot[i+1] == '/' {
							i += 2
							for i < n && dot[i] != '\n' {
								i++
							}
							continue
						}
						if dot[i] == '/' && i+1 < n && dot[i+1] == '*' {
							i = skipBlockComment(dot, i)
							continue
						}
						if dot[i] == '{' {
							depth++
						} else if dot[i] == '}' {
							depth--
						}
						i++
					}

					if depth == 0 {
						// nvdisasm names the whole module "f" and puts the
						// functions in clusters, so the clusters are the graphs
						// a caller can name. A digraph without them is its own.
						if cs := clusterSpans(dot, graphStart, i); len(cs) > 0 {
							spans = append(spans, cs...)
						} else {
							spans = append(spans, dotGraphSpan{
								name:  dotUnquote(graphName),
								start: graphStart,
								end:   i,
							})
						}
						continue
					}
				}
			}
		}

		i++
	}

	return spans
}

// clusterSpans returns the "cluster_<name>" subgraphs of the digraph occupying
// dot[start:end], with the prefix stripped from each name. It returns nothing
// for a digraph that has no such subgraph, which is the shape a caller that
// writes its own DOT produces.
func clusterSpans(dot string, start, end int) []dotGraphSpan {
	const prefix = "cluster_"
	var spans []dotGraphSpan
	for i := start; i < end; {
		// A node label holds arbitrary text, so the keyword counts only
		// where the scanner is reading DOT and not a string or a comment.
		switch {
		case dot[i] == '"':
			for i++; i < end && dot[i] != '"'; i++ {
				if dot[i] == '\\' {
					i++
				}
			}
			i++
			continue
		case dot[i] == '/' && i+1 < end && dot[i+1] == '/':
			for i += 2; i < end && dot[i] != '\n'; i++ {
			}
			continue
		case dot[i] == '/' && i+1 < end && dot[i+1] == '*':
			i = skipBlockComment(dot, i)
			continue
		}
		j := i + len("subgraph")
		// "subgraph" is a keyword only as a whole token: a graph named
		// subgraphcluster_x holds the text of one and is not a cluster.
		if !strings.HasPrefix(dot[i:end], "subgraph") ||
			(i > start && isDOTNameByte(dot[i-1])) ||
			(j < end && isDOTNameByte(dot[j])) {
			i++
			continue
		}
		for j < end && unicode.IsSpace(rune(dot[j])) {
			j++
		}
		name, j := dotName(dot, j, end)
		clean := dotUnquote(name)
		if !strings.HasPrefix(clean, prefix) {
			i = j
			continue
		}
		fn := strings.TrimPrefix(clean, prefix)
		if dotUnquote(fn) == "" {
			// A cluster carrying no function name names nothing a caller
			// can ask for, so it is not one of the graphs.
			i = j
			continue
		}
		rel := openBrace(dot[j:end])
		if rel < 0 {
			break
		}
		j += rel
		body := skipBraced(dot, j, end)
		if body < 0 {
			break
		}
		spans = append(spans, dotGraphSpan{
			name:    fn,
			start:   i,
			end:     body,
			cluster: true,
		})
		i = body
	}
	return spans
}

// dotName reads the identifier at dot[i:end], quoted or bare, and returns it
// with the index just past it. A bare name stops at a quote: letting one
// absorb a quote puts the scan out of phase with every scanner here that
// skips quoted text, so a brace inside a string would open a graph body.
func dotName(dot string, i, end int) (string, int) {
	if i < end && dot[i] == '"' {
		j := i + 1
		for j < end && dot[j] != '"' {
			if dot[j] == '\\' && j+1 < end {
				j += 2
				continue
			}
			j++
		}
		if j < end {
			j++
		}
		return dot[i:j], j
	}
	j := i
	for j < end && !unicode.IsSpace(rune(dot[j])) && dot[j] != '{' && dot[j] != '"' {
		j++
	}
	return dot[i:j], j
}

// skipBraced returns the index just past the "}" matching the "{" at dot[i],
// or -1 if the braces do not balance within end. Quoted strings are skipped,
// because a node label holds braces of its own.
func skipBraced(dot string, i, end int) int {
	if i >= end || dot[i] != '{' {
		return -1
	}
	depth := 0
	for ; i < end; i++ {
		switch dot[i] {
		case '"':
			for i++; i < end && dot[i] != '"'; i++ {
				if dot[i] == '\\' && i+1 < end {
					i++
				}
			}
		case '{':
			depth++
		case '}':
			if depth--; depth == 0 {
				return i + 1
			}
		}
	}
	return -1
}
