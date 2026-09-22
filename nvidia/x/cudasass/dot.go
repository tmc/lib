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
	body := text[strings.IndexByte(text, '{')+1:]
	body = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(body), "}"))
	return "digraph " + strings.Trim(s.name, `"`) + " {\n" + body + "\n}\n"
}

// ListDOTGraphs returns the names of the function graphs in Graphviz DOT text.
//
// nvdisasm prints a whole module as one digraph named "f" and puts each
// function in a "cluster_<function>" subgraph of it, so the top-level name
// identifies nothing a caller can ask for. Where those clusters are present
// they are the graphs, and the names returned are the functions'. A digraph
// with no clusters is reported under its own name.
func ListDOTGraphs(dot string) []string {
	spans := parseDOTGraphs(dot)
	names := make([]string, len(spans))
	for i, s := range spans {
		names[i] = strings.Trim(s.name, `"`)
	}
	return names
}

// ExtractDOTGraph extracts the digraph definition for the named kernel from Graphviz DOT text.
// If name is empty, the entire dot text is returned.
// Matching checks exact name, trimmed quoted name, and substring match against mangled symbols.
func ExtractDOTGraph(dot string, name string) (string, error) {
	if name == "" {
		if strings.TrimSpace(dot) == "" {
			return "", fmt.Errorf("empty control flow graph")
		}
		return dot, nil
	}
	spans := parseDOTGraphs(dot)
	if len(spans) == 0 {
		return "", fmt.Errorf("no control flow graphs found")
	}

	cleanName := strings.Trim(name, `"`)
	// 1. Exact match
	for _, s := range spans {
		if s.name == name || strings.Trim(s.name, `"`) == cleanName {
			return s.render(dot), nil
		}
	}

	// 2. Case-insensitive exact match
	for _, s := range spans {
		if strings.EqualFold(strings.Trim(s.name, `"`), cleanName) {
			return s.render(dot), nil
		}
	}

	// 3. Substring match
	var matched []dotGraphSpan
	for _, s := range spans {
		sClean := strings.Trim(s.name, `"`)
		if strings.Contains(sClean, cleanName) {
			matched = append(matched, s)
		}
	}
	if len(matched) == 1 {
		return matched[0].render(dot), nil
	}
	if len(matched) > 1 {
		names := make([]string, len(matched))
		for i, m := range matched {
			names[i] = strings.Trim(m.name, `"`)
		}
		return "", fmt.Errorf("ambiguous kernel name %q (matches: %s)", name, strings.Join(names, ", "))
	}

	available := make([]string, len(spans))
	for i, s := range spans {
		available[i] = strings.Trim(s.name, `"`)
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
			i += 2
			for i+1 < n && !(dot[i] == '*' && dot[i+1] == '/') {
				i++
			}
			if i+1 < n {
				i += 2
			}
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
						i += 2
						for i+1 < n && !(dot[i] == '*' && dot[i+1] == '/') {
							i++
						}
						if i+1 < n {
							i += 2
						}
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
					for i < n && !unicode.IsSpace(rune(dot[i])) && dot[i] != '{' && dot[i] != '/' {
						i++
					}
					graphName = dot[nameStart:i]
				}

				// Find opening brace '{'
				for i < n && dot[i] != '{' {
					i++
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
							i += 2
							for i+1 < n && !(dot[i] == '*' && dot[i+1] == '/') {
								i++
							}
							if i+1 < n {
								i += 2
							}
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
								name:  graphName,
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
		if !strings.HasPrefix(dot[i:end], "subgraph") {
			i++
			continue
		}
		j := i + len("subgraph")
		for j < end && unicode.IsSpace(rune(dot[j])) {
			j++
		}
		name, j := dotName(dot, j, end)
		if !strings.HasPrefix(strings.Trim(name, `"`), prefix) {
			i = j
			continue
		}
		for j < end && dot[j] != '{' {
			j++
		}
		body := skipBraced(dot, j, end)
		if body < 0 {
			break
		}
		spans = append(spans, dotGraphSpan{
			name:    strings.TrimPrefix(strings.Trim(name, `"`), prefix),
			start:   i,
			end:     body,
			cluster: true,
		})
		i = body
	}
	return spans
}

// dotName reads the identifier at dot[i:end], quoted or bare, and returns it
// with the index just past it.
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
	for j < end && !unicode.IsSpace(rune(dot[j])) && dot[j] != '{' {
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
