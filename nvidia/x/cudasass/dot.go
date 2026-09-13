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
}

// ListDOTGraphs returns the names of all top-level digraphs found in Graphviz DOT text.
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
			return strings.TrimSpace(dot[s.start:s.end]) + "\n", nil
		}
	}

	// 2. Case-insensitive exact match
	for _, s := range spans {
		if strings.EqualFold(strings.Trim(s.name, `"`), cleanName) {
			return strings.TrimSpace(dot[s.start:s.end]) + "\n", nil
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
		return strings.TrimSpace(dot[matched[0].start:matched[0].end]) + "\n", nil
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
						spans = append(spans, dotGraphSpan{
							name:  graphName,
							start: graphStart,
							end:   i,
						})
						continue
					}
				}
			}
		}

		i++
	}

	return spans
}
