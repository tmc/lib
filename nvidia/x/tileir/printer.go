package tileir

import (
	"bytes"
	"fmt"
	"io"
	"strings"
)

// String formats the entire Module into canonical textual Tile IR MLIR.
func (m *Module) String() string {
	var buf bytes.Buffer
	_, _ = m.WriteTo(&buf)
	return buf.String()
}

// WriteTo formats and writes the Module into w.
func (m *Module) WriteTo(w io.Writer) (int64, error) {
	var total int64

	writeStr := func(s string) error {
		n, err := io.WriteString(w, s)
		total += int64(n)
		return err
	}

	for _, c := range m.Comments {
		if err := writeStr(fmt.Sprintf("// %s\n", c)); err != nil {
			return total, err
		}
	}

	modName := m.Name
	if modName == "" {
		modName = "kernels"
	}
	if err := writeStr(fmt.Sprintf("cuda_tile.module @%s {\n", modName)); err != nil {
		return total, err
	}

	for _, fn := range m.Functions {
		kind := "func"
		if fn.Entry {
			kind = "entry"
		}

		var params []string
		for _, p := range fn.Params {
			if p.Type != "" {
				params = append(params, fmt.Sprintf("%s: %s", p.Name, p.Type))
			} else {
				params = append(params, p.Name)
			}
		}

		sig := fmt.Sprintf("  cuda_tile.%s @%s(%s)", kind, fn.Name, strings.Join(params, ", "))
		if len(fn.Returns) > 0 {
			sig += fmt.Sprintf(" -> (%s)", strings.Join(fn.Returns, ", "))
		}
		sig += " {\n"
		if err := writeStr(sig); err != nil {
			return total, err
		}

		for _, op := range fn.Operations {
			var line string
			if op.Raw != "" {
				line = "    " + op.Raw + "\n"
			} else {
				var b strings.Builder
				b.WriteString("    ")
				if len(op.Results) > 0 {
					b.WriteString(strings.Join(op.Results, ", "))
					b.WriteString(" = ")
				}
				b.WriteString(op.OpCode)
				if op.Attributes != "" {
					b.WriteString(" ")
					b.WriteString(op.Attributes)
				}
				if len(op.Operands) > 0 {
					b.WriteString(" ")
					b.WriteString(strings.Join(op.Operands, ", "))
				}
				if op.Type != "" {
					b.WriteString(" : ")
					b.WriteString(op.Type)
				}
				b.WriteString("\n")
				line = b.String()
			}
			if err := writeStr(line); err != nil {
				return total, err
			}
		}

		if err := writeStr("  }\n"); err != nil {
			return total, err
		}
	}

	if err := writeStr("}\n"); err != nil {
		return total, err
	}

	if len(m.Locations) > 0 {
		if err := writeStr("\n"); err != nil {
			return total, err
		}
		for sym, loc := range m.Locations {
			var locStr string
			if loc.Column > 0 {
				locStr = fmt.Sprintf("#%s = loc(%q:%d:%d)\n", sym, loc.File, loc.Line, loc.Column)
			} else if loc.Line > 0 {
				locStr = fmt.Sprintf("#%s = loc(%q:%d)\n", sym, loc.File, loc.Line)
			} else if loc.File != "" {
				locStr = fmt.Sprintf("#%s = loc(%q)\n", sym, loc.File)
			} else if loc.Scope != "" {
				locStr = fmt.Sprintf("#%s = loc(%s)\n", sym, loc.Scope)
			}
			if locStr != "" {
				if err := writeStr(locStr); err != nil {
					return total, err
				}
			}
		}
	}

	return total, nil
}
