package cudaview

import (
	_ "embed"
	"fmt"
	"html/template"
	"io"
	"math"
)

//go:embed diff.html
var diffHTML string

// RenderDiff writes an offline comparison page.
func RenderDiff(w io.Writer, data *DiffViewData) error {
	tmpl, err := template.New("diff").Funcs(template.FuncMap{
		"coreJS": func() template.JS { return template.JS(coreJS) },
		"stallWidth": func(pct float64) string {
			if pct <= 0 {
				return "0"
			}
			w := pct
			if w < 4 {
				w = 4
			}
			if w > 100 {
				w = 100
			}
			return fmt.Sprintf("%.1f", w)
		},
		"deltaWidth": func(delta float64) string {
			w := math.Abs(delta)
			if w <= 0 {
				return "0"
			}
			if w < 4 {
				w = 4
			}
			if w > 100 {
				w = 100
			}
			return fmt.Sprintf("%.1f", w)
		},
		"abs": func(v float64) float64 {
			return math.Abs(v)
		},
	}).Parse(diffHTML)
	if err != nil {
		return fmt.Errorf("parse diff template: %w", err)
	}
	return tmpl.Execute(w, data)
}
