package cudaview

import (
	_ "embed"
	"fmt"
	"html/template"
	"io"
	"strings"
)

//go:embed pipeline.html
var pipelineHTML string

//go:embed listing.js
var listingJS string

//go:embed core.js
var coreJS string

// Render writes a self-contained pipeline page.
func Render(w io.Writer, data *PipelineViewData) error {
	tmpl, err := template.New("pipeline").Funcs(template.FuncMap{
		"coreJS":    func() template.JS { return template.JS(coreJS) },
		"join":      strings.Join,
		"listingJS": func() template.JS { return template.JS(listingJS) },
	}).Parse(pipelineHTML)
	if err != nil {
		return fmt.Errorf("parse template: %w", err)
	}
	return tmpl.Execute(w, struct {
		*PipelineViewData
		Snapshot snapshot
	}{data, makeSnapshot(data)})
}
