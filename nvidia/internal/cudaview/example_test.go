package cudaview_test

import (
	"bytes"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/internal/cudaview"
)

func ExampleReadArtifacts() {
	data, err := cudaview.ReadArtifacts([]string{"testdata/kernel.sass"})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(data.Tiers[4].Lines[1].Opcode)
	// Output: EXIT
}

func ExampleAnnotate() {
	tier := cudaview.PipelineTier{Language: "cuda", FileName: "kernel.cu", Lines: []cudaview.PipelineLine{{Num: 1, Text: "// CUDA source"}}}
	cudaview.Annotate(&tier)
	fmt.Println(tier.Lines[0].SourceFile, tier.Lines[0].SourceLine)
	// Output: kernel.cu 1
}

func ExampleRender() {
	data := &cudaview.PipelineViewData{Title: "Kernel listing"}
	var out bytes.Buffer
	if err := cudaview.Render(&out, data); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(strings.Contains(out.String(), "<title>Kernel listing</title>"))
	// Output: true
}

func ExampleRenderDiff() {
	var out bytes.Buffer
	if err := cudaview.RenderDiff(&out, &cudaview.DiffViewData{Title: "Comparison"}); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(strings.Contains(out.String(), "<title>Comparison</title>"))
	// Output: true
}

func ExampleWriteSnapshot() {
	var out bytes.Buffer
	data := &cudaview.PipelineViewData{Tiers: []cudaview.PipelineTier{{Name: "SASS", Lines: []cudaview.PipelineLine{{Num: 1, Text: "EXIT;"}}}}}
	if err := cudaview.WriteSnapshot(&out, data); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(strings.Count(out.String(), "EXIT;"))
	// Output: 1
}
