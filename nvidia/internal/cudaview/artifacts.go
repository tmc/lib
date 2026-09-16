package cudaview

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ReadArtifacts opens source and assembly files without invoking external tools.
func ReadArtifacts(paths []string) (*PipelineViewData, error) {
	data := &PipelineViewData{Title: "CUDA artifacts", KernelTiers: make(map[string][]PipelineTier)}
	data.Tiers = []PipelineTier{{Name: "Go", Language: "go"}, {Name: "CUDA C++", Language: "cuda"}, {Name: "Tile IR", Language: "mlir"}, {Name: "PTX", Language: "ptx"}, {Name: "SASS", Language: "sass"}}
	seen := make(map[int]bool)
	for _, path := range paths {
		index := -1
		switch strings.ToLower(filepath.Ext(path)) {
		case ".go":
			index = 0
		case ".cu", ".cpp":
			index = 1
		case ".tileir", ".mlir":
			index = 2
		case ".ptx":
			index = 3
		case ".sass":
			index = 4
		}
		if index < 0 {
			return nil, fmt.Errorf("unsupported artifact %q", path)
		}
		if seen[index] {
			return nil, fmt.Errorf("multiple %s artifacts; open them separately", data.Tiers[index].Name)
		}
		seen[index] = true
		content, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read artifact: %w", err)
		}
		tier := &data.Tiers[index]
		tier.FileName = path
		for i, text := range strings.Split(string(content), "\n") {
			tier.Lines = append(tier.Lines, PipelineLine{Num: i + 1, Text: text})
		}
	}
	for i := range data.Tiers {
		Annotate(&data.Tiers[i])
	}
	return data, nil
}
