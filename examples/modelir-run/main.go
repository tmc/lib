package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"sort"
	"strings"

	"github.com/tmc/modelir"
	"github.com/tmc/modelir/encoding/sexpr"
	irtext "github.com/tmc/modelir/encoding/text"
	"github.com/tmc/modelir/interp"
)

// Config defines a ModelIR graph execution.
type Config struct {
	ModelPath   string
	Function    string
	InputsPath  string
	WeightsPath string
}

// Model is a parsed ModelIR description and, when present, its executable graph.
type Model struct {
	Program   *modelir.Program
	ModuleSet *modelir.ModuleSet
}

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	cfg, err := parseConfig(args, stderr)
	if err != nil {
		return 2
	}

	model, err := loadModel(cfg.ModelPath)
	if err != nil {
		fmt.Fprintf(stderr, "modelir-run: load model: %v\n", err)
		return 1
	}
	if err := validateModel(model); err != nil {
		fmt.Fprintf(stderr, "modelir-run: invalid model: %v\n", err)
		return 1
	}

	printSummary(stdout, model, cfg)
	if cfg.InputsPath == "" {
		if model.Program != nil && len(model.Program.Functions) > 0 {
			fmt.Fprintln(stdout, "modelir-run: no graph executed; pass -inputs to execute the reference interpreter")
		} else {
			fmt.Fprintln(stdout, "modelir-run: no executable graph declared")
		}
		return 0
	}
	if model.Program == nil || len(model.Program.Functions) == 0 {
		fmt.Fprintln(stderr, "modelir-run: model has no executable graph")
		return 1
	}

	inputs, err := readTensors(cfg.InputsPath)
	if err != nil {
		fmt.Fprintf(stderr, "modelir-run: read inputs: %v\n", err)
		return 1
	}
	weights := map[string]interp.Tensor{}
	if cfg.WeightsPath != "" {
		weights, err = readTensors(cfg.WeightsPath)
		if err != nil {
			fmt.Fprintf(stderr, "modelir-run: read weights: %v\n", err)
			return 1
		}
	}

	function := cfg.Function
	if function == "" {
		function = model.Program.Entry
	}
	outputs, err := interp.Run(model.Program, function, inputs, weights)
	if err != nil {
		fmt.Fprintf(stderr, "modelir-run: execute %q: %v\n", function, err)
		return 1
	}
	printOutputs(stdout, function, outputs)
	return 0
}

func parseConfig(args []string, stderr io.Writer) (Config, error) {
	fs := flag.NewFlagSet("modelir-run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	function := fs.String("function", "", "function to execute (default: model entry)")
	inputs := fs.String("inputs", "", "path to a JSON object of input tensors")
	weights := fs.String("weights", "", "path to a JSON object of weight tensors")

	var modelPath string
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		modelPath = args[0]
		args = args[1:]
	}
	if err := fs.Parse(args); err != nil {
		return Config{}, err
	}
	if modelPath == "" {
		if fs.NArg() != 1 {
			return Config{}, errors.New("usage: modelir-run [flags] <model.sexpr|model.modelir>")
		}
		modelPath = fs.Arg(0)
	}
	if fs.NArg() > 0 && modelPath != fs.Arg(0) {
		return Config{}, errors.New("usage: modelir-run [flags] <model.sexpr|model.modelir>")
	}
	return Config{
		ModelPath:   modelPath,
		Function:    *function,
		InputsPath:  *inputs,
		WeightsPath: *weights,
	}, nil
}

func loadModel(path string) (*Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %q: %w", path, err)
	}
	if program, err := irtext.Unmarshal(data); err == nil {
		modules, moduleErr := irtext.UnmarshalModuleSet(data)
		if program.Version == "1" || moduleErr != nil || !hasModuleData(modules) {
			modules = nil
		}
		return &Model{Program: program, ModuleSet: modules}, nil
	}
	if modules, err := sexpr.Unmarshal(data); err == nil && hasModuleData(modules) {
		return &Model{ModuleSet: modules}, nil
	}
	return nil, errors.New("not ModelIR text or an S-expression module set")
}

func hasModuleData(modules *modelir.ModuleSet) bool {
	return modules.EntryModule != "" || len(modules.Configs) != 0 || len(modules.Modules) != 0 ||
		len(modules.StandaloneFuncs) != 0 || modules.Registration != nil || modules.WeightTree != nil ||
		modules.Pooling != nil || len(modules.ClassOrigins) != 0
}

func validateModel(model *Model) error {
	if model.ModuleSet == nil {
		return nil
	}
	if diagnostics := modelir.ValidateModuleSet(model.ModuleSet); len(diagnostics) > 0 {
		return fmt.Errorf("%s", diagnostics[0])
	}
	return nil
}

func printSummary(w io.Writer, model *Model, cfg Config) {
	name := ""
	if model.ModuleSet != nil {
		name = model.ModuleSet.EntryModule
	}
	if name == "" && model.Program != nil {
		name = model.Program.Entry
	}
	fmt.Fprintf(w, "=== ModelIR Graph Runner (%s on %s/%s) ===\n", name, runtime.GOOS, runtime.GOARCH)
	fmt.Fprintf(w, "Model: %s\n", cfg.ModelPath)
	if model.Program != nil && len(model.Program.Functions) > 0 {
		function := cfg.Function
		if function == "" {
			function = model.Program.Entry
		}
		fmt.Fprintf(w, "Entry function: %s\n", function)
	}
	if model.ModuleSet != nil {
		fmt.Fprintf(w, "Modules: %d\n", len(model.ModuleSet.Modules))
		if summary := architectureSummary(model.ModuleSet.Arch); summary != "" {
			fmt.Fprintf(w, "Architecture: %s\n", summary)
		}
	}
}

func architectureSummary(arch *modelir.ArchSpec) string {
	if arch == nil {
		return ""
	}
	var fields []string
	if arch.Attention != nil && arch.Attention.QKVKind != "" {
		fields = append(fields, "attention="+arch.Attention.QKVKind)
	}
	if arch.MLP != nil && arch.MLP.Kind != "" {
		fields = append(fields, "mlp="+string(arch.MLP.Kind))
	}
	if arch.MoE != nil && arch.MoE.Routing != "" {
		fields = append(fields, "moe="+arch.MoE.Routing)
	}
	if arch.KVCache != nil && arch.KVCache.Kind != "" {
		fields = append(fields, "cache="+arch.KVCache.Kind)
	}
	if arch.Quant != nil && arch.Quant.Scheme != "" {
		fields = append(fields, "quant="+arch.Quant.Scheme)
	}
	return strings.Join(fields, ", ")
}

func readTensors(path string) (map[string]interp.Tensor, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %q: %w", path, err)
	}
	var tensors map[string]interp.Tensor
	if err := json.Unmarshal(data, &tensors); err != nil {
		return nil, fmt.Errorf("decode %q: %w", path, err)
	}
	for name, tensor := range tensors {
		if err := validTensor(tensor); err != nil {
			return nil, fmt.Errorf("tensor %q: %w", name, err)
		}
	}
	return tensors, nil
}

func validTensor(tensor interp.Tensor) error {
	size := 1
	for _, dim := range tensor.Shape {
		if dim < 0 {
			return errors.New("negative dimension")
		}
		size *= dim
	}
	if len(tensor.Data) != size {
		return fmt.Errorf("shape has %d values, data has %d", size, len(tensor.Data))
	}
	return nil
}

func printOutputs(w io.Writer, function string, outputs map[string]interp.Tensor) {
	fmt.Fprintf(w, "Executed %s.\n", function)
	names := make([]string, 0, len(outputs))
	for name := range outputs {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		tensor := outputs[name]
		fmt.Fprintf(w, "%s: shape=%v data=%v\n", name, tensor.Shape, tensor.Data)
	}
}
