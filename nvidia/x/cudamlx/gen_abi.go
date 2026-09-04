//go:build ignore

// Gen_abi generates abi.gen.go and abi_layout_gen_test.go from MLX's
// live CUDA headers, measured with clanglayout via the appledocs
// clangdump command. Set MLX_ROOT to the MLX checkout (default
// ~/ml-explore/mlx).
//
// Usage:
//
//	go run gen_abi.go
//	go run gen_abi.go -check
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
)

var checkOnly = flag.Bool("check", false, "report whether generated files are stale")

type layout struct {
	Name      string
	Size      int64
	Alignment int64
	Measured  bool
	Fields    []field
}

type field struct {
	Name              string
	Offset            int64 // bits
	Size              int64
	Measured          bool
	IsArray           bool
	ArrayElementType  string
	ArrayElementCount int64
}

func main() {
	flag.Parse()
	log.SetFlags(0)
	log.SetPrefix("gen_abi: ")

	mlxRoot := os.Getenv("MLX_ROOT")
	if mlxRoot == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			log.Fatal(err)
		}
		mlxRoot = filepath.Join(home, "ml-explore", "mlx")
	}
	config := filepath.Join(mlxRoot, "mlx", "backend", "cuda", "device", "config.h")
	utils := filepath.Join(mlxRoot, "mlx", "backend", "cuda", "device", "utils.cuh")
	if _, err := os.Stat(config); err != nil {
		log.Fatalf("MLX checkout not found: %v (set MLX_ROOT)", err)
	}
	commit, configHash, utilsHash := sourceIdentity(mlxRoot, config, utils)

	cmd := exec.Command("appledocs", "clangdump", "layout",
		"mlx_shape_abi", "mlx_strides_abi", "mlx_dim3_abi",
		"--header", "abi_probe.h", "-I", ".", "-I", mlxRoot, "--json")
	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			log.Fatalf("clangdump: %v: %s", err, ee.Stderr)
		}
		log.Fatalf("clangdump: %v", err)
	}
	var result struct {
		Layouts []layout
	}
	if err := json.Unmarshal(out, &result); err != nil {
		log.Fatalf("parse clangdump output: %v", err)
	}
	byName := make(map[string]layout)
	for _, l := range result.Layouts {
		if !l.Measured {
			log.Fatalf("record %s not measured", l.Name)
		}
		byName[l.Name] = l
	}
	shape, ok := byName["mlx_shape_abi"]
	if !ok {
		log.Fatal("mlx_shape_abi missing from measurement")
	}
	strides, ok := byName["mlx_strides_abi"]
	if !ok {
		log.Fatal("mlx_strides_abi missing from measurement")
	}
	dim3, ok := byName["mlx_dim3_abi"]
	if !ok {
		log.Fatal("mlx_dim3_abi missing from measurement")
	}
	ndim := shape.Fields[0].ArrayElementCount
	if n := strides.Fields[0].ArrayElementCount; n != ndim {
		log.Fatalf("shape ndim %d != strides ndim %d", ndim, n)
	}

	var b bytes.Buffer
	writeHeader(&b, commit, configHash, utilsHash)
	fmt.Fprintf(&b, "package cudamlx\n\n")
	fmt.Fprintf(&b, "const generatedMLXCommit = %q\n\n", commit)
	fmt.Fprintf(&b, "// MaxNDim is MLX's MAX_NDIM: the maximum dimensions of shape and\n")
	fmt.Fprintf(&b, "// strides passed as kernel parameters.\n")
	fmt.Fprintf(&b, "const MaxNDim = %d\n\n", ndim)
	fmt.Fprintf(&b, "// Shape mirrors cuda::std::array<int32_t, MAX_NDIM>, passed to MLX\n// kernels by value.\ntype Shape [MaxNDim]int32\n\n")
	fmt.Fprintf(&b, "// Strides mirrors cuda::std::array<int64_t, MAX_NDIM>, passed to MLX\n// kernels by value.\ntype Strides [MaxNDim]int64\n")
	writeFile("abi.gen.go", b.Bytes(), *checkOnly)

	b.Reset()
	writeHeader(&b, commit, configHash, utilsHash)
	fmt.Fprintf(&b, "package cudamlx\n\n")
	fmt.Fprintf(&b, "import (\n\t\"testing\"\n\t\"unsafe\"\n)\n\n")
	fmt.Fprintf(&b, "func TestGeneratedLayouts(t *testing.T) {\n")
	check := func(goType string, l layout) {
		fmt.Fprintf(&b, "\tif got := unsafe.Sizeof(%s{}); got != %d {\n", goType, l.Size)
		fmt.Fprintf(&b, "\t\tt.Errorf(\"sizeof(%s) = %%d, want %d\", got)\n\t}\n", goType, l.Size)
		fmt.Fprintf(&b, "\tif got := unsafe.Alignof(%s{}); got != %d {\n", goType, l.Alignment)
		fmt.Fprintf(&b, "\t\tt.Errorf(\"alignof(%s) = %%d, want %d\", got)\n\t}\n", goType, l.Alignment)
	}
	check("Shape", shape)
	check("Strides", strides)
	check("Dim3", dim3)
	for _, f := range dim3.Fields {
		goField := map[string]string{"x": "X", "y": "Y", "z": "Z"}[f.Name]
		fmt.Fprintf(&b, "\tif got := unsafe.Offsetof(Dim3{}.%s); got != %d {\n", goField, f.Offset/8)
		fmt.Fprintf(&b, "\t\tt.Errorf(\"offsetof(Dim3.%s) = %%d, want %d\", got)\n\t}\n", goField, f.Offset/8)
	}
	fmt.Fprintf(&b, "}\n")
	writeFile("abi_layout_gen_test.go", b.Bytes(), *checkOnly)
}

func sourceIdentity(root, config, utils string) (commit, configHash, utilsHash string) {
	cmd := exec.Command("git", "-C", root, "diff", "--quiet", "HEAD", "--",
		"mlx/backend/cuda/device/config.h", "mlx/backend/cuda/device/utils.cuh")
	if err := cmd.Run(); err != nil {
		log.Fatalf("MLX ABI headers differ from HEAD: %v", err)
	}
	cmd = exec.Command("git", "-C", root, "rev-parse", "HEAD")
	out, err := cmd.Output()
	if err != nil {
		log.Fatalf("read MLX commit: %v", err)
	}
	commit = string(bytes.TrimSpace(out))
	return commit, fileHash(config), fileHash(utils)
}

func fileHash(name string) string {
	data, err := os.ReadFile(name)
	if err != nil {
		log.Fatalf("read %s: %v", name, err)
	}
	return fmt.Sprintf("%x", sha256.Sum256(data))
}

func writeHeader(b *bytes.Buffer, commit, configHash, utilsHash string) {
	fmt.Fprintln(b, "// Code generated by gen_abi.go. DO NOT EDIT.")
	fmt.Fprintf(b, "// MLX commit: %s\n", commit)
	fmt.Fprintf(b, "// mlx/backend/cuda/device/config.h SHA-256: %s\n", configHash)
	fmt.Fprintf(b, "// mlx/backend/cuda/device/utils.cuh SHA-256: %s\n\n", utilsHash)
}

func writeFile(name string, data []byte, check bool) {
	if check {
		old, err := os.ReadFile(name)
		if err != nil {
			log.Fatalf("read generated file %s: %v", name, err)
		}
		if !bytes.Equal(old, data) {
			log.Fatalf("%s is stale; run go generate", name)
		}
		fmt.Printf("checked %s\n", name)
		return
	}
	if err := os.WriteFile(name, data, 0o666); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("wrote %s\n", name)
}
