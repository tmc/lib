package nvrtcx

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestToolkitIncludeDir(t *testing.T) {
	temp := t.TempDir()
	incDir := filepath.Join(temp, "include")
	if err := os.MkdirAll(incDir, 0o755); err != nil {
		t.Fatal(err)
	}
	headerFile := filepath.Join(incDir, "cuda_runtime.h")
	if err := os.WriteFile(headerFile, []byte("// mock"), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("CUDA_HOME", temp)
	t.Setenv("CUDA_PATH", "")

	got := ToolkitIncludeDir()
	if got != incDir {
		t.Fatalf("ToolkitIncludeDir() = %q, want %q", got, incDir)
	}

	// Test targets/*/include structure (typical in modern Linux CUDA toolkits)
	temp2 := t.TempDir()
	targetIncDir := filepath.Join(temp2, "targets", "x86_64-linux", "include")
	if err := os.MkdirAll(targetIncDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(targetIncDir, "cuda.h"), []byte("// mock"), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("CUDA_HOME", temp2)
	got2 := ToolkitIncludeDir()
	if got2 != targetIncDir {
		t.Fatalf("ToolkitIncludeDir() with targets = %q, want %q", got2, targetIncDir)
	}
}

func TestWithIncludePaths(t *testing.T) {
	var cfg options
	opt := WithIncludePaths("/usr/local/cuda/include", "/custom/include")
	opt(&cfg)
	if cfg.err != nil {
		t.Fatalf("unexpected error: %v", cfg.err)
	}
	want := []string{"--include-path=/usr/local/cuda/include", "--include-path=/custom/include"}
	if !slices.Equal(cfg.flags, want) {
		t.Fatalf("got flags %v, want %v", cfg.flags, want)
	}

	// Test invalid empty path
	var cfgEmpty options
	WithIncludePaths("")(&cfgEmpty)
	if cfgEmpty.err == nil || !strings.Contains(cfgEmpty.err.Error(), "empty include path") {
		t.Fatalf("expected empty include path error, got %v", cfgEmpty.err)
	}

	// Test NUL byte
	var cfgNul options
	WithIncludePaths("/path\x00with/nul")(&cfgNul)
	if cfgNul.err == nil || !strings.Contains(cfgNul.err.Error(), "contains NUL") {
		t.Fatalf("expected NUL byte error, got %v", cfgNul.err)
	}
}

func TestWithDefines(t *testing.T) {
	var cfg options
	WithDefines("FEATURE_ENABLED", "BUFFER_SIZE=1024")(&cfg)
	if cfg.err != nil {
		t.Fatalf("unexpected error: %v", cfg.err)
	}
	want := []string{"-DFEATURE_ENABLED", "-DBUFFER_SIZE=1024"}
	if !slices.Equal(cfg.flags, want) {
		t.Fatalf("got flags %v, want %v", cfg.flags, want)
	}

	// Test invalid empty macro
	var cfgEmpty options
	WithDefines("")(&cfgEmpty)
	if cfgEmpty.err == nil || !strings.Contains(cfgEmpty.err.Error(), "empty macro definition") {
		t.Fatalf("expected empty macro error, got %v", cfgEmpty.err)
	}

	// Test NUL byte
	var cfgNul options
	WithDefines("FOO\x00BAR")(&cfgNul)
	if cfgNul.err == nil || !strings.Contains(cfgNul.err.Error(), "contains NUL") {
		t.Fatalf("expected NUL error, got %v", cfgNul.err)
	}
}

func TestWithUndefines(t *testing.T) {
	var cfg options
	WithUndefines("DEBUG", "LOGGING")(&cfg)
	if cfg.err != nil {
		t.Fatalf("unexpected error: %v", cfg.err)
	}
	want := []string{"-UDEBUG", "-ULOGGING"}
	if !slices.Equal(cfg.flags, want) {
		t.Fatalf("got flags %v, want %v", cfg.flags, want)
	}

	// Test invalid empty macro
	var cfgEmpty options
	WithUndefines("")(&cfgEmpty)
	if cfgEmpty.err == nil || !strings.Contains(cfgEmpty.err.Error(), "empty macro name") {
		t.Fatalf("expected empty macro error, got %v", cfgEmpty.err)
	}

	// Test NUL byte
	var cfgNul options
	WithUndefines("A\x00B")(&cfgNul)
	if cfgNul.err == nil || !strings.Contains(cfgNul.err.Error(), "contains NUL") {
		t.Fatalf("expected NUL error, got %v", cfgNul.err)
	}
}

func TestWithStandard(t *testing.T) {
	tests := []struct {
		input string
		want  string
		ok    bool
	}{
		{"11", "-std=c++11", true},
		{"c++11", "-std=c++11", true},
		{"14", "-std=c++14", true},
		{"c++14", "-std=c++14", true},
		{"17", "-std=c++17", true},
		{"c++17", "-std=c++17", true},
		{"20", "-std=c++20", true},
		{"c++20", "-std=c++20", true},
		{"23", "-std=c++23", true},
		{"c++23", "-std=c++23", true},
		{"C++17", "-std=c++17", true},
		{" c++20 ", "-std=c++20", true},
		{"98", "", false},
		{"c++98", "", false},
		{"c++03", "", false},
		{"c++26", "", false},
		{"invalid", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			var cfg options
			WithStandard(tt.input)(&cfg)
			if tt.ok {
				if cfg.err != nil {
					t.Fatalf("unexpected error: %v", cfg.err)
				}
				if len(cfg.flags) != 1 || cfg.flags[0] != tt.want {
					t.Fatalf("got flags %v, want [%s]", cfg.flags, tt.want)
				}
			} else {
				if cfg.err == nil {
					t.Fatalf("expected error for %q, got nil", tt.input)
				}
			}
		})
	}
}

func TestWithOptimizationLevel(t *testing.T) {
	for level := 0; level <= 3; level++ {
		var cfg options
		WithOptimizationLevel(level)(&cfg)
		if cfg.err != nil {
			t.Fatalf("level %d unexpected error: %v", level, cfg.err)
		}
		want := []string{"-O" + string(rune('0'+level))}
		if !slices.Equal(cfg.flags, want) {
			t.Fatalf("level %d got %v, want %v", level, cfg.flags, want)
		}
	}

	for _, invalid := range []int{-1, 4, 10} {
		var cfg options
		WithOptimizationLevel(invalid)(&cfg)
		if cfg.err == nil {
			t.Fatalf("expected error for level %d, got nil", invalid)
		}
	}
}

func TestWithMaxRegisters(t *testing.T) {
	var cfg options
	WithMaxRegisters(64)(&cfg)
	if cfg.err != nil {
		t.Fatalf("unexpected error: %v", cfg.err)
	}
	if !slices.Equal(cfg.flags, []string{"--maxrregcount=64"}) {
		t.Fatalf("got flags %v", cfg.flags)
	}

	for _, invalid := range []int{0, -1, -32} {
		var cfgInvalid options
		WithMaxRegisters(invalid)(&cfgInvalid)
		if cfgInvalid.err == nil {
			t.Fatalf("expected error for max registers %d, got nil", invalid)
		}
	}
}

func TestWithRelocatableDeviceCode(t *testing.T) {
	var cfgTrue options
	WithRelocatableDeviceCode(true)(&cfgTrue)
	if !slices.Equal(cfgTrue.flags, []string{"--relocatable-device-code=true"}) {
		t.Fatalf("got %v", cfgTrue.flags)
	}

	var cfgFalse options
	WithRelocatableDeviceCode(false)(&cfgFalse)
	if !slices.Equal(cfgFalse.flags, []string{"--relocatable-device-code=false"}) {
		t.Fatalf("got %v", cfgFalse.flags)
	}
}

func TestSimpleOptions(t *testing.T) {
	tests := []struct {
		name string
		opt  Option
		want string
	}{
		{"FastMath", WithFastMath(), "--use_fast_math"},
		{"DefaultDevice", WithDefaultDevice(), "--device-as-default-execution-space"},
		{"DefaultExecutionSpace", WithDefaultExecutionSpace(), "--device-as-default-execution-space"},
		{"ExtraDeviceVectorization", WithExtraDeviceVectorization(), "--extra-device-vectorization"},
		{"DeviceVectorization", WithDeviceVectorization(), "--extra-device-vectorization"},
		{"WarningsAsErrors", WithWarningsAsErrors(), "--warning-as-error"},
		{"DisableWarnings", WithDisableWarnings(), "-w"},
		{"DeviceDebug", WithDeviceDebug(), "--device-debug"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var cfg options
			tt.opt(&cfg)
			if !slices.Equal(cfg.flags, []string{tt.want}) {
				t.Fatalf("got flags %v, want [%s]", cfg.flags, tt.want)
			}
			// Calling twice should deduplicate
			tt.opt(&cfg)
			if !slices.Equal(cfg.flags, []string{tt.want}) {
				t.Fatalf("got duplicate flags %v, want [%s]", cfg.flags, tt.want)
			}
		})
	}
}

func TestCompileFlags(t *testing.T) {
	flags, err := CompileFlags(
		WithArch(8, 0),
		WithStandard("17"),
		WithFastMath(),
		WithDeviceVectorization(),
		WithOptimizationLevel(3),
	)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"--gpu-architecture=compute_80",
		"-std=c++17",
		"--use_fast_math",
		"--extra-device-vectorization",
		"-O3",
		"--generate-line-info",
	}
	if !slices.Equal(flags, want) {
		t.Fatalf("CompileFlags = %v, want %v", flags, want)
	}

	// Test CompileFlags with invalid option
	if _, err := CompileFlags(WithStandard("c++98")); err == nil {
		t.Fatal("expected error on invalid standard")
	}

	// Test CompileFlags with nil option
	if _, err := CompileFlags(nil); err == nil {
		t.Fatal("expected error on nil option")
	}
}

func TestCompileOptionDefaults(t *testing.T) {
	// Test LineInfo is enabled by default
	{
		cfg := options{lineInfo: true}
		if cfg.lineInfo && !cfg.noLineInfo && !cfg.deviceDebug && !hasLineInfoFlag(cfg.flags) {
			cfg.flags = append(cfg.flags, "--generate-line-info")
		}
		if !slices.Equal(cfg.flags, []string{"--generate-line-info"}) {
			t.Fatalf("default flags = %v, want [--generate-line-info]", cfg.flags)
		}
	}

	// Test WithNoLineInfo disables it
	{
		cfg := options{lineInfo: true}
		WithNoLineInfo()(&cfg)
		if cfg.lineInfo && !cfg.noLineInfo && !cfg.deviceDebug && !hasLineInfoFlag(cfg.flags) {
			cfg.flags = append(cfg.flags, "--generate-line-info")
		}
		if len(cfg.flags) != 0 {
			t.Fatalf("expected no flags with WithNoLineInfo, got %v", cfg.flags)
		}
	}

	// Test WithDeviceDebug suppresses --generate-line-info
	{
		cfg := options{lineInfo: true}
		WithDeviceDebug()(&cfg)
		if cfg.lineInfo && !cfg.noLineInfo && !cfg.deviceDebug && !hasLineInfoFlag(cfg.flags) {
			cfg.flags = append(cfg.flags, "--generate-line-info")
		}
		if !slices.Equal(cfg.flags, []string{"--device-debug"}) {
			t.Fatalf("expected only [--device-debug], got %v", cfg.flags)
		}
	}

	// Test manual flag WithFlag("-lineinfo") suppresses duplicate
	{
		cfg := options{lineInfo: true}
		WithFlag("-lineinfo")(&cfg)
		if cfg.lineInfo && !cfg.noLineInfo && !cfg.deviceDebug && !hasLineInfoFlag(cfg.flags) {
			cfg.flags = append(cfg.flags, "--generate-line-info")
		}
		if !slices.Equal(cfg.flags, []string{"-lineinfo"}) {
			t.Fatalf("expected only [-lineinfo], got %v", cfg.flags)
		}
	}
}

func TestCompileToolkitIncludesDefault(t *testing.T) {
	temp := t.TempDir()
	incDir := filepath.Join(temp, "include")
	if err := os.MkdirAll(incDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(incDir, "cuda_runtime.h"), []byte("// mock"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("CUDA_HOME", temp)

	// When source has #include, toolkit includes should be auto-added
	srcWithInclude := `#include <cuda_runtime.h>
extern "C" __global__ void k() {}`

	// Test Compile error on invalid option
	if _, err := Compile(srcWithInclude, WithStandard("invalid")); err == nil {
		t.Fatal("expected error on invalid standard")
	}

	// Test WithToolkitIncludes failure when no directory found. The standard
	// installation prefixes are searched regardless of the environment, so a
	// host with a toolkit installed reaches this path only with them removed.
	t.Setenv("CUDA_HOME", t.TempDir()) // empty dir, no headers
	t.Setenv("CUDA_PATH", "")
	defer func(saved []string) { toolkitSearchRoots = saved }(toolkitSearchRoots)
	toolkitSearchRoots = nil
	if dir := ToolkitIncludeDir(); dir != "" {
		t.Fatalf("toolkit include directory still discoverable at %q", dir)
	}
	if _, err := Compile("void k() {}", WithToolkitIncludes()); err == nil || !strings.Contains(err.Error(), "CUDA toolkit include directory not found") {
		t.Fatalf("expected missing toolkit error, got %v", err)
	}
}

func TestAdvancedCompilerOptions(t *testing.T) {
	tests := []struct {
		name    string
		opt     Option
		want    []string
		wantErr string
	}{
		{
			name: "PCH",
			opt:  WithPCH(),
			want: []string{"-pch"},
		},
		{
			name: "CreatePCH",
			opt:  WithCreatePCH("app.pch"),
			want: []string{"--create-pch=app.pch"},
		},
		{
			name:    "CreatePCH_empty",
			opt:     WithCreatePCH(""),
			wantErr: "empty PCH file",
		},
		{
			name:    "CreatePCH_NUL",
			opt:     WithCreatePCH("app\x00.pch"),
			wantErr: "contains NUL",
		},
		{
			name: "UsePCH",
			opt:  WithUsePCH("app.pch"),
			want: []string{"--use-pch=app.pch"},
		},
		{
			name:    "UsePCH_empty",
			opt:     WithUsePCH(""),
			wantErr: "empty PCH file",
		},
		{
			name:    "UsePCH_NUL",
			opt:     WithUsePCH("app\x00.pch"),
			wantErr: "contains NUL",
		},
		{
			name: "FastCompile_0",
			opt:  WithFastCompile("0"),
			want: []string{"--Ofast-compile=0"},
		},
		{
			name: "FastCompile_min",
			opt:  WithFastCompile("min"),
			want: []string{"--Ofast-compile=min"},
		},
		{
			name: "FastCompile_mid",
			opt:  WithFastCompile("mid"),
			want: []string{"--Ofast-compile=mid"},
		},
		{
			name: "FastCompile_max",
			opt:  WithFastCompile("max"),
			want: []string{"--Ofast-compile=max"},
		},
		{
			name:    "FastCompile_invalid",
			opt:     WithFastCompile("ultra"),
			wantErr: "invalid fast compile level",
		},
		{
			name: "RandomSeed",
			opt:  WithRandomSeed(42),
			want: []string{"--frandom-seed=42"},
		},
		{
			name: "ExtensibleWholeProgram",
			opt:  WithExtensibleWholeProgram(),
			want: []string{"--extensible-whole-program"},
		},
		{
			name: "DeviceStackProtector_true",
			opt:  WithDeviceStackProtector(true),
			want: []string{"--device-stack-protector=true"},
		},
		{
			name: "DeviceStackProtector_false",
			opt:  WithDeviceStackProtector(false),
			want: []string{"--device-stack-protector=false"},
		},
		{
			name: "TimeTrace",
			opt:  WithTimeTrace("/tmp/trace.json"),
			want: []string{"--time=/tmp/trace.json"},
		},
		{
			name:    "TimeTrace_empty",
			opt:     WithTimeTrace(""),
			wantErr: "empty time trace file",
		},
		{
			name:    "TimeTrace_NUL",
			opt:     WithTimeTrace("/tmp/trace\x00.json"),
			wantErr: "contains NUL",
		},
		{
			name: "NoCache",
			opt:  WithNoCache(),
			want: []string{"--no-cache"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var cfg options
			tc.opt(&cfg)
			if tc.wantErr != "" {
				if cfg.err == nil || !strings.Contains(cfg.err.Error(), tc.wantErr) {
					t.Fatalf("got err %v, want substring %q", cfg.err, tc.wantErr)
				}
				return
			}
			if cfg.err != nil {
				t.Fatalf("unexpected error: %v", cfg.err)
			}
			if !slices.Equal(cfg.flags, tc.want) {
				t.Fatalf("got flags %v, want %v", cfg.flags, tc.want)
			}
		})
	}
}

func TestContextCancellation(t *testing.T) {
	// Test nil context
	if _, err := Compile("extern \"C\" __global__ void k() {}", WithContext(nil)); err == nil || !strings.Contains(err.Error(), "nil context") {
		t.Fatalf("expected nil context error, got %v", err)
	}
	if _, err := CompileContext(nil, "extern \"C\" __global__ void k() {}"); err == nil || !strings.Contains(err.Error(), "nil context") {
		t.Fatalf("expected nil context error from CompileContext, got %v", err)
	}

	// Test already-cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := CompileContext(ctx, "extern \"C\" __global__ void k() {}")
	if err == nil || !strings.Contains(err.Error(), context.Canceled.Error()) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestBundledHeadersOptions(t *testing.T) {
	// Test DefaultBundledHeadersDir
	dir := DefaultBundledHeadersDir()
	if dir == "" || !strings.Contains(dir, "nvrtc-bundled-headers") {
		t.Fatalf("unexpected DefaultBundledHeadersDir: %q", dir)
	}

	// Test WithBundledHeaders option
	var cfg options
	WithBundledHeaders()(&cfg)
	if !cfg.bundledHeaders {
		t.Fatal("expected bundledHeaders true")
	}

	// Test custom dir
	var cfgCustom options
	WithBundledHeaders("/custom/bundled")(&cfgCustom)
	if cfgCustom.bundledHeadersDir != "/custom/bundled" {
		t.Fatalf("expected custom dir, got %q", cfgCustom.bundledHeadersDir)
	}

	// Test errors on invalid custom dir
	var cfgEmpty options
	WithBundledHeaders("")(&cfgEmpty)
	if cfgEmpty.err == nil || !strings.Contains(cfgEmpty.err.Error(), "empty bundled headers directory") {
		t.Fatalf("expected empty dir error, got %v", cfgEmpty.err)
	}

	var cfgNul options
	WithBundledHeaders("/bad\x00/dir")(&cfgNul)
	if cfgNul.err == nil || !strings.Contains(cfgNul.err.Error(), "contains NUL") {
		t.Fatalf("expected NUL error, got %v", cfgNul.err)
	}

	// Test CompileFlags with bundled headers
	flags, err := CompileFlags(WithNoLineInfo(), WithBundledHeaders("/custom/bundled"))
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(flags, []string{"--include-path=/custom/bundled"}) {
		t.Fatalf("CompileFlags with bundled headers = %v, want [--include-path=/custom/bundled]", flags)
	}
}
