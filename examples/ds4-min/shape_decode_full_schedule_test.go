package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"slices"
	"testing"
)

func TestShapeDecodeBindingAdmissionOnce(t *testing.T) {
	fn := shapeScheduleFunction(t, "cuda_shape_decode_linux.go", "DS4_SHAPE_DECODE_SOURCE", "newCUDAShapeDecode")
	names := []string{"planShapeForward", "validateShapeAttentionWeights", "validateShapeAttentionWeightsAdmitted", "validateShapeFFNWeights", "validateShapeCompressorWeights", "validateShapeFFNWeightsAdmitted", "validateShapeCompressorWeightsAdmitted"}
	calls := shapeScheduleCalls(fn.Body, names)
	want := []string{"planShapeForward", "validateShapeAttentionWeightsAdmitted", "validateShapeFFNWeightsAdmitted", "validateShapeCompressorWeightsAdmitted", "validateShapeCompressorWeightsAdmitted"}
	if !slices.Equal(calls, want) {
		t.Fatalf("constructor admission calls %v want %v", calls, want)
	}
}

func shapeScheduleFunction(t *testing.T, path, env, name string) *ast.FuncDecl {
	t.Helper()
	if alternate := os.Getenv(env); alternate != "" {
		path = alternate
	}
	f, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, decl := range f.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == name {
			return fn
		}
	}
	t.Fatalf("schedule function %s absent", name)
	return nil
}

func shapeScheduleCalls(body ast.Node, names []string) []string {
	var calls []string
	ast.Inspect(body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		name := ""
		switch f := call.Fun.(type) {
		case *ast.Ident:
			name = f.Name
		case *ast.SelectorExpr:
			name = f.Sel.Name
		}
		if slices.Contains(names, name) {
			calls = append(calls, name)
		}
		return true
	})
	return calls
}

// These are actual source-wiring gates, not CUDA arithmetic execution. The
// schedule matches layer_forward_raw_swa_one at pinned ds4.c:14703-14866.
func TestShapeFullLayerSchedule(t *testing.T) {
	fn := shapeScheduleFunction(t, "cuda_shape_decode_linux.go", "DS4_SHAPE_DECODE_TEST_SOURCE", "Step")
	var body *ast.BlockStmt
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		loop, ok := n.(*ast.ForStmt)
		if !ok {
			return true
		}
		cond, ok := loop.Cond.(*ast.BinaryExpr)
		if !ok || cond.Op != token.LSS {
			return true
		}
		left, ok := cond.X.(*ast.Ident)
		if !ok || left.Name != "layer" {
			return true
		}
		right, ok := cond.Y.(*ast.SelectorExpr)
		if !ok || right.Sel.Name != "Layers" {
			t.Fatal("LAYER_BOUND_NOT_CONFIG")
		}
		config, ok := right.X.(*ast.Ident)
		if !ok || config.Name != "c" {
			t.Fatal("LAYER_BOUND_NOT_CONFIG")
		}
		start, ok := loop.Init.(*ast.AssignStmt)
		if !ok || len(start.Rhs) != 1 {
			t.Fatal("LAYER_START")
		}
		zero, ok := start.Rhs[0].(*ast.BasicLit)
		if !ok || zero.Value != "0" {
			t.Fatal("LAYER_START")
		}
		post, ok := loop.Post.(*ast.IncDecStmt)
		if !ok || post.Tok != token.INC {
			t.Fatal("LAYER_ADVANCE")
		}
		if body != nil {
			t.Fatal("DUPLICATE_LAYER_LOOP")
		}
		body = loop.Body
		return true
	})
	if body == nil {
		t.Fatal("FULL_LAYER_LOOP_ABSENT")
	}
	want := []string{"BeforePhase", "encodeShapeQKV", "encodeShapeRoPE", "commitShapeRaw", "encodeShapeCompressor", "encodeShapeCompressor", "encodeShapeIndexer", "encodeShapeAttentionExternal", "encodeShapeAttentionInverseRoPE", "encodeShapeAttentionOutput", "readShapeFiniteOutput", "AfterPhase", "FFNPhase"}
	if got := shapeScheduleCalls(body, want); !slices.Equal(got, want) {
		t.Fatalf("FULL_LAYER_ORDER got=%v want=%v", got, want)
	}
	var routes []string
	ast.Inspect(body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		id, ok := call.Fun.(*ast.Ident)
		if !ok || id.Name != "encodeShapeCompressor" {
			return true
		}
		if len(call.Args) != 7 {
			t.Fatal("COMPRESSOR_ARGUMENTS")
		}
		route, ok := call.Args[4].(*ast.Ident)
		if !ok {
			t.Fatal("COMPRESSOR_ROUTE")
		}
		routes = append(routes, route.Name)
		return true
	})
	if !slices.Equal(routes, []string{"false", "true"}) {
		t.Fatalf("COMPRESSOR_ROUTE_ORDER got=%v", routes)
	}
}

func TestShapeFFNRouteOrder(t *testing.T) {
	fn := shapeScheduleFunction(t, "cuda_shape_ffn_linux.go", "DS4_SHAPE_FFN_TEST_SOURCE", "encodeShapeFFN")
	want := []string{"encodeF16Matvec", "encodeQuantQ81", "encodeQ80MatvecQ81", "encodeQ80MatvecQ81", "encodeSwiGLUQuantQ80", "encodeQ80MatvecQuantized", "encodeMatvecIQ2SelectedQ81", "encodeSwiGLUQuantQ81", "encodeMatvecQ2KSelectedQ81", "encodeSumRoutedShared"}
	if got := shapeScheduleCalls(fn.Body, want); !slices.Equal(got, want) {
		t.Fatalf("FFN_ROUTE_ORDER got=%v want=%v", got, want)
	}
	invalidCheck, firstExpert := token.NoPos, token.NoPos
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if stmt, ok := n.(*ast.IfStmt); ok {
			if cond, ok := stmt.Cond.(*ast.BinaryExpr); ok && cond.Op == token.NEQ {
				if index, ok := cond.X.(*ast.IndexExpr); ok {
					if id, ok := index.X.(*ast.Ident); ok && id.Name == "invalid" {
						invalidCheck = stmt.Pos()
					}
				}
			}
		}
		if call, ok := n.(*ast.CallExpr); ok {
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "encodeMatvecIQ2SelectedQ81" {
				firstExpert = call.Pos()
			}
		}
		return true
	})
	if invalidCheck == token.NoPos || firstExpert == token.NoPos || invalidCheck >= firstExpert {
		t.Fatal("EXPERT_BEFORE_ROUTE_ADMISSION")
	}
	phase := shapeScheduleFunction(t, "cuda_shape_ffn_linux.go", "DS4_SHAPE_FFN_TEST_SOURCE", "FFNPhase")
	phaseOrder := []string{"BeforePhase", "encodeShapeFFN", "readShapeFiniteOutput", "AfterPhase"}
	if got := shapeScheduleCalls(phase.Body, phaseOrder); !slices.Equal(got, phaseOrder) {
		t.Fatalf("FFN_HC_ORDER got=%v", got)
	}
}
