package main

import (
	"bytes"
	"context"
	"go/ast"
	goformat "go/format"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestGeneratedConstructorCleanupOwnership(t *testing.T) {
	cfg := &config{typeName: "fixture", pkg: "fixture", output: "fixture.go", file: "fixture.cu"}
	body, _, err := generate(cfg, &module{kernels: []*kernel{{name: "kernel", method: "kernel"}}})
	if err != nil {
		t.Fatal(err)
	}
	fs := token.NewFileSet()
	f, err := parser.ParseFile(fs, "fixture.go", body, 0)
	if err != nil {
		t.Fatal(err)
	}
	var b bytes.Buffer
	b.WriteString("package fixture\nimport(\"errors\";\"fmt\";\"sync\")\n")
	for _, d := range f.Decls {
		keep := false
		switch x := d.(type) {
		case *ast.FuncDecl:
			keep = x.Name.Name == "newFixture" || x.Name.Name == "fixtureCleanup" || x.Recv != nil && strings.Contains(string(body[x.Pos()-1:x.End()-1]), "fixtureCleanupError")
		case *ast.GenDecl:
			for _, s := range x.Specs {
				if x, ok := s.(*ast.TypeSpec); ok && x.Name.Name == "fixtureCleanupError" {
					keep = true
				}
			}
		}
		if keep {
			if err := goformat.Node(&b, fs, d); err != nil {
				t.Fatal(err)
			}
			b.WriteByte('\n')
		}
	}
	src := strings.NewReplacer("cuda.Device", "device", "nvrtcx.CompiledModule", "compilation", "cuda.OwnModule", "ownModule").Replace(b.String())
	dir := t.TempDir()
	for n, s := range map[string]string{"go.mod": "module fixture\ngo 1.23\n", "fixture.go": src, "fixture_test.go": constructorCleanupControls} {
		if err := os.WriteFile(filepath.Join(dir, n), []byte(s), 0600); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "go", "test", "-race", "-count=3", ".")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOWORK=off")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("emitted constructor controls: %v\n%s", err, out)
	}
}

const constructorCleanupControls = `package fixture
import("errors";"io";"sync";"testing")
var primary=errors.New("primary");var cleanup=errors.New("cleanup")
type rawModule struct{close func()error};func(r rawModule)Close()error{return r.close()}
type device struct{restore error};func(d *device)WithContext(f func()error)error{if err:=f();err!=nil{return err};return d.restore}
type compilation struct{raw rawModule;err error};func(c *compilation)Load(*device)(rawModule,error){return c.raw,c.err};func(*compilation)LoweredName(s string)(string,error){return s,nil}
type moduleOwner struct{close func()error;err error};func(m *moduleOwner)Close()error{return m.close()};func(m *moduleOwner)Function(string)(*int,error){x:=1;return &x,m.err}
type fixture struct{dev *device;mod *moduleOwner;raw rawModule;kernels[1]*int}
var fixtureKernels=[]struct{name string;cxx bool}{{name:"kernel"}}
var owned *moduleOwner;var ownErr error
func ownModule(*device,rawModule)(*moduleOwner,error){return owned,ownErr}
func TestBranches(t *testing.T){for _,branch:=range []string{"raw","owned"}{for _,fail:=range []bool{false,true}{t.Run(branch+map[bool]string{false:"_closed",true:"_retry"}[fail],func(t *testing.T){calls:=0;close:=func()error{calls++;if fail&&calls<=2{return cleanup};return nil};owned=&moduleOwner{close:close,err:primary};ownErr=nil;if branch=="raw"{ownErr=primary};m,err:=newFixture(&device{},&compilation{raw:rawModule{close}});if m!=nil||!errors.Is(err,primary)||calls!=1{t.Fatal("initial owner/error")};var c io.Closer;if errors.As(err,&c)!=fail{t.Fatal("retry owner mismatch")};if fail{if !errors.Is(err,cleanup){t.Fatal("cleanup error lost")};if !errors.Is(c.Close(),cleanup){t.Fatal("retry error lost")};if c.Close()!=nil||c.Close()!=nil||calls!=3{t.Fatal("retry not idempotent")}}})}}}
func TestLoadErrorOwner(t *testing.T){calls:=0;e:=fixtureCleanup(primary,func()error{calls++;if calls==1{return cleanup};return nil});_,err:=newFixture(&device{},&compilation{err:e});var c io.Closer;if !errors.As(err,&c)||c.Close()!=nil||calls!=2{t.Fatal("load cleanup owner lost")}}
func TestSuccessfulConstructor(t *testing.T){owned=&moduleOwner{close:func()error{return nil}};ownErr=nil;m,err:=newFixture(&device{},&compilation{});if err!=nil||m==nil||m.mod!=owned||m.kernels[0]==nil{t.Fatal("successful constructor changed")}}
func TestRestorationFailureDoesNotUnloadTwice(t *testing.T){calls:=0;ownErr=primary;_,err:=newFixture(&device{restore:cleanup},&compilation{raw:rawModule{func()error{calls++;return nil}}});var c io.Closer;if !errors.Is(err,primary)||!errors.Is(err,cleanup)||!errors.As(err,&c){t.Fatal("restore error/owner lost")};if c.Close()!=nil||c.Close()!=nil||calls!=1{t.Fatal("module unloaded twice after restore failure")}}
func TestConcurrentRetry(t *testing.T){calls:=0;e:=fixtureCleanup(primary,func()error{calls++;if calls==1{return cleanup};return nil});var c io.Closer;if !errors.As(e,&c){t.Fatal("owner missing")};var wg sync.WaitGroup;for i:=0;i<20;i++{wg.Add(1);go func(){defer wg.Done();if c.Close()!=nil{t.Error("retry")}}()};wg.Wait();if calls!=2{t.Fatal("repeated release")}}
`
