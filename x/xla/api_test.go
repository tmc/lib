package xla_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/openxla"
	"github.com/tmc/lib/x/xla"
)

func ExampleNewEngineFromAPI() {
	_, err := xla.NewEngineFromAPI(nil)
	fmt.Println(err != nil)
	// Output: true
}

func TestNativeBorrowedAPI(t *testing.T) {
	path := os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if path == "" {
		t.Skip("set XLA_PJRT_CPU_PLUGIN to a CPU PJRT library")
	}
	plugin, err := openxla.OpenPlugin(path)
	if err != nil {
		t.Fatal(err)
	}
	// Closing one client must leave the borrowed API usable by a new client.
	for range 2 {
		engine, err := xla.NewEngineFromAPI(plugin.Api())
		if err != nil {
			t.Fatal(err)
		}
		run := func() error {
			p, err := engine.Compile([]byte(`module { func.func @main() -> tensor<f32> {
    %v = stablehlo.constant dense<7.0> : tensor<f32>
    return %v : tensor<f32>
   } }`), "mlir", nil)
			if err != nil {
				return err
			}
			defer func() {
				if err := p.Close(); err != nil {
					t.Error(err)
				}
			}()
			outputs, err := p.Execute(context.Background(), nil)
			if err != nil {
				return err
			}
			defer func() {
				for _, b := range outputs {
					if err := b.Close(); err != nil {
						t.Error(err)
					}
				}
			}()
			if len(outputs) != 1 {
				return fmt.Errorf("got %d outputs", len(outputs))
			}
			var host bytes.Buffer
			if _, err := outputs[0].WriteTo(&host); err != nil {
				return err
			}
			if host.Len() != 4 || math.Float32frombits(binary.LittleEndian.Uint32(host.Bytes())) != 7 {
				return fmt.Errorf("unexpected scalar %x", host.Bytes())
			}
			return nil
		}
		err = run()
		closeErr := engine.Close()
		if err != nil {
			t.Fatal(err)
		}
		if closeErr != nil {
			t.Fatal(closeErr)
		}
		if _, err := engine.Devices(); err == nil {
			t.Fatal("closed client remains usable")
		}
	}
}
