package cudafuse_test

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudair"
)

func Example() {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{256}, cudair.Float32)
	y := g.AddInput("y", []int{256}, cudair.Float32)
	sum := g.AddBinary("add", x, y)
	_ = g.AddUnary("relu", sum)

	src, err := cudafuse.Fuse(g, "fused_add_relu")
	if err != nil {
		fmt.Printf("err: %v\n", err)
		return
	}
	fmt.Println(strings.Contains(src, "fused_add_relu"))
	fmt.Println(strings.Contains(src, "v_x + v_y"))
	// Output:
	// true
	// true
}
