package breakout_test

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/puffernet/breakout"
)

func Example() {
	dev, err := cuda.NewDevice(0)
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Close()
	const n = 1024
	v, err := breakout.New(dev, breakout.DefaultConfig, n)
	if err != nil {
		log.Fatal(err)
	}
	defer v.Close()

	a := cudamem.NewArena(dev)
	defer a.Free()
	obs := cudamem.AllocIn[uint16](a, "obs", n*breakout.ObsSize)
	actions := cudamem.AllocIn[float32](a, "actions", n)
	rewards := cudamem.AllocIn[float32](a, "rewards", n)
	terminals := cudamem.AllocIn[float32](a, "terminals", n)
	if err := a.Err(); err != nil {
		log.Fatal(err)
	}

	if err := actions.CopyFromHost(make([]float32, n)); err != nil {
		log.Fatal(err)
	}

	stream := dev.DefaultStream()
	if err := v.Reset(stream, obs, rewards, terminals); err != nil {
		log.Fatal(err)
	}
	// Take no action (0) for 1000 steps.
	for range 1000 {
		if err := v.Step(stream, actions, obs, rewards, terminals); err != nil {
			log.Fatal(err)
		}
	}
	l, err := v.Log(context.Background(), stream)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%.0f episodes, mean score %.1f\n", l.N, l.Score/l.N)
}
