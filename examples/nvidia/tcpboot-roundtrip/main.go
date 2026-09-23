// Tcpboot-roundtrip demonstrates a two-rank broadcast and barrier.
package main

import (
	"fmt"
	"log"
	"net"
	"sync"

	"github.com/tmc/lib/nvidia/x/nvlshm/tcpboot"
)

func main() {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		log.Fatal(err)
	}
	addr := ln.Addr().String()
	ln.Close()

	var wg sync.WaitGroup
	errs := make(chan error, 2)
	for rank := range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			b, err := tcpboot.New(rank, 2, addr)
			if err != nil {
				errs <- err
				return
			}
			defer b.Close()
			value, err := b.BroadcastString("ready", 0)
			if err == nil && value != "ready" {
				err = fmt.Errorf("broadcast = %q, want ready", value)
			}
			if err == nil {
				err = b.Barrier()
			}
			errs <- err
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			log.Fatal(err)
		}
	}
	fmt.Println("two ranks completed broadcast and barrier")
}
