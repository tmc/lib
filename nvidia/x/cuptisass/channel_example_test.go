package cuptisass_test

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"log"
	"sync/atomic"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cuptisass"
)

func ExampleChannel_Receive() {
	// A device end has written three 8-byte records, the last the sentinel,
	// and rung the doorbell with their length.
	doorbell := new(int32)
	buf := make([]byte, 64)
	for i, cta := range []int32{0, 1, cuptisass.SentinelCTA} {
		binary.LittleEndian.PutUint32(buf[8*i:], uint32(cta))
	}
	atomic.StoreInt32(doorbell, 24)

	c, err := cuptisass.NewChannel(doorbell, buf)
	if err != nil {
		log.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	st, err := c.Receive(ctx, cuptisass.RecordLayout{Size: 8}, func(rec []byte) error {
		fmt.Println("cta", int32(binary.LittleEndian.Uint32(rec)))
		return nil
	})
	fmt.Println(st.Records, st.Terminated, err)
	// Output:
	// cta 0
	// cta 1
	// 2 true <nil>
}

func ExampleChannel_Recv() {
	doorbell := new(int32)
	buf := []byte("abcdefgh")
	atomic.StoreInt32(doorbell, 6)
	c, err := cuptisass.NewChannel(doorbell, buf)
	if err != nil {
		log.Fatal(err)
	}
	p := make([]byte, 4)
	for {
		n, err := c.Recv(p)
		if err != nil {
			log.Fatal(err)
		}
		if n == 0 {
			break
		}
		fmt.Printf("%q, %d left\n", p[:n], atomic.LoadInt32(doorbell))
	}
	// Output:
	// "abcd", 2 left
	// "ef", 0 left
}

func ExampleCheckDeviceChannel() {
	err := cuptisass.CheckDeviceChannel("sm_121a")
	fmt.Println(errors.Is(err, cudaerrors.ErrHardwarePending))
	err = cuptisass.CheckDeviceChannel("sm_30")
	fmt.Println(errors.Is(err, cudaerrors.ErrUnsupported))
	// Output:
	// true
	// true
}
