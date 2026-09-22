package cudacensus_test

import (
	"fmt"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacensus"
)

func ExampleNewDCGMReader() {
	// A connection must be supplied; construction never opens one implicitly.
	_, err := cudacensus.NewDCGMReader(0, 0, []cudacensus.Counter{cudacensus.SMActive}, time.Second)
	fmt.Println(err != nil)
	// Output: true
}

func ExampleDCGMReader_Fields() {
	// Field discovery is local. Use a real connected handle before Read.
	r, _ := cudacensus.NewDCGMReader(1, 0, []cudacensus.Counter{cudacensus.SMActive}, time.Second)
	fmt.Println(len(r.Fields()))
	// Output: 1
}

func ExampleDCGMReader_Read() {
	var r cudacensus.DCGMReader
	_, err := r.Read(make([]cudacensus.Reading, 1))
	fmt.Println(err != nil)
	// Output: true
}

func ExampleWatchDCGM() {
	// Supply a connected session and close it even if setup fails.
	_, err := cudacensus.WatchDCGM(nil, "census", 0, []cudacensus.Counter{cudacensus.SMActive}, time.Millisecond, time.Second)
	fmt.Println(err != nil)
	// Output: true
}
