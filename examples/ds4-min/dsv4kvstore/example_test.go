package dsv4kvstore

import (
	"bytes"
	"encoding/binary"
	"fmt"
)

func ExampleDefaultOptions() {
	o := DefaultOptions()
	fmt.Println(o.MinTokens, o.ColdMaxTokens, o.ContinuedInterval)
	// Output: 512 30000 10000
}

func ExampleStoreLen() {
	fmt.Println(StoreLen(2080, Options{}))
	// Output: 2048
}

func ExampleContinuedStoreTarget() {
	o := DefaultOptions()
	fmt.Println(ContinuedStoreTarget(10240, 0, o))
	// Output: 10240
}

func ExampleChatAnchor() {
	tokens := []int32{1, 2, 7, 3, 7, 8}
	fmt.Println(ChatAnchor(tokens, 7, 8, Options{MinTokens: 3}))
	// Output: 4
}

func ExampleInspectUpstream() {
	var b bytes.Buffer
	header := make([]byte, 48)
	copy(header, []byte("KVC"))
	header[3] = 1
	header[4] = QuantQ2
	header[20] = 2
	binary.LittleEndian.PutUint32(header[8:12], 2048)
	binary.LittleEndian.PutUint32(header[16:20], 4096)
	binary.LittleEndian.PutUint64(header[40:48], 2)
	b.Write(header)
	var textLen [4]byte
	binary.LittleEndian.PutUint32(textLen[:], 1)
	b.Write(textLen[:])
	b.WriteString("x")
	b.WriteString("yz")
	meta, err := InspectUpstream(bytes.NewReader(b.Bytes()), UpstreamInspectOptions{})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(meta.QuantBits, meta.Tokens, meta.TextBytes, meta.PayloadBytes)
	// Output: 2 2048 1 2
}
