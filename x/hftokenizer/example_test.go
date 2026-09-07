package hftokenizer_test

import (
	"fmt"
	"os"

	"github.com/tmc/lib/x/hftokenizer"
)

func exampleTokenizer() (*hftokenizer.Tokenizer, error) {
	f, err := os.Open("testdata/gpt2.json")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return hftokenizer.New(f)
}

func ExampleNew() {
	f, err := os.Open("testdata/gpt2.json")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer f.Close()
	tok, err := hftokenizer.New(f)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(tok.MaxTokenID())
	// Output: 269
}

func ExampleTokenizer_Encode() {
	tok, err := exampleTokenizer()
	if err != nil {
		fmt.Println(err)
		return
	}
	ids, err := tok.Encode("abc")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(ids)
	// Output: [260]
}

func ExampleTokenizer_Decode() {
	tok, err := exampleTokenizer()
	if err != nil {
		fmt.Println(err)
		return
	}
	text, err := tok.Decode([]int32{260})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(text)
	// Output: abc
}

func ExampleTokenizer_DecodeBytes() {
	tok, err := exampleTokenizer()
	if err != nil {
		fmt.Println(err)
		return
	}
	data, err := tok.DecodeBytes([]int32{260})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(data)
	// Output: [97 98 99]
}

func ExampleTokenizer_MaxTokenID() {
	tok, err := exampleTokenizer()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(tok.MaxTokenID())
	// Output: 269
}

func ExampleTokenizer_NewDecoder() {
	tok, err := exampleTokenizer()
	if err != nil {
		fmt.Println(err)
		return
	}
	d := tok.NewDecoder()
	first, err := d.Decode([]int32{114}, false)
	fmt.Printf("%q %v\n", first, err)
	last, err := d.Decode([]int32{18, 12}, true)
	fmt.Printf("%q %v\n", last, err)
	// Output:
	// "" <nil>
	// "€" <nil>
}

func ExampleDecoder_Decode() {
	tok, err := exampleTokenizer()
	if err != nil {
		fmt.Println(err)
		return
	}
	d := tok.NewDecoder()
	text, err := d.Decode([]int32{114, 18}, true)
	fmt.Printf("%q %v\n", text, err)
	// Output: "�" <nil>
}
