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
