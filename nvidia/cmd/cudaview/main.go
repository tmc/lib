// Cudaview opens retained CUDA source and assembly artifacts without a capture.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"

	"github.com/tmc/lib/nvidia/internal/cudaview"
)

func main() {
	log.SetFlags(0)
	if err := run(os.Args[1:], os.Stdout); err != nil {
		log.Fatal(err)
	}
}

func run(args []string, w io.Writer) error {
	flags := flag.NewFlagSet("cudaview", flag.ContinueOnError)
	html := flags.String("html", "", "write an offline HTML file")
	addr := flags.String("http", "127.0.0.1:8088", "HTTP listen address")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() == 0 {
		return errors.New("usage: cudaview [-html=file] [-http=addr] source.go kernel.cu kernel.ptx kernel.sass")
	}
	data, err := cudaview.ReadArtifacts(flags.Args())
	if err != nil {
		return err
	}
	if *html != "" {
		file, err := os.Create(*html)
		if err != nil {
			return err
		}
		renderErr := cudaview.Render(file, data)
		closeErr := file.Close()
		return errors.Join(renderErr, closeErr)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		if err := cudaview.Render(w, data); err != nil {
			log.Printf("render: %v", err)
		}
	})
	fmt.Fprintf(w, "Viewer: http://%s\n", *addr)
	return http.ListenAndServe(*addr, mux)
}
