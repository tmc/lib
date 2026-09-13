package main

import (
	"flag"
	"fmt"
	"log"
	"os"
)

func main() {
	log.SetFlags(0)

	formatFlag := flag.String("format", "text", "output format: text, markdown, json")
	smsFlag := flag.Int("sms", 132, "number of GPU streaming multiprocessors (SMs)")
	tilesFlag := flag.Int("tiles", 4, "number of tiles per CTA")
	ptxFlag := flag.Bool("ptx", false, "output generated PTX assembly and exit")
	auditFlag := flag.Bool("audit", false, "audit generated PTX opcodes with x/cudaptx and exit")
	flag.Parse()

	ptxCfg := DefaultPTXConfig()
	ptxText := ModulePTX(ptxCfg)

	if *ptxFlag {
		fmt.Print(ptxText)
		return
	}

	if *auditFlag {
		m, err := AuditModule(ptxText)
		if err != nil {
			log.Fatalf("audit failed: %v", err)
		}
		fmt.Printf("PTX audit PASSED: %d kernels, target=%s\n", len(m.Kernels), m.Target)
		for _, k := range m.Kernels {
			fmt.Printf(" - %s: %d instructions\n", k.Name, len(k.Instructions))
		}
		return
	}

	suite := DefaultBenchmarkSuite()
	if *smsFlag > 0 {
		suite.Hardware.SMCount = *smsFlag
	}
	if *tilesFlag > 0 {
		suite.Workload.NumTiles = *tilesFlag
	}

	report, err := suite.Run()
	if err != nil {
		log.Fatalf("benchmark suite failed: %v", err)
	}

	switch *formatFlag {
	case "markdown", "md":
		fmt.Print(report.Markdown())
	case "json":
		data, err := report.JSON()
		if err != nil {
			log.Fatalf("failed to marshal JSON: %v", err)
		}
		os.Stdout.Write(data)
		fmt.Println()
	default:
		fmt.Print(report.Table())
	}
}
