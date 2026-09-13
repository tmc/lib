// Command genptxtables generates the PTX identifier, category, type, and
// directive tables used by cudaptx from a local mirror of the PTX ISA documentation.
//
// The docs live in a different module, so they cannot be a build input: this
// command is run by hand or via go generate, and its output is committed.
//
//	go run ./internal/genptxtables -docs ../../../../libgen/docs/ptx -out tables.gen.go
package main

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"go/format"
	"io"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

func main() {
	docs := flag.String("docs", "", "path to a docs/ptx mirror")
	out := flag.String("out", "tables.gen.go", "output file")
	docout := flag.String("docout", "ptxdoc.jsonl.gz", "path to write compressed JSONL documentation database")
	flag.Parse()
	if *docs == "" {
		log.Fatal("genptxtables: -docs is required")
	}

	version, digest, err := source(*docs)
	if err != nil {
		log.Fatalf("genptxtables: %v", err)
	}

	// 1. Directives (Table 1 of 04-syntax.md + Chapter 11)
	directives, err := extractDirectives(filepath.Join(*docs, "04-syntax.md"), filepath.Join(*docs, "11-directives.md"))
	if err != nil {
		log.Fatalf("genptxtables: directives: %v", err)
	}

	// 2. Special Registers (Table 3 of 04-syntax.md + Chapter 10)
	specialRegs, err := extractSpecialRegisters(filepath.Join(*docs, "04-syntax.md"), filepath.Join(*docs, "10-special-registers.md"))
	if err != nil {
		log.Fatalf("genptxtables: special registers: %v", err)
	}

	// 3. Data Types (Table 8 & 9 + Section 5.2 of 05-state-spaces-types-and-variables.md)
	types, err := extractDataTypes(filepath.Join(*docs, "05-state-spaces-types-and-variables.md"))
	if err != nil {
		log.Fatalf("genptxtables: data types: %v", err)
	}

	// 4. Instructions & Categories (09-instruction-set.md + Table 2 of 04-syntax.md)
	instructions, err := extractInstructions(filepath.Join(*docs, "04-syntax.md"), filepath.Join(*docs, "09-instruction-set.md"))
	if err != nil {
		log.Fatalf("genptxtables: instructions: %v", err)
	}

	src, err := generate(version, digest, directives, specialRegs, types, instructions)
	if err != nil {
		log.Fatalf("genptxtables: generate: %v", err)
	}

	if err := os.WriteFile(*out, src, 0o644); err != nil {
		log.Fatalf("genptxtables: write %s: %v", *out, err)
	}
	fmt.Printf("%s: %d keywords, %d types, %d directives, %d special registers from PTX ISA %s\n",
		*out, len(instructions), len(types), len(directives), len(specialRegs), version)

	if *docout != "" {
		docData, docCount, err := generateDocs(*docs, *docout, types)
		if err != nil {
			log.Fatalf("genptxtables: generate docs: %v", err)
		}
		if err := os.WriteFile(*docout, docData, 0o644); err != nil {
			log.Fatalf("genptxtables: write %s: %v", *docout, err)
		}
		fmt.Printf("%s: %d documentation records (%d bytes)\n", *docout, docCount, len(docData))
	}
}

func source(docs string) (version, digest string, err error) {
	path := filepath.Join(docs, "source.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return "", "", err
	}
	var meta struct {
		Version string `json:"version"`
		SHA256  string `json:"sha256"`
	}
	if err := json.Unmarshal(data, &meta); err != nil {
		return "", "", fmt.Errorf("%s: %w", path, err)
	}
	if meta.Version == "" {
		return "", "", fmt.Errorf("%s: no version", path)
	}
	sum := sha256.Sum256(data)
	if meta.SHA256 != "" {
		return meta.Version, meta.SHA256, nil
	}
	return meta.Version, fmt.Sprintf("%x", sum), nil
}

func extractDirectives(syntaxPath, ch11Path string) ([]string, error) {
	synData, err := os.ReadFile(syntaxPath)
	if err != nil {
		return nil, err
	}
	synText := string(synData)
	end := strings.Index(synText, "Table 1 PTX Directives")
	if end < 0 {
		return nil, fmt.Errorf("%s: Table 1 caption missing", syntaxPath)
	}
	start := end - 2000
	if start < 0 {
		start = 0
	}
	re := regexp.MustCompile("`(\\.[a-z0-9_]+)`")
	seen := make(map[string]bool)
	for _, m := range re.FindAllStringSubmatch(synText[start:end], -1) {
		seen[m[1]] = true
	}

	ch11Data, err := os.ReadFile(ch11Path)
	if err != nil {
		return nil, err
	}
	reCh11 := regexp.MustCompile(`### 11\.\d+\.\d+\.\s+[^:]+:\s*` + "`" + `(\.[a-z0-9_]+)` + "`")
	for _, m := range reCh11.FindAllStringSubmatch(string(ch11Data), -1) {
		seen[m[1]] = true
	}

	var list []string
	for d := range seen {
		list = append(list, d)
	}
	sort.Strings(list)
	return list, nil
}

func extractSpecialRegisters(syntaxPath, ch10Path string) ([]string, error) {
	synData, err := os.ReadFile(syntaxPath)
	if err != nil {
		return nil, err
	}
	synText := string(synData)
	end := strings.Index(synText, "Table 3 Predefined Identifiers")
	if end < 0 {
		return nil, fmt.Errorf("%s: Table 3 caption missing", syntaxPath)
	}
	start := end - 2000
	if start < 0 {
		start = 0
	}
	re := regexp.MustCompile("`(%[a-z0-9_<>]+|WARP_SZ)`")
	seen := make(map[string]bool)
	for _, m := range re.FindAllStringSubmatch(synText[start:end], -1) {
		seen[m[1]] = true
	}

	ch10Data, err := os.ReadFile(ch10Path)
	if err != nil {
		return nil, err
	}
	reCh10 := regexp.MustCompile(`## 10\.\d+\.\s+Special Registers:\s*([^\n]+)`)
	for _, m := range reCh10.FindAllStringSubmatch(string(ch10Data), -1) {
		for _, r := range re.FindAllStringSubmatch(m[1], -1) {
			seen[r[1]] = true
		}
	}

	// Expand standard hardware indices
	for i := 0; i < 8; i++ {
		seen[fmt.Sprintf("%%pm%d", i)] = true
		seen[fmt.Sprintf("%%pm%d_64", i)] = true
	}
	for i := 0; i < 2; i++ {
		seen[fmt.Sprintf("%%reserved_smem_offset_%d", i)] = true
	}

	var list []string
	for r := range seen {
		list = append(list, r)
	}
	sort.Strings(list)
	return list, nil
}

type typeInfo struct {
	Name      string
	BitSize   int
	Packed    bool
	ElemCount int
}

func extractDataTypes(ch5Path string) ([]typeInfo, error) {
	data, err := os.ReadFile(ch5Path)
	if err != nil {
		return nil, err
	}
	text := string(data)

	seen := make(map[string]typeInfo)

	add := func(name string, bitSize int, packed bool, elemCount int) {
		name = strings.TrimPrefix(name, ".")
		seen[name] = typeInfo{Name: name, BitSize: bitSize, Packed: packed, ElemCount: elemCount}
	}

	// Fundamental integer & float & bit
	add("s8", 8, false, 1)
	add("s16", 16, false, 1)
	add("s32", 32, false, 1)
	add("s64", 64, false, 1)
	add("u8", 8, false, 1)
	add("u16", 16, false, 1)
	add("u32", 32, false, 1)
	add("u64", 64, false, 1)
	add("f16", 16, false, 1)
	add("f32", 32, false, 1)
	add("f64", 64, false, 1)
	add("b8", 8, false, 1)
	add("b16", 16, false, 1)
	add("b32", 32, false, 1)
	add("b64", 64, false, 1)
	add("b128", 128, false, 1)
	add("pred", 1, false, 1)

	// Alternate scalar formats (Section 5.2.3 & 5.2.4)
	add("bf16", 16, false, 1)
	add("tf32", 32, false, 1)
	add("e4m3", 8, false, 1)
	add("e5m2", 8, false, 1)
	add("e2m1", 4, false, 1)
	add("e2m3", 6, false, 1)
	add("e3m2", 6, false, 1)
	add("ue8m0", 8, false, 1)
	add("ue4m3", 8, false, 1) // 7-bit padded in 8-bit
	add("ue5m3", 8, false, 1)
	add("s2f6", 8, false, 1)

	// Packed types (Table 9 & Section 5.2.5.2)
	add("f16x2", 32, true, 2)
	add("f32x2", 64, true, 2)
	add("bf16x2", 32, true, 2)
	add("e4m3x2", 16, true, 2)
	add("e5m2x2", 16, true, 2)
	add("e2m3x2", 16, true, 2)
	add("e3m2x2", 16, true, 2)
	add("ue8m0x2", 16, true, 2)
	add("ue4m3x2", 16, true, 2)
	add("ue5m3x2", 16, true, 2)
	add("s2f6x2", 16, true, 2)
	add("e2m1x2", 8, true, 2)

	add("e4m3x4", 32, true, 4)
	add("e5m2x4", 32, true, 4)
	add("e2m3x4", 32, true, 4)
	add("e3m2x4", 32, true, 4)
	add("e2m1p4x4", 32, true, 4)
	add("ue8m0x4", 32, true, 4)
	add("e2m1x4", 16, true, 4)

	add("u16x2", 32, true, 2)
	add("s16x2", 32, true, 2)
	add("u8x4", 32, true, 4)
	add("s8x4", 32, true, 4)

	// Verify all types in text are present
	reT9 := regexp.MustCompile("`\\.(f\\d+x2|bf16x2|e\\d+m\\d+.*|ue\\d+m\\d+.*|s\\d+f\\d+.*)`")
	for _, m := range reT9.FindAllStringSubmatch(text, -1) {
		name := strings.TrimPrefix(m[1], ".")
		if idx := strings.IndexAny(name, " `|\n\t"); idx >= 0 {
			name = name[:idx]
		}
		if _, ok := seen[name]; !ok {
			log.Printf("genptxtables: note newly discovered type: %s", name)
		}
	}

	var list []typeInfo
	for _, t := range seen {
		list = append(list, t)
	}
	sort.Slice(list, func(i, j int) bool { return list[i].Name < list[j].Name })
	return list, nil
}

type instInfo struct {
	Mnemonic  string
	Category  string
	CatEnum   string
	MinPTX    string
	MinTarget string
}

func categoryToEnum(cat string) string {
	lower := strings.ToLower(cat)
	switch {
	case strings.Contains(lower, "extended-precision"):
		return "CatExtendedPrecision"
	case strings.Contains(lower, "integer arithmetic"):
		return "CatInteger"
	case strings.Contains(lower, "floating-point"):
		return "CatFloatingPoint"
	case strings.Contains(lower, "comparison"):
		return "CatComparison"
	case strings.Contains(lower, "logic"):
		return "CatLogicShift"
	case strings.Contains(lower, "data movement"):
		return "CatDataMovement"
	case strings.Contains(lower, "fabric"):
		return "CatFabric"
	case strings.Contains(lower, "texture"):
		return "CatTexture"
	case strings.Contains(lower, "surface"):
		return "CatSurface"
	case strings.Contains(lower, "control flow"):
		return "CatControlFlow"
	case strings.Contains(lower, "parallel synchronization"):
		return "CatParallelSync"
	case strings.Contains(lower, "warp level matrix"):
		return "CatMatrixWMMA"
	case strings.Contains(lower, "warpgroup level matrix"):
		return "CatMatrixWGMMA"
	case strings.Contains(lower, "tensorcore 5th"):
		return "CatTensorCore5thGen"
	case strings.Contains(lower, "stack"):
		return "CatStack"
	case strings.Contains(lower, "video"):
		return "CatVideo"
	case strings.Contains(lower, "miscellaneous"):
		return "CatMiscellaneous"
	default:
		return "CatUnknown"
	}
}

func extractInstructions(syntaxPath, ch9Path string) ([]instInfo, error) {
	// 1. Table 2 from 04-syntax.md
	synData, err := os.ReadFile(syntaxPath)
	if err != nil {
		return nil, err
	}
	synLines := strings.Split(string(synData), "\n")
	endT2 := -1
	for i, l := range synLines {
		if strings.Contains(l, "Table 2 Reserved Instruction Keywords") {
			endT2 = i
			break
		}
	}
	if endT2 < 0 {
		return nil, fmt.Errorf("%s: Table 2 caption missing", syntaxPath)
	}
	startT2 := endT2
	for startT2 > 0 && (strings.HasPrefix(synLines[startT2-1], "|") || strings.TrimSpace(synLines[startT2-1]) == "") {
		startT2--
	}
	cell := regexp.MustCompile("`([a-z][a-z0-9_]*)`")
	t2Seen := make(map[string]bool)
	for _, l := range synLines[startT2:endT2] {
		if !strings.HasPrefix(l, "|") || strings.Contains(l, "---") {
			continue
		}
		for _, m := range cell.FindAllStringSubmatch(l, -1) {
			t2Seen[m[1]] = true
		}
	}

	// 2. Chapter 9 instruction set
	ch9Data, err := os.ReadFile(ch9Path)
	if err != nil {
		return nil, err
	}
	ch9Text := string(ch9Data)

	catMatches := regexp.MustCompile(`### (9\.7\.\d+)\.\s+([^\n]+)`).FindAllStringSubmatch(ch9Text, -1)
	catMap := make(map[string]string)
	for _, m := range catMatches {
		catMap[m[1]] = strings.TrimSpace(m[2])
	}

	headerPattern := regexp.MustCompile(`(?m)^(#{3,6})\s+(9\.7\.\d+(?:\.\d+)*)\.?\s+([^\n]+)$`)
	headers := headerPattern.FindAllStringSubmatchIndex(ch9Text, -1)
	mnemMap := make(map[string]*instInfo)

	for i, idxs := range headers {
		secNum := ch9Text[idxs[4]:idxs[5]]
		title := ch9Text[idxs[6]:idxs[7]]

		// Find 9.7.X prefix
		mPrefix := regexp.MustCompile(`(9\.7\.\d+)`).FindString(secNum)
		if mPrefix == "" {
			continue
		}
		catName := catMap[mPrefix]
		if catName == "" {
			catName = title
		}

		tokens := regexp.MustCompile("`([a-zA-Z0-9_\\.]+)`").FindAllStringSubmatch(title, -1)
		if len(tokens) == 0 {
			continue
		}

		startPos := idxs[1]
		endPos := len(ch9Text)
		if i+1 < len(headers) {
			endPos = headers[i+1][0]
		}
		body := ch9Text[startPos:endPos]

		ptxVerM := regexp.MustCompile(`(?i)Introduced in PTX ISA version\s+(\d+\.\d+)`).FindStringSubmatch(body)
		ptxVer := ""
		if len(ptxVerM) >= 2 {
			ptxVer = ptxVerM[1]
		}

		targetSM := ""
		if idx := strings.Index(body, "Target ISA Notes"); idx >= 0 {
			notePart := body[idx:]
			if exIdx := strings.Index(notePart, "Examples"); exIdx >= 0 {
				notePart = notePart[:exIdx]
			}
			if strings.Contains(strings.ToLower(notePart), "all target") {
				targetSM = "all"
			} else {
				smMatch := regexp.MustCompile(`sm_\d+[a-z]?`).FindString(notePart)
				if smMatch != "" {
					targetSM = smMatch
				}
			}
		}

		for _, tok := range tokens {
			base := strings.ToLower(strings.Split(tok[1], ".")[0])
			if base == "" {
				continue
			}
			if existing, ok := mnemMap[base]; !ok {
				mnemMap[base] = &instInfo{
					Mnemonic:  base,
					Category:  catName,
					CatEnum:   categoryToEnum(catName),
					MinPTX:    ptxVer,
					MinTarget: targetSM,
				}
			} else {
				if existing.CatEnum == "CatUnknown" || existing.CatEnum == "CatMiscellaneous" {
					if newCat := categoryToEnum(catName); newCat != "CatUnknown" && newCat != "CatMiscellaneous" {
						existing.Category = catName
						existing.CatEnum = newCat
					}
				}
				if existing.MinPTX == "" && ptxVer != "" {
					existing.MinPTX = ptxVer
				}
				if existing.MinTarget == "" && targetSM != "" {
					existing.MinTarget = targetSM
				}
			}
		}
	}

	// Family overrides for versions documented in family introductions
	overrides := map[string]struct{ ptx, sm string }{
		"tcgen05":   {"8.6", "sm_100a"},
		"wgmma":     {"8.0", "sm_90a"},
		"wmma":      {"6.0", "sm_70"},
		"mma":       {"6.4", "sm_70"},
		"mbarrier":  {"7.0", "sm_80"},
		"fabric":    {"9.3", "sm_100"},
		"nanosleep": {"6.3", "sm_70"},
	}
	for m, ov := range overrides {
		if info, ok := mnemMap[m]; ok {
			info.MinPTX = ov.ptx
			info.MinTarget = ov.sm
		}
	}

	// Ensure all keywords from Table 2 are present
	for k := range t2Seen {
		if _, ok := mnemMap[k]; !ok {
			mnemMap[k] = &instInfo{
				Mnemonic:  k,
				Category:  "Miscellaneous Instructions",
				CatEnum:   "CatMiscellaneous",
				MinPTX:    "1.0",
				MinTarget: "all",
			}
		}
	}

	var list []instInfo
	for _, info := range mnemMap {
		if info.Mnemonic == "" {
			continue
		}
		list = append(list, *info)
	}
	sort.Slice(list, func(i, j int) bool { return list[i].Mnemonic < list[j].Mnemonic })
	return list, nil
}

func generate(version, digest string, directives, specialRegs []string, types []typeInfo, instructions []instInfo) ([]byte, error) {
	var b strings.Builder
	fmt.Fprintf(&b, "// Code generated by internal/genptxtables. DO NOT EDIT.\n//\n")
	fmt.Fprintf(&b, "// Source: PTX ISA %s\n", version)
	fmt.Fprintf(&b, "// docs/ptx/source.json sha256: %s\n\n", digest)
	fmt.Fprintf(&b, "package cudaptx\n\nimport \"strings\"\n\n")
	fmt.Fprintf(&b, "// PTXISAVersion is the PTX ISA version the generated tables describe.\nconst PTXISAVersion = %q\n\n", version)

	// Category enum
	fmt.Fprintf(&b, `// Category represents the functional category of a PTX instruction.
type Category uint8

const (
	CatUnknown Category = iota
	CatInteger
	CatExtendedPrecision
	CatFloatingPoint
	CatComparison
	CatLogicShift
	CatDataMovement
	CatFabric
	CatTexture
	CatSurface
	CatControlFlow
	CatParallelSync
	CatMatrixWMMA
	CatMatrixWGMMA
	CatTensorCore5thGen
	CatStack
	CatVideo
	CatMiscellaneous
)

func (c Category) String() string {
	switch c {
	case CatInteger:
		return "Integer"
	case CatExtendedPrecision:
		return "ExtendedPrecision"
	case CatFloatingPoint:
		return "FloatingPoint"
	case CatComparison:
		return "Comparison"
	case CatLogicShift:
		return "LogicShift"
	case CatDataMovement:
		return "DataMovement"
	case CatFabric:
		return "Fabric"
	case CatTexture:
		return "Texture"
	case CatSurface:
		return "Surface"
	case CatControlFlow:
		return "ControlFlow"
	case CatParallelSync:
		return "ParallelSync"
	case CatMatrixWMMA:
		return "MatrixWMMA"
	case CatMatrixWGMMA:
		return "MatrixWGMMA"
	case CatTensorCore5thGen:
		return "TensorCore5thGen"
	case CatStack:
		return "Stack"
	case CatVideo:
		return "Video"
	case CatMiscellaneous:
		return "Miscellaneous"
	default:
		return "Unknown"
	}
}

// IsCompute reports whether the category represents compute arithmetic, logic, or matrix operations.
func (c Category) IsCompute() bool {
	switch c {
	case CatInteger, CatExtendedPrecision, CatFloatingPoint, CatComparison, CatLogicShift,
		CatMatrixWMMA, CatMatrixWGMMA, CatTensorCore5thGen, CatVideo:
		return true
	default:
		return false
	}
}

// IsMemory reports whether the category represents data movement, load, store, or texture access.
func (c Category) IsMemory() bool {
	switch c {
	case CatDataMovement, CatFabric, CatTexture, CatSurface:
		return true
	default:
		return false
	}
}

// IsMatrix reports whether the category represents Tensor Core or MMA matrix operations.
func (c Category) IsMatrix() bool {
	switch c {
	case CatMatrixWMMA, CatMatrixWGMMA, CatTensorCore5thGen:
		return true
	default:
		return false
	}
}

// IsControlFlow reports whether the category represents branching, calls, or exits.
func (c Category) IsControlFlow() bool {
	return c == CatControlFlow
}

`)

	// IsInstructionKeyword
	fmt.Fprintf(&b, `// IsInstructionKeyword reports whether mnemonic is a recognized PTX instruction keyword.
//
// A false result does not mean the mnemonic is invalid: ptxas, not this table,
// decides what PTX is accepted, and a newer ISA has keywords this table does
// not. Use it to audit parser output, never to reject input.
func IsInstructionKeyword(mnemonic string) bool {
	_, ok := instructionKeywords[strings.ToLower(mnemonic)]
	return ok
}

// InstructionCategory reports the functional category for a PTX instruction mnemonic.
func InstructionCategory(mnemonic string) Category {
	return instructionCategories[strings.ToLower(mnemonic)]
}

// InstructionMinPTX returns the earliest PTX ISA version that introduced the instruction,
// or an empty string if unknown.
func InstructionMinPTX(mnemonic string) string {
	return instructionMinPTX[strings.ToLower(mnemonic)]
}

// InstructionMinTarget returns the minimum GPU architecture required by the instruction
// (e.g. "sm_90", "sm_100a"), "all", or an empty string if unknown.
func InstructionMinTarget(mnemonic string) string {
	return instructionMinTarget[strings.ToLower(mnemonic)]
}

var instructionKeywords = map[string]struct{}{
`)
	for _, inst := range instructions {
		fmt.Fprintf(&b, "\t%q: {},\n", inst.Mnemonic)
	}
	fmt.Fprintf(&b, "}\n\n")

	fmt.Fprintf(&b, "var instructionCategories = map[string]Category{\n")
	for _, inst := range instructions {
		fmt.Fprintf(&b, "\t%q: %s,\n", inst.Mnemonic, inst.CatEnum)
	}
	fmt.Fprintf(&b, "}\n\n")

	fmt.Fprintf(&b, "var instructionMinPTX = map[string]string{\n")
	for _, inst := range instructions {
		if inst.MinPTX != "" {
			fmt.Fprintf(&b, "\t%q: %q,\n", inst.Mnemonic, inst.MinPTX)
		}
	}
	fmt.Fprintf(&b, "}\n\n")

	fmt.Fprintf(&b, "var instructionMinTarget = map[string]string{\n")
	for _, inst := range instructions {
		if inst.MinTarget != "" {
			fmt.Fprintf(&b, "\t%q: %q,\n", inst.Mnemonic, inst.MinTarget)
		}
	}
	fmt.Fprintf(&b, "}\n\n")

	// Data types
	fmt.Fprintf(&b, `// IsDataType reports whether typ (with or without leading dot) is a recognized
// PTX fundamental, packed, sub-byte, or microscaling data type.
func IsDataType(typ string) bool {
	typ = strings.TrimPrefix(strings.ToLower(strings.TrimSpace(typ)), ".")
	_, ok := ptxDataTypes[typ]
	return ok
}

// TypeBitSize returns the size in bits of typ, or 0 if unknown.
func TypeBitSize(typ string) int {
	typ = strings.TrimPrefix(strings.ToLower(strings.TrimSpace(typ)), ".")
	if info, ok := ptxDataTypes[typ]; ok {
		return info.bitSize
	}
	return 0
}

// IsPackedType reports whether typ is a packed multi-element vector type (e.g. .f16x2, .s8x4).
func IsPackedType(typ string) bool {
	typ = strings.TrimPrefix(strings.ToLower(strings.TrimSpace(typ)), ".")
	if info, ok := ptxDataTypes[typ]; ok {
		return info.packed
	}
	return false
}

// VectorElementCount returns the number of vector elements in typ (1 for scalar, 2 for x2, 4 for x4).
func VectorElementCount(typ string) int {
	typ = strings.TrimPrefix(strings.ToLower(strings.TrimSpace(typ)), ".")
	if info, ok := ptxDataTypes[typ]; ok {
		return info.elemCount
	}
	return 1
}

type dataTypeInfo struct {
	bitSize   int
	packed    bool
	elemCount int
}

var ptxDataTypes = map[string]dataTypeInfo{
`)
	for _, t := range types {
		fmt.Fprintf(&b, "\t%q: {bitSize: %d, packed: %t, elemCount: %d},\n", t.Name, t.BitSize, t.Packed, t.ElemCount)
	}
	fmt.Fprintf(&b, "}\n\n")

	// Special Registers
	fmt.Fprintf(&b, `// IsSpecialRegister reports whether name is a predefined PTX special register or identifier,
// such as %%tid, %%ctaid, %%clock64, %%laneid, %%cluster_ctaid, %%dynamic_smem_size, etc.
func IsSpecialRegister(name string) bool {
	name = strings.TrimSpace(name)
	if _, ok := ptxSpecialRegisters[name]; ok {
		return true
	}
	// Hardware indexed registers: %%envreg<N>, %%pm<N>, %%pm<N>_64
	if strings.HasPrefix(name, "%%envreg") {
		return true
	}
	return false
}

var ptxSpecialRegisters = map[string]struct{}{
`)
	for _, r := range specialRegs {
		fmt.Fprintf(&b, "\t%q: {},\n", r)
	}
	fmt.Fprintf(&b, "}\n\n")

	// Directives
	fmt.Fprintf(&b, `// IsDirective reports whether name (with or without leading dot) is a recognized
// PTX directive statement or qualifier.
func IsDirective(name string) bool {
	name = strings.TrimSpace(name)
	if !strings.HasPrefix(name, ".") {
		name = "." + name
	}
	_, ok := ptxDirectives[strings.ToLower(name)]
	return ok
}

var ptxDirectives = map[string]struct{}{
`)
	for _, d := range directives {
		fmt.Fprintf(&b, "\t%q: {},\n", d)
	}
	fmt.Fprintf(&b, "}\n")

	return format.Source([]byte(b.String()))
}

type DocKind string

const (
	DocKindInstruction DocKind = "instruction"
	DocKindDirective   DocKind = "directive"
	DocKindRegister    DocKind = "register"
	DocKindType        DocKind = "type"
	DocKindModifier    DocKind = "modifier"
)

type Doc struct {
	Name        string   `json:"name"`
	Kind        DocKind  `json:"kind"`
	Title       string   `json:"title,omitempty"`
	Summary     string   `json:"summary,omitempty"`
	Syntax      []string `json:"syntax,omitempty"`
	Description string   `json:"description,omitempty"`
	Semantics   string   `json:"semantics,omitempty"`
	Category    string   `json:"category,omitempty"`
	MinPTX      string   `json:"min_ptx,omitempty"`
	MinTarget   string   `json:"min_target,omitempty"`
	Notes       string   `json:"notes,omitempty"`
}

func generateDocs(docsDir, docout string, types []typeInfo) ([]byte, int, error) {
	docs := make(map[string]*Doc)

	// 1. Instructions from 09-instruction-set.md
	ch9Path := filepath.Join(docsDir, "09-instruction-set.md")
	ch9, err := os.ReadFile(ch9Path)
	if err != nil {
		return nil, 0, fmt.Errorf("read ch9: %w", err)
	}
	ch9Text := string(ch9)

	catMatches := regexp.MustCompile(`### (9\.7\.\d+)\.\s+([^\n]+)`).FindAllStringSubmatch(ch9Text, -1)
	catMap := make(map[string]string)
	for _, m := range catMatches {
		catMap[m[1]] = strings.TrimSpace(m[2])
	}

	headerPattern := regexp.MustCompile(`(?m)^(#{3,6})\s+(9\.7\.\d+(?:\.\d+)*)\.?\s+([^\n]+)$`)
	headers := headerPattern.FindAllStringSubmatchIndex(ch9Text, -1)

	for i, idxs := range headers {
		secNum := ch9Text[idxs[4]:idxs[5]]
		title := ch9Text[idxs[6]:idxs[7]]

		mPrefix := regexp.MustCompile(`(9\.7\.\d+)`).FindString(secNum)
		catName := catMap[mPrefix]
		if catName == "" {
			catName = title
		}

		startPos := idxs[1]
		endPos := len(ch9Text)
		if i+1 < len(headers) {
			endPos = headers[i+1][0]
		}
		body := ch9Text[startPos:endPos]

		tokens := regexp.MustCompile("`([a-zA-Z0-9_\\.]+)`").FindAllStringSubmatch(title, -1)
		if len(tokens) == 0 {
			continue
		}

		summary := extractDocSummary(body)
		syntax := extractDocCodeBlocks(extractDocSection(body, "Syntax"))
		desc := cleanDocProse(extractDocSection(body, "Description"))
		semantics := cleanDocProse(extractDocSection(body, "Semantics"))
		minPTX := extractDocMatch(body, `(?i)Introduced in PTX ISA version\s+(\d+\.\d+)`)
		minTarget := extractDocTarget(body)
		notes := cleanDocProse(extractDocSection(body, "Notes"))

		for _, tok := range tokens {
			name := strings.ToLower(tok[1])
			base := strings.Split(name, ".")[0]
			if base == "" {
				continue
			}

			existing := docs[base]
			if existing == nil {
				existing = &Doc{
					Name:        base,
					Kind:        DocKindInstruction,
					Title:       title,
					Summary:     summary,
					Syntax:      syntax,
					Description: desc,
					Semantics:   semantics,
					Category:    catName,
					MinPTX:      minPTX,
					MinTarget:   minTarget,
					Notes:       notes,
				}
				docs[base] = existing
			} else {
				for _, s := range syntax {
					if !containsString(existing.Syntax, s) {
						existing.Syntax = append(existing.Syntax, s)
					}
				}
				if existing.Summary == "" {
					existing.Summary = summary
				}
				if existing.MinPTX == "" {
					existing.MinPTX = minPTX
				}
				if existing.MinTarget == "" {
					existing.MinTarget = minTarget
				}
			}

			// Also index compound instructions like tcgen05.alloc
			if name != base && docs[name] == nil {
				docs[name] = &Doc{
					Name:        name,
					Kind:        DocKindInstruction,
					Title:       title,
					Summary:     summary,
					Syntax:      syntax,
					Description: desc,
					Semantics:   semantics,
					Category:    catName,
					MinPTX:      minPTX,
					MinTarget:   minTarget,
					Notes:       notes,
				}
			}
		}
	}

	// Apply family overrides
	overrides := map[string]struct{ ptx, sm string }{
		"tcgen05":   {"8.6", "sm_100a"},
		"wgmma":     {"8.0", "sm_90a"},
		"wmma":      {"6.0", "sm_70"},
		"mma":       {"6.4", "sm_70"},
		"mbarrier":  {"7.0", "sm_80"},
		"fabric":    {"9.3", "sm_100"},
		"nanosleep": {"6.3", "sm_70"},
	}
	for m, ov := range overrides {
		if d, ok := docs[m]; ok {
			if d.MinPTX == "" {
				d.MinPTX = ov.ptx
			}
			if d.MinTarget == "" {
				d.MinTarget = ov.sm
			}
		}
	}
	// Inherit base mnemonic min versions for compounds if unset
	for name, d := range docs {
		if d.Kind == DocKindInstruction && strings.Contains(name, ".") {
			base := strings.Split(name, ".")[0]
			if baseDoc, ok := docs[base]; ok {
				if d.MinPTX == "" {
					d.MinPTX = baseDoc.MinPTX
				}
				if d.MinTarget == "" {
					d.MinTarget = baseDoc.MinTarget
				}
			}
		}
	}

	// 2. Special Registers from 10-special-registers.md
	ch10Path := filepath.Join(docsDir, "10-special-registers.md")
	if ch10, err := os.ReadFile(ch10Path); err == nil {
		ch10Text := string(ch10)
		headers10 := regexp.MustCompile(`(?m)^## 10\.\d+\.\s+Special Registers:\s*([^\n]+)$`).FindAllStringSubmatchIndex(ch10Text, -1)
		for i, idxs := range headers10 {
			title := ch10Text[idxs[2]:idxs[3]]
			startPos := idxs[1]
			endPos := len(ch10Text)
			if i+1 < len(headers10) {
				endPos = headers10[i+1][0]
			}
			body := ch10Text[startPos:endPos]
			summary := extractDocSummary(body)
			syntax := extractDocCodeBlocks(extractDocSection(body, "Syntax (predefined)"))
			if len(syntax) == 0 {
				syntax = extractDocCodeBlocks(extractDocSection(body, "Syntax"))
			}
			desc := cleanDocProse(extractDocSection(body, "Description"))
			minPTX := extractDocMatch(body, `(?i)Introduced in PTX ISA version\s+(\d+\.\d+)`)
			minTarget := extractDocTarget(body)

			tokens := regexp.MustCompile("`(%[a-zA-Z0-9_<>]+)`").FindAllStringSubmatch(title, -1)
			for _, tok := range tokens {
				name := tok[1]
				docs[name] = &Doc{
					Name:        name,
					Kind:        DocKindRegister,
					Title:       "Special Register: " + name,
					Summary:     summary,
					Syntax:      syntax,
					Description: desc,
					Category:    "Special Registers",
					MinPTX:      minPTX,
					MinTarget:   minTarget,
				}
				// Vector register components .x, .y, .z
				if name == "%tid" || name == "%ntid" || name == "%ctaid" || name == "%nctaid" || name == "%cluster_ctaid" || name == "%cluster_nctaid" {
					for _, comp := range []string{".x", ".y", ".z"} {
						docs[name+comp] = &Doc{
							Name:        name + comp,
							Kind:        DocKindRegister,
							Title:       fmt.Sprintf("Special Register: %s%s", name, comp),
							Summary:     fmt.Sprintf("%s (%s component)", summary, strings.TrimPrefix(comp, ".")),
							Syntax:      syntax,
							Description: desc,
							Category:    "Special Registers",
							MinPTX:      minPTX,
							MinTarget:   minTarget,
						}
					}
				}
			}
		}
	}

	// 3. Directives from 11-directives.md
	ch11Path := filepath.Join(docsDir, "11-directives.md")
	if ch11, err := os.ReadFile(ch11Path); err == nil {
		ch11Text := string(ch11)
		headers11 := regexp.MustCompile(`(?m)^### 11\.\d+\.\d+\.\s+[^:]+:\s*` + "`" + `(\.[a-z0-9_]+)` + "`").FindAllStringSubmatchIndex(ch11Text, -1)
		for i, idxs := range headers11 {
			name := ch11Text[idxs[2]:idxs[3]]
			startPos := idxs[1]
			endPos := len(ch11Text)
			if i+1 < len(headers11) {
				endPos = headers11[i+1][0]
			}
			body := ch11Text[startPos:endPos]
			summary := extractDocSummary(body)
			syntax := extractDocCodeBlocks(extractDocSection(body, "Syntax"))
			desc := cleanDocProse(extractDocSection(body, "Description"))
			semantics := cleanDocProse(extractDocSection(body, "Semantics"))
			minPTX := extractDocMatch(body, `(?i)Introduced in PTX ISA version\s+(\d+\.\d+)`)
			minTarget := extractDocTarget(body)

			docs[name] = &Doc{
				Name:        name,
				Kind:        DocKindDirective,
				Title:       "Directive: " + name,
				Summary:     summary,
				Syntax:      syntax,
				Description: desc,
				Semantics:   semantics,
				Category:    "Directives",
				MinPTX:      minPTX,
				MinTarget:   minTarget,
			}
		}
	}

	// 4. State spaces
	stateSpaces := []struct{ name, desc, access, sharing string }{
		{".reg", "Registers, fast per-thread storage", "R/W", "per-thread"},
		{".sreg", "Special registers, read-only platform-specific registers", "RO", "per-CTA"},
		{".const", "Constant memory, shared, read-only memory across grid", "RO", "per-grid"},
		{".global", "Global memory, shared by all threads across contexts", "R/W", "Context"},
		{".local", "Local memory, private to each thread", "R/W", "per-thread"},
		{".param", "Kernel parameters (per-grid) or function parameters (per-thread)", "R/W", "per-grid / per-thread"},
		{".shared", "Shared memory, defined per CTA, accessible within cluster", "R/W", "per-cluster"},
		{".tex", "Global texture memory (deprecated)", "RO", "Context"},
	}
	for _, ss := range stateSpaces {
		docs[ss.name] = &Doc{
			Name:        ss.name,
			Kind:        DocKindModifier,
			Title:       "State Space: " + ss.name,
			Summary:     ss.desc,
			Description: fmt.Sprintf("%s. Access: %s, Sharing: %s.", ss.desc, ss.access, ss.sharing),
			Category:    "State Spaces",
		}
	}

	// 5. Data Types from types slice
	typeDescriptions := map[string]string{
		"s8":       "Signed 8-bit integer",
		"s16":      "Signed 16-bit integer",
		"s32":      "Signed 32-bit integer",
		"s64":      "Signed 64-bit integer",
		"u8":       "Unsigned 8-bit integer",
		"u16":      "Unsigned 16-bit integer",
		"u32":      "Unsigned 32-bit integer",
		"u64":      "Unsigned 64-bit integer",
		"f16":      "16-bit half-precision floating point (IEEE 754-2008)",
		"f32":      "32-bit single-precision floating point (IEEE 754)",
		"f64":      "64-bit double-precision floating point (IEEE 754)",
		"b8":       "8-bit untyped bit field",
		"b16":      "16-bit untyped bit field",
		"b32":      "32-bit untyped bit field",
		"b64":      "64-bit untyped bit field",
		"b128":     "128-bit untyped bit field",
		"pred":     "1-bit predicate register",
		"bf16":     "16-bit Brain floating point",
		"tf32":     "19-bit TensorFloat-32 floating point (stored in 32 bits)",
		"e4m3":     "8-bit FP8 floating point (1 sign, 4 exponent, 3 mantissa)",
		"e5m2":     "8-bit FP8 floating point (1 sign, 5 exponent, 2 mantissa)",
		"e2m1":     "4-bit microscopic floating point",
		"e2m3":     "6-bit floating point",
		"e3m2":     "6-bit floating point",
		"ue8m0":    "8-bit unsigned scale factor",
		"ue4m3":    "7-bit unsigned floating point (stored in 8 bits)",
		"ue5m3":    "8-bit unsigned floating point",
		"s2f6":     "8-bit fixed point (2 signed integer bits, 6 fractional bits)",
		"f16x2":    "Pair of 16-bit half-precision floats packed in 32 bits",
		"f32x2":    "Pair of 32-bit single-precision floats packed in 64 bits",
		"bf16x2":   "Pair of 16-bit Brain floats packed in 32 bits",
		"e4m3x2":   "Pair of FP8 e4m3 floats packed in 16 bits",
		"e5m2x2":   "Pair of FP8 e5m2 floats packed in 16 bits",
		"e2m3x2":   "Pair of 6-bit e2m3 floats packed in 16 bits",
		"e3m2x2":   "Pair of 6-bit e3m2 floats packed in 16 bits",
		"ue8m0x2":  "Pair of 8-bit unsigned scale factors packed in 16 bits",
		"ue4m3x2":  "Pair of 7-bit unsigned floats packed in 16 bits",
		"ue5m3x2":  "Pair of 8-bit unsigned floats packed in 16 bits",
		"s2f6x2":   "Pair of 8-bit fixed-point values packed in 16 bits",
		"e2m1x2":   "Pair of 4-bit microscopic floats packed in 8 bits",
		"e4m3x4":   "Four FP8 e4m3 floats packed in 32 bits",
		"e5m2x4":   "Four FP8 e5m2 floats packed in 32 bits",
		"e2m3x4":   "Four 6-bit e2m3 floats packed in 32 bits",
		"e3m2x4":   "Four 6-bit e3m2 floats packed in 32 bits",
		"e2m1p4x4": "Four 4-bit e2m1 floats packed with padding in 32 bits",
		"ue8m0x4":  "Four 8-bit unsigned scale factors packed in 32 bits",
		"e2m1x4":   "Four 4-bit e2m1 floats packed in 16 bits",
		"u16x2":    "Pair of unsigned 16-bit integers packed in 32 bits",
		"s16x2":    "Pair of signed 16-bit integers packed in 32 bits",
		"u8x4":     "Four unsigned 8-bit integers packed in 32 bits",
		"s8x4":     "Four signed 8-bit integers packed in 32 bits",
	}
	for _, t := range types {
		name := "." + t.Name
		desc := typeDescriptions[t.Name]
		if desc == "" {
			if t.Packed {
				desc = fmt.Sprintf("%d-bit packed data type (%d elements)", t.BitSize, t.ElemCount)
			} else {
				desc = fmt.Sprintf("%d-bit scalar data type", t.BitSize)
			}
		}
		docs[name] = &Doc{
			Name:        name,
			Kind:        DocKindType,
			Title:       "Data Type: " + name,
			Summary:     desc,
			Description: fmt.Sprintf("%s. Total size: %d bits.", desc, t.BitSize),
			Category:    "Data Types",
		}
	}

	// 6. Standard Modifiers
	modifiers := []struct{ name, meaning, cat string }{
		{".rn", "Round to nearest even (IEEE 754 default)", "Rounding Mode"},
		{".rz", "Round towards zero (truncate)", "Rounding Mode"},
		{".rm", "Round towards negative infinity", "Rounding Mode"},
		{".rp", "Round towards positive infinity", "Rounding Mode"},
		{".rni", "Round to nearest integer, ties to even", "Rounding Mode"},
		{".rzi", "Round to nearest integer towards zero", "Rounding Mode"},
		{".rmi", "Round to nearest integer towards negative infinity", "Rounding Mode"},
		{".rpi", "Round to nearest integer towards positive infinity", "Rounding Mode"},
		{".ftz", "Flush subnormal inputs and results to sign-preserving zero", "Floating-Point Option"},
		{".sat", "Saturate result to [0.0, 1.0] for float or [min, max] for integer", "Format Option"},
		{".satfinite", "Saturate result to finite range [min, max]", "Format Option"},
		{".relu", "Apply rectified linear unit (clamp negative numbers to zero)", "Math Option"},
		{".ca", "Cache at all levels (L1 and L2)", "Cache Hint"},
		{".cg", "Cache at global level only (L2, bypass L1)", "Cache Hint"},
		{".cs", "Cache streaming (likely to be accessed once, discard soon)", "Cache Hint"},
		{".lu", "Last use (discard from cache after this access)", "Cache Hint"},
		{".cv", "Cache volatile (bypass caches, fetch latest memory)", "Cache Hint"},
		{".wb", "Write-back to L2 cache", "Cache Hint"},
		{".wt", "Write-through directly to system memory", "Cache Hint"},
		{".relaxed", "Relaxed memory consistency (no acquire/release barrier)", "Memory Consistency"},
		{".acquire", "Acquire memory barrier (subsequent reads/writes ordered after)", "Memory Consistency"},
		{".release", "Release memory barrier (prior reads/writes ordered before)", "Memory Consistency"},
		{".acq_rel", "Acquire-release fence/barrier (both acquire and release)", "Memory Consistency"},
		{".volatile", "Volatile memory access (compiler cannot reorder or optimize away)", "Memory Consistency"},
		{".cta", "CTA scope (synchronization/consistency visible within thread block)", "Memory Scope"},
		{".cluster", "Cluster scope (synchronization/consistency visible within cluster)", "Memory Scope"},
		{".gpu", "GPU scope (synchronization/consistency visible across device)", "Memory Scope"},
		{".sys", "System scope (synchronization/consistency visible across GPU and CPU host)", "Memory Scope"},
		{".sync", "Synchronous execution across participating threads", "Synchronization"},
		{".aligned", "Threads in warp execute in lockstep at this instruction", "Synchronization"},
		{".arrive", "Arrive at asynchronous barrier without waiting", "Asynchronous Barrier"},
		{".wait", "Wait at asynchronous barrier until arrival count met", "Asynchronous Barrier"},
		{".cta_group::1", "CTA group size 1 (single CTA in distributed shared memory)", "Cluster Modifier"},
		{".cta_group::2", "CTA group size 2 (pair of CTAs in distributed shared memory)", "Cluster Modifier"},
		{".eq", "Equal comparison", "Comparison Condition"},
		{".ne", "Not-equal comparison", "Comparison Condition"},
		{".lt", "Less-than comparison", "Comparison Condition"},
		{".le", "Less-than-or-equal comparison", "Comparison Condition"},
		{".gt", "Greater-than comparison", "Comparison Condition"},
		{".ge", "Greater-than-or-equal comparison", "Comparison Condition"},
		{".lo", "Lower (unsigned <) comparison", "Comparison Condition"},
		{".ls", "Lower-or-same (unsigned <=) comparison", "Comparison Condition"},
		{".hi", "Higher (unsigned >) comparison", "Comparison Condition"},
		{".hs", "Higher-or-same (unsigned >=) comparison", "Comparison Condition"},
		{".equ", "Equal or unordered comparison", "Comparison Condition"},
		{".neu", "Not-equal or unordered comparison", "Comparison Condition"},
		{".ltu", "Less-than or unordered comparison", "Comparison Condition"},
		{".leu", "Less-than-or-equal or unordered comparison", "Comparison Condition"},
		{".gtu", "Greater-than or unordered comparison", "Comparison Condition"},
		{".geu", "Greater-than-or-equal or unordered comparison", "Comparison Condition"},
		{".num", "Ordered comparison (neither operand is NaN)", "Comparison Condition"},
		{".nan", "Unordered comparison (either operand is NaN)", "Comparison Condition"},
		{".v2", "Two-element vector tuple", "Vector Specifier"},
		{".v4", "Four-element vector tuple", "Vector Specifier"},
		{".v8", "Eight-element vector tuple", "Vector Specifier"},
	}
	for _, m := range modifiers {
		docs[m.name] = &Doc{
			Name:        m.name,
			Kind:        DocKindModifier,
			Title:       "Modifier: " + m.name,
			Summary:     m.meaning,
			Description: m.meaning,
			Category:    m.cat,
		}
	}

	var names []string
	for name := range docs {
		names = append(names, name)
	}
	sort.Strings(names)

	var buf bytes.Buffer
	var out io.Writer = &buf
	var gw *gzip.Writer
	if strings.HasSuffix(docout, ".gz") {
		gw = gzip.NewWriter(&buf)
		out = gw
	}

	for _, name := range names {
		line, err := json.Marshal(docs[name])
		if err != nil {
			return nil, 0, fmt.Errorf("marshal doc %s: %w", name, err)
		}
		if _, err := out.Write(line); err != nil {
			return nil, 0, err
		}
		if _, err := out.Write([]byte("\n")); err != nil {
			return nil, 0, err
		}
	}

	if gw != nil {
		if err := gw.Close(); err != nil {
			return nil, 0, fmt.Errorf("gzip docs: %w", err)
		}
	}

	return buf.Bytes(), len(docs), nil
}

func extractDocSummary(body string) string {
	lines := strings.Split(strings.TrimSpace(body), "\n")
	var textLines []string
	inCode := false
	for _, l := range lines {
		trimmed := strings.TrimSpace(l)
		if strings.HasPrefix(trimmed, "```") {
			inCode = !inCode
			continue
		}
		if inCode || trimmed == "" || strings.HasPrefix(trimmed, "`") || strings.HasPrefix(trimmed, "#") {
			continue
		}
		if strings.HasPrefix(trimmed, "Syntax") || strings.HasPrefix(trimmed, "Description") || strings.HasPrefix(trimmed, "Semantics") || strings.HasPrefix(trimmed, "PTX ISA Notes") {
			break
		}
		textLines = append(textLines, trimmed)
		if len(textLines) >= 2 {
			break
		}
	}
	return strings.Join(textLines, " ")
}

func extractDocSection(body, secName string) string {
	pattern := regexp.MustCompile(`(?m)^` + regexp.QuoteMeta(secName) + `\s*$`)
	loc := pattern.FindStringIndex(body)
	if loc == nil {
		return ""
	}
	rest := body[loc[1]:]
	nextPat := regexp.MustCompile(`(?m)^(?:#{2,6}\s+|Syntax\s*$|Description\s*$|Semantics\s*$|PTX ISA Notes\s*$|Target ISA Notes\s*$|Examples\s*$)`)
	nextLoc := nextPat.FindStringIndex(rest)
	if nextLoc != nil {
		return strings.TrimSpace(rest[:nextLoc[0]])
	}
	return strings.TrimSpace(rest)
}

func extractDocCodeBlocks(text string) []string {
	re := regexp.MustCompile("(?s)```(?:ptx)?\\n(.*?)\\n```")
	matches := re.FindAllStringSubmatch(text, -1)
	var blocks []string
	for _, m := range matches {
		code := strings.TrimSpace(m[1])
		if code != "" {
			blocks = append(blocks, code)
		}
	}
	return blocks
}

func cleanDocProse(text string) string {
	text = strings.ReplaceAll(text, "Copy to clipboard\n", "")
	text = strings.ReplaceAll(text, "Copy to clipboard", "")
	return strings.TrimSpace(text)
}

func extractDocMatch(body, pattern string) string {
	m := regexp.MustCompile(pattern).FindStringSubmatch(body)
	if len(m) >= 2 {
		return m[1]
	}
	return ""
}

func extractDocTarget(body string) string {
	idx := strings.Index(body, "Target ISA Notes")
	if idx < 0 {
		return ""
	}
	notePart := body[idx:]
	if exIdx := strings.Index(notePart, "Examples"); exIdx >= 0 {
		notePart = notePart[:exIdx]
	}
	if strings.Contains(strings.ToLower(notePart), "all target") {
		return "all"
	}
	m := regexp.MustCompile(`sm_\d+[a-z]?`).FindString(notePart)
	return m
}

func containsString(slice []string, s string) bool {
	for _, item := range slice {
		if item == s {
			return true
		}
	}
	return false
}
