// Command egbundle compiles and inspects zero-copy, 4096-byte page-aligned
// Safetensors + cross-layer Megakernel AOT archives (.egbundle) with PGO
// cache manifests and SHA-256 integrity verification.
package main

import (
	"bytes"
	"cmp"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"slices"

	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/specdecode"
)

const (
	// PageAlignment is the 4096-byte OS page alignment enforced for all weight
	// and cubin sections in an .egbundle archive for zero-copy mmap.
	PageAlignment = 4096

	bundleMagic   = "EGBNDL01"
	headerPrefix  = 16 // 8-byte magic + 8-byte uint64 manifest JSON length
	formatVersion = 1
)

// WeightTensorSpec describes an input weight tensor to pack into an .egbundle
// at a 4096-byte page-aligned offset.
type WeightTensorSpec struct {
	Name  string
	DType string
	Shape []int
	Data  []byte
}

// SafetensorsTensorHeader records Safetensors-compatible metadata and the
// 4096-byte page-aligned absolute file offset for zero-copy mmap access.
type SafetensorsTensorHeader struct {
	DType       string    `json:"dtype"`
	Shape       []int     `json:"shape"`
	DataOffsets [2]uint64 `json:"data_offsets"`
	PageOffset  uint64    `json:"page_offset"`
	ByteLength  uint64    `json:"byte_length"`
	SHA256      string    `json:"sha256"`
}

// BucketKernelArtifact stores the pre-lowered cross-layer megakernel CUDA/PTX
// translation unit for one speculative/batch shape bucket M in {1, 2, 4, 8}.
type BucketKernelArtifact struct {
	BucketM      int    `json:"bucket_m"`
	KernelName   string `json:"kernel_name"`
	Regime       string `json:"regime"`
	CUDASource   string `json:"cuda_source"`
	PTX          string `json:"ptx,omitempty"`
	SourceSHA256 string `json:"source_sha256"`
}

// PGOManifestEntry binds a verified PGO promotion proof and its page-aligned
// cubin slice inside an .egbundle archive.
type PGOManifestEntry struct {
	CacheID     string                     `json:"cache_id"`
	Key         autotune.PGOCacheKey       `json:"key"`
	Proof       autotune.PGOPromotionProof `json:"proof"`
	CubinOffset uint64                     `json:"cubin_offset"`
	CubinLength uint64                     `json:"cubin_length"`
	CubinSHA256 string                     `json:"cubin_sha256"`
}

// BundleManifest is the top-level manifest embedded at the start of an
// .egbundle archive.
type BundleManifest struct {
	FormatVersion int                                `json:"format_version"`
	Arch          string                             `json:"arch"`
	PageAlignment int                                `json:"page_alignment"`
	DataBasePage  uint64                             `json:"data_base_page"`
	Buckets       []BucketKernelArtifact             `json:"buckets"`
	PGOManifest   []PGOManifestEntry                 `json:"pgo_manifest,omitempty"`
	Weights       map[string]SafetensorsTensorHeader `json:"weights"`
	PayloadSHA256 string                             `json:"payload_sha256"`
}

// BundleBuildConfig configures compilation of a deterministic .egbundle archive.
type BundleBuildConfig struct {
	Arch     string
	Buckets  []BucketKernelArtifact
	PGOCache *autotune.VerifiedPGOCache
	Weights  []WeightTensorSpec
}

func alignUpPage(offset uint64) uint64 {
	rem := offset % PageAlignment
	if rem == 0 {
		return offset
	}
	return offset + (PageAlignment - rem)
}

// DefaultBucketKernelArtifacts builds the canonical M in {1, 2, 4, 8}
// cross-layer megakernel CUDA translation units for arch.
func DefaultBucketKernelArtifacts(arch string) ([]BucketKernelArtifact, error) {
	if arch == "" {
		arch = "sm_121a"
	}
	table, err := specdecode.NewMultiBucketMegakernelTable(specdecode.MultiBucketTableConfig{
		Arch: arch,
	})
	if err != nil {
		return nil, fmt.Errorf("egbundle default bucket kernels: %w", err)
	}
	entries := table.Entries()
	out := make([]BucketKernelArtifact, 0, len(entries))
	for _, e := range entries {
		sum := sha256.Sum256([]byte(e.CUDASource))
		out = append(out, BucketKernelArtifact{
			BucketM:      e.Spec.BucketM,
			KernelName:   e.Megakernel.Name,
			Regime:       string(e.Spec.Regime),
			CUDASource:   e.CUDASource,
			SourceSHA256: hex.EncodeToString(sum[:]),
		})
	}
	return out, nil
}

// BuildBundle packages verified cross-layer megakernel sources across buckets
// (M=1,2,4,8), VerifiedPGOCache entries, and 4096-byte page-aligned weight
// tensors into a deterministic .egbundle archive.
func BuildBundle(cfg BundleBuildConfig) (*BundleManifest, []byte, error) {
	arch := cfg.Arch
	if arch == "" {
		arch = "sm_121a"
	}

	buckets := slices.Clone(cfg.Buckets)
	if len(buckets) == 0 {
		var err error
		buckets, err = DefaultBucketKernelArtifacts(arch)
		if err != nil {
			return nil, nil, err
		}
	}
	slices.SortFunc(buckets, func(a, b BucketKernelArtifact) int {
		return cmp.Compare(a.BucketM, b.BucketM)
	})
	for i := range buckets {
		if buckets[i].CUDASource == "" {
			return nil, nil, fmt.Errorf("egbundle: bucket M=%d has empty CUDA source", buckets[i].BucketM)
		}
		sum := sha256.Sum256([]byte(buckets[i].CUDASource))
		buckets[i].SourceSHA256 = hex.EncodeToString(sum[:])
	}

	weights := slices.Clone(cfg.Weights)
	slices.SortFunc(weights, func(a, b WeightTensorSpec) int {
		return cmp.Compare(a.Name, b.Name)
	})

	type pgoBlob struct {
		entry PGOManifestEntry
		cubin []byte
	}
	var pgoBlobs []pgoBlob
	if cfg.PGOCache != nil {
		keys, err := cfg.PGOCache.List()
		if err != nil {
			return nil, nil, fmt.Errorf("egbundle list pgo cache: %w", err)
		}
		for _, k := range keys {
			proof, cubin, err := cfg.PGOCache.LoadVerified(k)
			if err != nil {
				return nil, nil, fmt.Errorf("egbundle load verified pgo %s: %w", k.CacheID(), err)
			}
			sum := sha256.Sum256(cubin)
			pgoBlobs = append(pgoBlobs, pgoBlob{
				entry: PGOManifestEntry{
					CacheID:     k.CacheID(),
					Key:         k,
					Proof:       proof,
					CubinLength: uint64(len(cubin)),
					CubinSHA256: hex.EncodeToString(sum[:]),
				},
				cubin: cubin,
			})
		}
	}

	// Build the page-aligned payload buffer (relative to dataBasePage = 0 first,
	// then shifted by dataBasePage once manifest size is known).
	var relPayload []byte
	relAlign := func() uint64 {
		aligned := alignUpPage(uint64(len(relPayload)))
		if pad := int(aligned) - len(relPayload); pad > 0 {
			relPayload = append(relPayload, make([]byte, pad)...)
		}
		return aligned
	}

	relWeightOffsets := make([]uint64, len(weights))
	for i, w := range weights {
		if w.Name == "" {
			return nil, nil, errors.New("egbundle: weight tensor name is required")
		}
		off := relAlign()
		relWeightOffsets[i] = off
		relPayload = append(relPayload, w.Data...)
	}

	relCubinOffsets := make([]uint64, len(pgoBlobs))
	for i, pb := range pgoBlobs {
		off := relAlign()
		relCubinOffsets[i] = off
		relPayload = append(relPayload, pb.cubin...)
	}
	_ = relAlign()

	payloadSum := sha256.Sum256(relPayload)
	payloadHex := hex.EncodeToString(payloadSum[:])

	populateManifest := func(basePage uint64) *BundleManifest {
		wMap := make(map[string]SafetensorsTensorHeader, len(weights))
		for i, w := range weights {
			relOff := relWeightOffsets[i]
			absOff := basePage + relOff
			wSum := sha256.Sum256(w.Data)
			dtype := w.DType
			if dtype == "" {
				dtype = "BF16"
			}
			wMap[w.Name] = SafetensorsTensorHeader{
				DType:       dtype,
				Shape:       slices.Clone(w.Shape),
				DataOffsets: [2]uint64{relOff, relOff + uint64(len(w.Data))},
				PageOffset:  absOff,
				ByteLength:  uint64(len(w.Data)),
				SHA256:      hex.EncodeToString(wSum[:]),
			}
		}
		pgoEntries := make([]PGOManifestEntry, len(pgoBlobs))
		for i, pb := range pgoBlobs {
			e := pb.entry
			e.CubinOffset = basePage + relCubinOffsets[i]
			pgoEntries[i] = e
		}
		return &BundleManifest{
			FormatVersion: formatVersion,
			Arch:          arch,
			PageAlignment: PageAlignment,
			DataBasePage:  basePage,
			Buckets:       buckets,
			PGOManifest:   pgoEntries,
			Weights:       wMap,
			PayloadSHA256: payloadHex,
		}
	}

	// Two-pass fixed-point computation for DataBasePage so that changing
	// DataBasePage digit width cannot shift the aligned page boundary.
	var (
		basePage  = uint64(PageAlignment)
		manifest  *BundleManifest
		manJSON   []byte
		err       error
	)
	for range 3 {
		manifest = populateManifest(basePage)
		manJSON, err = json.MarshalIndent(manifest, "", "  ")
		if err != nil {
			return nil, nil, fmt.Errorf("egbundle marshal manifest: %w", err)
		}
		needed := alignUpPage(uint64(headerPrefix + len(manJSON)))
		if needed == basePage {
			break
		}
		basePage = needed
	}

	totalSize := int(basePage) + len(relPayload)
	archive := make([]byte, totalSize)
	copy(archive[0:8], []byte(bundleMagic))
	binary.LittleEndian.PutUint64(archive[8:16], uint64(len(manJSON)))
	copy(archive[16:16+len(manJSON)], manJSON)
	copy(archive[int(basePage):], relPayload)

	return manifest, archive, nil
}

// WriteBundleFile compiles cfg and writes the resulting .egbundle archive to path.
func WriteBundleFile(path string, cfg BundleBuildConfig) (*BundleManifest, error) {
	manifest, archive, err := BuildBundle(cfg)
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(path, archive, 0o644); err != nil {
		return nil, fmt.Errorf("egbundle write %s: %w", path, err)
	}
	return manifest, nil
}

// InspectBundle reads and verifies an .egbundle archive from r, checking
// 4096-byte page alignment and recomputing SHA-256 digests for all bucket
// megakernels, PGO cubins, weight tensors, and payload pages.
func InspectBundle(r io.Reader) (*BundleManifest, error) {
	if r == nil {
		return nil, errors.New("egbundle inspect: nil reader")
	}
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("egbundle inspect read: %w", err)
	}
	if len(data) < headerPrefix {
		return nil, errors.New("egbundle inspect: archive shorter than header prefix")
	}
	if string(data[:8]) != bundleMagic {
		return nil, fmt.Errorf("egbundle inspect: bad magic %q", string(data[:8]))
	}
	manLen := binary.LittleEndian.Uint64(data[8:16])
	if uint64(len(data)) < headerPrefix+manLen {
		return nil, errors.New("egbundle inspect: truncated manifest JSON")
	}

	var manifest BundleManifest
	if err := json.Unmarshal(data[16:16+manLen], &manifest); err != nil {
		return nil, fmt.Errorf("egbundle inspect unmarshal manifest: %w", err)
	}
	if manifest.PageAlignment != PageAlignment {
		return nil, fmt.Errorf("egbundle inspect: page_alignment=%d != %d", manifest.PageAlignment, PageAlignment)
	}
	if manifest.DataBasePage == 0 || manifest.DataBasePage%PageAlignment != 0 {
		return nil, fmt.Errorf("egbundle inspect: data_base_page=%d is not %d-byte aligned", manifest.DataBasePage, PageAlignment)
	}
	if uint64(len(data)) < manifest.DataBasePage {
		return nil, errors.New("egbundle inspect: archive truncated before data_base_page")
	}

	payload := data[manifest.DataBasePage:]
	gotPayloadSum := sha256.Sum256(payload)
	if gotHex := hex.EncodeToString(gotPayloadSum[:]); gotHex != manifest.PayloadSHA256 {
		return nil, fmt.Errorf("egbundle inspect: payload sha256 mismatch: got %s, want %s", gotHex, manifest.PayloadSHA256)
	}

	for _, b := range manifest.Buckets {
		sum := sha256.Sum256([]byte(b.CUDASource))
		if gotHex := hex.EncodeToString(sum[:]); gotHex != b.SourceSHA256 {
			return nil, fmt.Errorf("egbundle inspect: bucket M=%d source sha256 mismatch: got %s, want %s", b.BucketM, gotHex, b.SourceSHA256)
		}
	}

	for name, w := range manifest.Weights {
		if w.PageOffset%PageAlignment != 0 {
			return nil, fmt.Errorf("egbundle inspect: weight %q page_offset %d not aligned to %d", name, w.PageOffset, PageAlignment)
		}
		end := w.PageOffset + w.ByteLength
		if end > uint64(len(data)) {
			return nil, fmt.Errorf("egbundle inspect: weight %q range [%d, %d) out of archive bounds %d", name, w.PageOffset, end, len(data))
		}
		sum := sha256.Sum256(data[w.PageOffset:end])
		if gotHex := hex.EncodeToString(sum[:]); gotHex != w.SHA256 {
			return nil, fmt.Errorf("egbundle inspect: weight %q sha256 mismatch: got %s, want %s", name, gotHex, w.SHA256)
		}
	}

	for _, pe := range manifest.PGOManifest {
		if pe.CubinOffset%PageAlignment != 0 {
			return nil, fmt.Errorf("egbundle inspect: pgo entry %s cubin_offset %d not aligned to %d", pe.CacheID, pe.CubinOffset, PageAlignment)
		}
		end := pe.CubinOffset + pe.CubinLength
		if end > uint64(len(data)) {
			return nil, fmt.Errorf("egbundle inspect: pgo entry %s range [%d, %d) out of bounds", pe.CacheID, pe.CubinOffset, end)
		}
		sum := sha256.Sum256(data[pe.CubinOffset:end])
		if gotHex := hex.EncodeToString(sum[:]); gotHex != pe.CubinSHA256 {
			return nil, fmt.Errorf("egbundle inspect: pgo entry %s cubin sha256 mismatch: got %s, want %s", pe.CacheID, gotHex, pe.CubinSHA256)
		}
	}

	return &manifest, nil
}

// InspectBundleFile opens path and verifies its .egbundle manifest and digests.
func InspectBundleFile(path string) (*BundleManifest, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("egbundle inspect %s: %w", path, err)
	}
	return InspectBundle(bytes.NewReader(b))
}

func run(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("egbundle", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var (
		inspectPath = fs.String("inspect", "", "verify and print manifest of an existing .egbundle file")
		outPath     = fs.String("out", "", "output path for compiled .egbundle archive")
		arch        = fs.String("arch", "sm_121a", "target GPU architecture")
		pgoDir      = fs.String("pgo-cache", "", "optional VerifiedPGOCache directory to embed")
	)
	if err := fs.Parse(args); err != nil {
		return err
	}

	if *inspectPath != "" {
		manifest, err := InspectBundleFile(*inspectPath)
		if err != nil {
			return err
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(manifest)
	}

	if *outPath == "" {
		return errors.New("egbundle: specify -out <file.egbundle> or -inspect <file.egbundle>")
	}
	var cache *autotune.VerifiedPGOCache
	if *pgoDir != "" {
		cache = autotune.NewVerifiedPGOCache(*pgoDir)
	}
	manifest, err := WriteBundleFile(*outPath, BundleBuildConfig{
		Arch:     *arch,
		PGOCache: cache,
	})
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(stdout, "built %s (arch=%s buckets=%d pgo=%d weights=%d payload_sha256=%s)\n",
		*outPath, manifest.Arch, len(manifest.Buckets), len(manifest.PGOManifest), len(manifest.Weights), manifest.PayloadSHA256)
	return err
}

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "egbundle: %v\n", err)
		os.Exit(1)
	}
}
