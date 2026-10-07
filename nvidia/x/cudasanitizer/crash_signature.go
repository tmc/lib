package cudasanitizer

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"hash"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/tmc/lib/nvidia/x/cudatrial"
)

// CrashIdentity identifies one caller-declared trial in an execution epoch.
// Image and Input are SHA-256 digests of the exact kernel image and input.
// Ordinal starts at zero and increases once per trial. All strings must be
// nonempty, printable UTF-8 and at most 1024 bytes. A zero identity is invalid.
// These declarations are not proof that a device executed the trial.
type CrashIdentity struct {
	Kernel  string
	Image   [32]byte
	Epoch   string
	Trial   string
	Input   [32]byte
	Ordinal uint64
}

// CompletedCrashTrace is the preceding trial's host trace. Complete must be
// true only after the caller observed completion. MapSize declares its exact
// expected size; Bitmap holds raw counters, not quantized hit classes.
// The caller must not mutate the bitmap while CrashSignature reads it.
type CompletedCrashTrace struct {
	Identity CrashIdentity
	Complete bool
	MapSize  int
	Bitmap   *cudatrial.EdgeBitmap
}

// CrashObservation associates a defect with a preceding completed trace.
// Signature is a deduplication key, not current or faulting-input coverage.
// Failure and Preceding retain distinct trial and input identities. Headlines
// are sorted unique exact report headlines; detail lines and occurrence counts
// are not part of the key. Version names the signature domain.
type CrashObservation struct {
	Version     string
	Signature   [32]byte
	TraceDigest [32]byte
	Tool        Tool
	Headlines   []string
	Failure     CrashIdentity
	Preceding   CrashIdentity
}

// CrashSignature reports an existing sanitizer defect using the immediately
// preceding completed trace. It requires FoundDefect and a complete parsed log,
// refuses other outcomes, and does not execute, gate or skip any work.
//
// The signature hashes a versioned domain, tool, sorted unique exact headlines,
// kernel/image identity and exact raw trace bytes. Trial/input/epoch identities
// stay in the observation instead of the key, so identical defect signatures
// from different trials can be grouped without losing failing-input identity.
// Adjacency is checked against caller-declared epoch and ordinal; neither this
// check nor nonzero bytes establish hardware provenance or a live oracle.
func CrashSignature(result *Result, preceding CompletedCrashTrace, failure CrashIdentity) (CrashObservation, error) {
	var observation CrashObservation
	if result == nil || !result.FoundDefect() || result.Err != nil || result.Log == nil || !result.Log.SummaryFound || result.Log.Summaries <= 0 {
		return observation, fmt.Errorf("crash signature: no complete defect verdict")
	}
	if !result.Config.Tool.valid() || result.Log.Errors < 0 || result.Log.RaceErrors < 0 || result.Log.Hazards < 0 {
		return observation, fmt.Errorf("crash signature: invalid defect classification")
	}
	if err := crashIdentityValid(failure); err != nil {
		return observation, err
	}
	if err := crashIdentityValid(preceding.Identity); err != nil {
		return observation, err
	}
	p := preceding.Identity
	if !preceding.Complete || p.Ordinal == ^uint64(0) || p.Ordinal+1 != failure.Ordinal || p.Epoch != failure.Epoch || p.Trial == failure.Trial || p.Kernel != failure.Kernel || p.Image != failure.Image {
		return observation, fmt.Errorf("crash signature: trace is not the preceding completed trial")
	}
	if preceding.Bitmap == nil || preceding.MapSize < 2 || preceding.MapSize > 16<<20 || preceding.MapSize%2 != 0 {
		return observation, fmt.Errorf("crash signature: invalid trace size")
	}
	raw := preceding.Bitmap.Bytes()
	if len(raw) != preceding.MapSize {
		return observation, fmt.Errorf("crash signature: trace size mismatch")
	}
	if !slices.ContainsFunc(raw, func(n byte) bool { return n != 0 }) {
		return observation, fmt.Errorf("crash signature: zero trace")
	}
	if len(result.Log.Findings) == 0 || len(result.Log.Findings) > 1024 {
		return observation, fmt.Errorf("crash signature: invalid finding count")
	}
	headlines := make([]string, len(result.Log.Findings))
	for i, finding := range result.Log.Findings {
		if !crashTextValid(finding.Headline) {
			return observation, fmt.Errorf("crash signature: invalid finding headline")
		}
		headlines[i] = finding.Headline
	}
	slices.Sort(headlines)
	headlines = slices.Compact(headlines)
	const version = "cudasanitizer/preceding-trace-defect/v1"
	h := sha256.New()
	for _, value := range []string{version, string(result.Config.Tool), p.Kernel} {
		crashHashField(h, []byte(value))
	}
	crashHashField(h, p.Image[:])
	var count [8]byte
	binary.BigEndian.PutUint64(count[:], uint64(len(headlines)))
	crashHashField(h, count[:])
	for _, headline := range headlines {
		crashHashField(h, []byte(headline))
	}
	crashHashField(h, raw)
	observation = CrashObservation{Version: version, TraceDigest: sha256.Sum256(raw), Tool: result.Config.Tool, Headlines: headlines, Failure: failure, Preceding: p}
	copy(observation.Signature[:], h.Sum(nil))
	return observation, nil
}

func crashIdentityValid(identity CrashIdentity) error {
	if !crashTextValid(identity.Kernel) || !crashTextValid(identity.Epoch) || !crashTextValid(identity.Trial) || identity.Image == [32]byte{} || identity.Input == [32]byte{} {
		return fmt.Errorf("crash signature: unknown trial provenance")
	}
	return nil
}

func crashTextValid(s string) bool {
	return len(s) > 0 && len(s) <= 1024 && utf8.ValidString(s) && strings.TrimSpace(s) == s && !strings.ContainsFunc(s, unicode.IsControl)
}

func crashHashField(h hash.Hash, body []byte) {
	var length [8]byte
	binary.BigEndian.PutUint64(length[:], uint64(len(body)))
	h.Write(length[:])
	h.Write(body)
}
