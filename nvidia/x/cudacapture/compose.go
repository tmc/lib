package cudacapture

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
)

// VerifyModule checks that the module generation id recorded in c is backed by
// the code artifacts retained in c, and returns that module's decoded code.
//
// imageSHA256 is the hex digest of the image the producer handed the driver,
// recorded when the module was loaded. VerifyModule recomputes it over the
// retained CUBIN bytes, so a reader holding only the capture learns that the
// retained artifact is the image the producer loaded. It is a producer record
// verified against retained bytes, not a driver attestation, and it is
// required: a kernel name alone does not identify a module generation.
//
// kernels are the kernel names that measurements in the capture are attributed
// to. Each must resolve in the module's disassembly; an unresolved name means
// the measurements and the artifacts describe different code.
func VerifyModule(c *Capture, id, imageSHA256 string, kernels []string) (*Code, error) {
	if c == nil {
		return nil, errors.New("cudacapture: nil capture")
	}
	if id == "" {
		return nil, errors.New("cudacapture: empty module id")
	}
	var module *Module
	for i := range c.Modules {
		if c.Modules[i].ID == id {
			if module != nil {
				return nil, fmt.Errorf("cudacapture: duplicate module %q", id)
			}
			module = &c.Modules[i]
		}
	}
	if module == nil {
		return nil, fmt.Errorf("cudacapture: no module %q in capture", id)
	}
	roles := []struct{ role, name string }{
		{"source", module.Code.Source},
		{"ptx", module.Code.PTX},
		{"cubin", module.Code.CUBIN},
		{"sass", module.Code.SASS},
	}
	for _, r := range roles {
		if r.name == "" {
			return nil, fmt.Errorf("cudacapture: module %q has no %s artifact", id, r.role)
		}
		if err := validateFilename(r.name); err != nil {
			return nil, fmt.Errorf("cudacapture: module %q %s artifact: %w", id, r.role, err)
		}
		if _, ok := c.Files[r.name]; !ok {
			return nil, fmt.Errorf("cudacapture: module %q %s file %q not retained", id, r.role, r.name)
		}
	}
	if err := verifyDigest(c.Files[module.Code.CUBIN], imageSHA256); err != nil {
		return nil, fmt.Errorf("cudacapture: module %q image: %w", id, err)
	}
	code, err := LoadCode(&Capture{Profile: c.Profile, Files: c.Files, Code: module.Code, CodeUnavailable: c.CodeUnavailable})
	if err != nil {
		return nil, err
	}
	for _, name := range kernels {
		if code.Kernel(name) == nil {
			return nil, fmt.Errorf("cudacapture: module %q does not define kernel %q", id, name)
		}
	}
	return code, nil
}

// verifyDigest reports whether data hashes to the lowercase hex digest want.
func verifyDigest(data []byte, want string) error {
	if want == "" {
		return errors.New("missing sha256 digest")
	}
	if len(want) != 2*sha256.Size || strings.ToLower(want) != want {
		return fmt.Errorf("malformed sha256 digest %q", want)
	}
	if _, err := hex.DecodeString(want); err != nil {
		return fmt.Errorf("malformed sha256 digest %q", want)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != want {
		return fmt.Errorf("sha256 is %s, want %s", got, want)
	}
	return nil
}
