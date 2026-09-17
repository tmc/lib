package nvrtcx

import (
	"fmt"
	"io/fs"
	"sort"
	"strings"
)

// WithHeadersFS adds files from an fs.FS as in-memory NVRTC headers.
//
// With no patterns, every regular file in the filesystem is added using its
// slash-separated fs.FS path as the include name. With patterns, each pattern
// is expanded with fs.Glob in argument order; duplicate matches are added only
// once. The last option wins when a file has the same name as a WithHeader or
// another WithHeadersFS option.
//
// The source must include the files by those names. For example, an embedded
// file named "include/math.cuh" is included with #include "include/math.cuh".
// This option reads the files when the option is applied, so changes to a
// mutable fs.FS do not affect a compilation after Compile has started.
func WithHeadersFS(fsys fs.FS, patterns ...string) Option {
	patterns = append([]string(nil), patterns...)
	return func(o *options) {
		if fsys == nil {
			o.err = fmt.Errorf("nvrtcx: nil header filesystem")
			return
		}

		names, err := headerFSNames(fsys, patterns)
		if err != nil {
			o.err = err
			return
		}
		if o.headers == nil {
			o.headers = make(map[string]string)
		}
		for _, name := range names {
			content, err := fs.ReadFile(fsys, name)
			if err != nil {
				o.err = fmt.Errorf("nvrtcx: read header %q: %w", name, err)
				return
			}
			if strings.ContainsRune(string(content), 0) {
				o.err = fmt.Errorf("nvrtcx: header %q contains NUL", name)
				return
			}
			o.headers[name] = string(content)
		}
	}
}

func headerFSNames(fsys fs.FS, patterns []string) ([]string, error) {
	if len(patterns) == 0 {
		var names []string
		err := fs.WalkDir(fsys, ".", func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if !entry.IsDir() {
				names = append(names, path)
			}
			return nil
		})
		if err != nil {
			return nil, fmt.Errorf("nvrtcx: walk header filesystem: %w", err)
		}
		sort.Strings(names)
		return names, nil
	}

	seen := make(map[string]bool)
	var names []string
	for _, pattern := range patterns {
		if pattern == "" {
			return nil, fmt.Errorf("nvrtcx: empty header filesystem pattern")
		}
		matches, err := fs.Glob(fsys, pattern)
		if err != nil {
			return nil, fmt.Errorf("nvrtcx: glob header filesystem pattern %q: %w", pattern, err)
		}
		if len(matches) == 0 {
			return nil, fmt.Errorf("nvrtcx: header filesystem pattern %q matched no files", pattern)
		}
		for _, name := range matches {
			info, err := fs.Stat(fsys, name)
			if err != nil {
				return nil, fmt.Errorf("nvrtcx: stat header %q: %w", name, err)
			}
			if info.IsDir() {
				continue
			}
			if !seen[name] {
				seen[name] = true
				names = append(names, name)
			}
		}
	}
	return names, nil
}
