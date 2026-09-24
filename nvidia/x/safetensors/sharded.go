package safetensors

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
)

// IndexFile is the name of the weight map that Hugging Face writes beside the
// shards of a checkpoint split across several files.
const IndexFile = "model.safetensors.index.json"

// A Sharded is a checkpoint held in one or more safetensors files in a
// directory.
type Sharded struct {
	// Shard maps each tensor name to the name of the file in the
	// directory that holds it.
	Shard map[string]string

	files map[string]*File
}

// OpenSharded opens the checkpoint in dir. It reads the weight map in
// dir/model.safetensors.index.json and opens each file it names, which must
// be a file directly in dir, with [Open]. Every tensor in the weight map must
// be in the file the map names. If dir has no index, OpenSharded opens
// dir/model.safetensors as a checkpoint of one file.
func OpenSharded(dir string) (*Sharded, error) {
	index, err := os.ReadFile(filepath.Join(dir, IndexFile))
	if errors.Is(err, fs.ErrNotExist) {
		return openSingle(dir)
	}
	if err != nil {
		return nil, fmt.Errorf("safetensors: %w", err)
	}
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(index, &idx); err != nil {
		return nil, fmt.Errorf("safetensors: decode %s: %w", IndexFile, err)
	}
	if len(idx.WeightMap) == 0 {
		return nil, fmt.Errorf("safetensors: %s has an empty weight map", IndexFile)
	}
	s := &Sharded{Shard: idx.WeightMap, files: make(map[string]*File)}
	for name, shard := range idx.WeightMap {
		if name == "" || shard == "" || filepath.Base(shard) != shard || !filepath.IsLocal(shard) {
			s.Close()
			return nil, fmt.Errorf("safetensors: %s maps %q to %q, not a file in %s", IndexFile, name, shard, dir)
		}
		f := s.files[shard]
		if f == nil {
			f, err = Open(filepath.Join(dir, shard))
			if err != nil {
				s.Close()
				return nil, err
			}
			s.files[shard] = f
		}
		if _, ok := f.Info(name); !ok {
			s.Close()
			return nil, fmt.Errorf("safetensors: tensor %q is not in shard %s", name, shard)
		}
	}
	return s, nil
}

func openSingle(dir string) (*Sharded, error) {
	const single = "model.safetensors"
	f, err := Open(filepath.Join(dir, single))
	if err != nil {
		return nil, err
	}
	s := &Sharded{Shard: make(map[string]string, len(f.Tensors)), files: map[string]*File{single: f}}
	for _, t := range f.Tensors {
		s.Shard[t.Name] = single
	}
	return s, nil
}

// Info describes the named tensor. Its Offset is relative to the data section
// of the file that holds it.
func (s *Sharded) Info(name string) (TensorInfo, bool) {
	f := s.files[s.Shard[name]]
	if f == nil {
		return TensorInfo{}, false
	}
	return f.Info(name)
}

// Tensor reads the named tensor from the file that holds it.
func (s *Sharded) Tensor(name string) (Tensor, error) {
	f := s.files[s.Shard[name]]
	if f == nil {
		return Tensor{}, fmt.Errorf("safetensors: no tensor %q", name)
	}
	return f.Tensor(name)
}

// Close closes every file in the checkpoint.
func (s *Sharded) Close() error {
	var errs []error
	for _, f := range s.files {
		errs = append(errs, f.Close())
	}
	s.files = nil
	return errors.Join(errs...)
}
