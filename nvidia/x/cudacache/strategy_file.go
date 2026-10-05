package cudacache

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
)

// SaveFile publishes a bounded snapshot by a same-directory atomic rename.
// Parent directories must already exist. It never writes a partial catalog.
func (k *StrategyKnowledgeBase) SaveFile(path string) error {
	if path == "" {
		return fmt.Errorf("cudacache: missing strategy path")
	}
	var b bytes.Buffer
	if err := k.Write(&b); err != nil {
		return err
	}
	if b.Len() > strategyBytes {
		return fmt.Errorf("cudacache: strategy byte limit")
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".strategy-*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	if _, err = f.Write(b.Bytes()); err != nil {
		f.Close()
		return err
	}
	if err = f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// LoadFile admits only a bounded regular file, then applies Load's atomic
// validation. Corrupt catalog files remain available for diagnosis.
func (k *StrategyKnowledgeBase) LoadFile(path string) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() || info.Size() > strategyBytes {
		return fmt.Errorf("cudacache: strategy file type or size refused")
	}
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil {
		return err
	}
	if !opened.Mode().IsRegular() || !os.SameFile(info, opened) {
		return fmt.Errorf("cudacache: strategy file changed")
	}
	return k.Load(f)
}
