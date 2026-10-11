//go:build !linux || (!amd64 && !arm64)

package bpfload

import "os"

func CreateStackMap(int, int) (*Object, error) { return nil, ErrUnsupported }

func InspectMap(int) (MapInfo, error) { return MapInfo{}, ErrUnsupported }

func CreateRingMap(int) (*Object, error)  { return nil, ErrUnsupported }
func (o *Object) File() (*os.File, error) { return nil, ErrUnsupported }

func CreateMap(s MapSpec) (*Object, error) {
	if err := validateMap(s); err != nil {
		return nil, err
	}
	return nil, ErrUnsupported
}
func (o *Object) Pin(string) error { return ErrUnsupported }

func closeFD(int) error { return ErrUnsupported }
func Load(s ProgramSpec) (*Object, string, error) {
	if _, err := validateSpec(s); err != nil {
		return nil, "", err
	}
	return nil, "", ErrUnsupported
}
func LoadBTF([]byte, int) (*Object, string, error)    { return nil, "", ErrUnsupported }
func OpenMap(string, bool) (*Object, MapInfo, error)  { return nil, MapInfo{}, ErrUnsupported }
func (o *Object) MapInfo() (MapInfo, error)           { return MapInfo{}, ErrUnsupported }
func (o *Object) Lookup([]byte, []byte) error         { return ErrUnsupported }
func (o *Object) Update([]byte, []byte, uint64) error { return ErrUnsupported }
