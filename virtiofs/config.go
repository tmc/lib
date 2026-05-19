package virtiofs

const (
	MaxTagLen        = 36
	DefaultQueueSize = 128
)

type Config struct {
	Tag       string
	NumQueues uint16
	QueueSize uint16
}

func DefaultConfig(tag string) Config {
	return Config{Tag: tag, NumQueues: 1, QueueSize: DefaultQueueSize}
}

func (c Config) Validate() error {
	if c.Tag == "" || len(c.Tag) > MaxTagLen {
		return ErrInvalidArgument
	}
	if c.NumQueues == 0 || c.QueueSize == 0 {
		return ErrInvalidArgument
	}
	if c.QueueSize&(c.QueueSize-1) != 0 {
		return ErrInvalidArgument
	}
	return nil
}
