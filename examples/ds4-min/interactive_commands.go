package main

import (
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/tmc/lib/examples/ds4-min/dsv4chat"
)

const interactiveHelp = "Commands:\n  /help          Show this help.\n  /think         Use normal thinking.\n  /think-max     Use maximum thinking (requires ctx >= 393216).\n  /nothink       Disable thinking.\n  /ctx N         Replace context capacity, preserving history.\n  /steer F       Query or set the live FFN steering scale.\n  /read FILE     Submit a text file.\n  /reset         Clear the conversation.\n  /quit, /exit   Leave the prompt.\n"

type steeringControl interface {
	steeringFFN() float32
	setSteeringFFN(float32) error
}

type interactiveCommands struct {
	mode          ThinkMode
	context       int
	output        io.Writer
	maxTokenBytes int
	steering      steeringControl
}

func (c *interactiveCommands) effectiveMode() ThinkMode {
	if c.mode == ThinkMax && c.context < maxThinkContext {
		return ThinkHigh
	}
	return c.mode
}

func (c *interactiveCommands) command(line string) (dsv4chat.Action, error) {
	name, arg, _ := strings.Cut(line, " ")
	// Match upstream whitespace separators, including tabs.
	if i := strings.IndexAny(line, " \t\r\n"); i >= 0 {
		name = line[:i]
		arg = strings.TrimSpace(line[i:])
	}
	done := dsv4chat.Action{Handled: true}
	if arg != "" && (name == "/help" || name == "/think-max" || name == "/nothink" || name == "/quit" || name == "/exit" || name == "/reset") {
		return done, fmt.Errorf("unknown command: %s; type /help for commands", line)
	}
	switch name {
	case "/help":
		_, err := io.WriteString(c.output, interactiveHelp)
		return done, err
	case "/think", "/think-max", "/nothink":
		if arg != "" {
			return done, errors.New("/think N requires V4.1 and a level from 0 to 100")
		}
		label := "high"
		c.mode = ThinkHigh
		if name == "/nothink" {
			c.mode = ThinkNone
			label = "none"
		}
		if name == "/think-max" {
			c.mode = ThinkMax
			label = "max"
			if c.effectiveMode() == ThinkHigh {
				label = "high (ctx below 393216)"
			}
		}
		_, err := fmt.Fprintf(c.output, "Thinking mode: %s.\n", label)
		return done, err
	case "/ctx":
		size, err := strconv.Atoi(arg)
		if err != nil || size <= 1 {
			return done, errors.New("/ctx needs an integer greater than one")
		}
		done.Context = size
		return done, nil
	case "/steer":
		if c.steering == nil {
			return done, errors.New("live steering is not supported by this request")
		}
		if arg != "" {
			scale, err := strconv.ParseFloat(arg, 32)
			if err != nil || steeringScale(float32(scale)) != nil {
				return done, errors.New("/steer needs a finite scale between -100 and 100")
			}
			if err := c.steering.setSteeringFFN(float32(scale)); err != nil {
				return done, err
			}
		}
		_, err := fmt.Fprintf(c.output, "Steering FFN: %g.\n", c.steering.steeringFFN())
		return done, err
	case "/read":
		if arg == "" {
			return done, errors.New("/read needs a file path")
		}
		// A raw BPE token cannot cover more source bytes than the vocabulary
		// spelling's byte length. This is a conservative context admission
		// bound, not a truncation policy; actual token admission follows render.
		if c.context <= 1 || int64(c.context) > math.MaxInt64/int64(max(1, c.maxTokenBytes)) {
			return done, errors.New("invalid input capacity")
		}
		data, err := readInteractiveText(arg, int64(c.context)*int64(max(1, c.maxTokenBytes)))
		if err != nil {
			return done, fmt.Errorf("read prompt file: %w", err)
		}
		if len(data) >= 8 && string(data[:8]) == "\x89PNG\r\n\x1a\n" || len(data) >= 2 && data[0] == 0xff && data[1] == 0xd8 {
			return done, errors.New("image input requires the vision backend")
		}
		if i := strings.IndexByte(string(data), 0); i >= 0 {
			data = data[:i]
		}
		return dsv4chat.Action{Text: string(data), Submit: true}, nil
	case "/reset", "/quit", "/exit":
		return dsv4chat.Action{}, nil
	default:
		return done, fmt.Errorf("unknown command: %s; type /help for commands", line)
	}
}

func readInteractiveText(path string, limit int64) (data []byte, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { err = errors.Join(err, f.Close()) }()
	// Stop at NUL as upstream's C string consumer does. Read incrementally
	// so a short C-string prefix does not force allocation of trailing bytes.
	var block [4096]byte
	for {
		n, readErr := f.Read(block[:])
		part := block[:n]
		if i := strings.IndexByte(string(part), 0); i >= 0 {
			part = part[:i]
			readErr = io.EOF
		}
		if int64(len(data))+int64(len(part)) > limit {
			return nil, errors.New("prompt file exceeds context byte admission")
		}
		data = append(data, part...)
		if readErr == io.EOF {
			return data, nil
		}
		if readErr != nil {
			return nil, readErr
		}
	}
}
