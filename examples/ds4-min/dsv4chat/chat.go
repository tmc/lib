package dsv4chat

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// Message is a completed user or assistant turn. Assistant Tokens contain the
// exact generated IDs, excluding the stop token. Render must use these IDs
// rather than tokenize Text again; tokenization is not generally reversible.
type Message struct {
	Role   string
	Text   string
	Tokens []int
	// Prefix retains the exact assistant prefix used for this turn.
	Prefix []int
}

// Action describes a parsed command. Submit sends Text as a user message;
// Context replaces the state capacity while preserving evaluated history.
// Handled consumes the input without generating an assistant response.
type Action struct {
	Text    string
	Submit  bool
	Context int
	Handled bool
}

// Config supplies model-specific rendering, sampling, and decoding. Render
// receives completed history followed by the new user message and must append
// the assistant generation prefix. It owns system prompt and thinking policy.
// Sample must return a token ID or an error. Decode returns its printable bytes.
type Config struct {
	Backend dsv4.Backend
	Context int
	Stops   dsv4.StopSet
	Tokens  int
	Render  func([]Message) ([]int, error)
	Sample  func([]float32) (int, error)
	Decode  func(int) (string, error)
	Prompt  string
	// Command optionally expands commands. Command errors are reported to
	// Errors and leave the conversation usable. Context replacements replay
	// the evaluated timeline before releasing the old state.
	Command         func(string) (Action, error)
	Errors          io.Writer
	AssistantPrefix func() []int
	IsStop          func(int) bool
	ContextChanged  func(int)
	BeforeTurn      func() error
	// ConfigureState initializes each new request before replay. On error,
	// the candidate is closed and the existing request remains unchanged.
	ConfigureState func(dsv4.State) error
	// StateChanged observes the initial request and committed replacements.
	// The state is borrowed until replacement or Run returns; this callback
	// must not close it and cannot fail. Failed replacements are not published.
	StateChanged func(dsv4.State)
}

type configuredBackend struct {
	dsv4.Backend
	configure func(dsv4.State) error
	state     dsv4.State
}

func (b *configuredBackend) NewState(size int) (dsv4.State, error) {
	s, err := b.Backend.NewState(size)
	if err != nil || s == nil {
		return s, err
	}
	if b.configure != nil {
		if err := b.configure(s); err != nil {
			cleanupErr := s.Close()
			if cleanupErr != nil {
				owners := append(cleanupOwners(err), s)
				return nil, retainCleanup(errors.Join(err, cleanupErr), owners...)
			}
			return nil, err
		}
	}
	b.state = s
	return s, nil
}

func (b *configuredBackend) EvalBatch(s dsv4.State, tokens []int, position int) error {
	if batch, ok := b.Backend.(dsv4.BatchEvaluator); ok {
		return batch.EvalBatch(s, tokens, position)
	}
	for i, token := range tokens {
		if err := b.Eval(s, token, position+i); err != nil {
			return err
		}
	}
	return nil
}

// Run reads one user message per line and writes generated text. Empty lines
// are ignored; /reset clears history and state; /quit and /exit end the loop.
// EOF processes a final unterminated line, then closes the state. Any error
// closes the state and returns without committing a partial assistant turn.
// Cancellation is checked between reads and tokens. To interrupt a blocked
// Read, the caller must close its reader. Run does not close reader or writer.
// If cleanup fails, the returned error may carry an io.Closer discoverable
// with errors.As. Its Close retries every unresolved state release; callers
// must not resume generation on those terminal requests.
func Run(ctx context.Context, r io.Reader, w io.Writer, cfg Config) (err error) {
	if ctx == nil || r == nil || w == nil || cfg.Render == nil || cfg.Sample == nil || cfg.Decode == nil || cfg.Tokens <= 0 {
		return errors.New("invalid chat configuration")
	}
	if cfg.Backend == nil {
		return errors.New("invalid chat backend")
	}
	backend := &configuredBackend{Backend: cfg.Backend, configure: cfg.ConfigureState}
	session, err := dsv4.NewSession(backend, cfg.Context, cfg.Stops)
	if err != nil {
		return retainCleanup(err, cleanupOwners(err)...)
	}
	var pending []io.Closer
	var cleanupErrors []error
	defer func() {
		pending = append(cleanupOwners(err), pending...)
		if closeErr := session.Close(); closeErr != nil {
			pending = append(pending, session)
			cleanupErrors = append(cleanupErrors, closeErr)
		}
		err = retainCleanup(errors.Join(append([]error{err}, cleanupErrors...)...), pending...)
	}()
	if cfg.StateChanged != nil {
		cfg.StateChanged(backend.state)
	}
	reader := bufio.NewReader(r)
	capacity := cfg.Context
	diagnostics := cfg.Errors
	if diagnostics == nil {
		diagnostics = w
	}
	var history []Message
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		if _, err := io.WriteString(w, cfg.Prompt); err != nil {
			return fmt.Errorf("write prompt: %w", err)
		}
		line, readErr := reader.ReadString('\n')
		if readErr != nil && readErr != io.EOF {
			return fmt.Errorf("read message: %w", readErr)
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		line = strings.TrimSpace(line)
		submitted := false
		if cfg.Command != nil && strings.HasPrefix(strings.TrimSpace(line), "/") {
			action, commandErr := cfg.Command(strings.TrimSpace(line))
			if commandErr == nil && action.Context != 0 {
				replacement, replaceErr := dsv4.NewSession(backend, action.Context, cfg.Stops)
				if replaceErr == nil && len(session.Tokens()) > 0 {
					replaceErr = replacement.Sync(session.Tokens())
				}
				if replaceErr != nil {
					if replacement != nil {
						closeErr := replacement.Close()
						if closeErr != nil {
							pending = append(pending, replacement)
							cleanupErrors = append(cleanupErrors, errors.Join(replaceErr, closeErr))
						}
						replaceErr = errors.Join(replaceErr, closeErr)
					}
					commandErr = replaceErr
				} else {
					if closeErr := session.Close(); closeErr != nil {
						cleanupErr := replacement.Close()
						if cleanupErr != nil {
							pending = append(pending, replacement)
							cleanupErrors = append(cleanupErrors, cleanupErr)
						}
						return errors.Join(closeErr, cleanupErr)
					}
					session = replacement
					if cfg.StateChanged != nil {
						cfg.StateChanged(backend.state)
					}
					capacity = action.Context
					if cfg.ContextChanged != nil {
						cfg.ContextChanged(capacity)
					}
				}
			}
			if commandErr != nil {
				owners := cleanupOwners(commandErr)
				if len(owners) > 0 {
					pending = append(pending, owners...)
					cleanupErrors = append(cleanupErrors, commandErr)
				}
				if _, err := fmt.Fprintf(diagnostics, "chat: %v\n", commandErr); err != nil {
					return err
				}
				if readErr == io.EOF {
					return nil
				}
				continue
			}
			if action.Handled {
				if readErr == io.EOF {
					return nil
				}
				continue
			}
			if action.Submit {
				line = action.Text
				submitted = true
			}
		}
		switch strings.TrimSpace(line) {
		case "/quit", "/exit":
			return nil
		case "/reset":
			if err := session.Reset(); err != nil {
				return err
			}
			history = nil
		case "":
			if !submitted {
				break
			}
			fallthrough
		default:
			turns := append(append([]Message(nil), history...), Message{Role: "user", Text: line})
			prompt, err := cfg.Render(cloneMessages(turns))
			if err != nil {
				return fmt.Errorf("render message: %w", err)
			}
			if err := session.Sync(prompt); err != nil {
				return err
			}
			if cfg.BeforeTurn != nil {
				if err := cfg.BeforeTurn(); err != nil {
					return err
				}
			}
			answer := Message{Role: "assistant"}
			if cfg.AssistantPrefix != nil {
				answer.Prefix = append([]int(nil), cfg.AssistantPrefix()...)
			}
			// Upstream reserves one token beyond the generation frontier.
			budget := min(cfg.Tokens, max(0, capacity-len(session.Tokens())-1))
			for i := 0; i < budget; i++ {
				if err := ctx.Err(); err != nil {
					return err
				}
				token, err := cfg.Sample(session.Logits())
				if err != nil {
					return fmt.Errorf("sample token: %w", err)
				}
				if err := ctx.Err(); err != nil {
					return err
				}
				if cfg.IsStop != nil && cfg.IsStop(token) {
					break
				}
				stop, err := session.Accept(token)
				if err != nil {
					return err
				}
				if stop {
					break
				}
				piece, err := cfg.Decode(token)
				if err != nil {
					return fmt.Errorf("decode token: %w", err)
				}
				if _, err := io.WriteString(w, piece); err != nil {
					return fmt.Errorf("write token: %w", err)
				}
				answer.Text += piece
				answer.Tokens = append(answer.Tokens, token)
			}
			if _, err := io.WriteString(w, "\n"); err != nil {
				return err
			}
			history = append(turns, answer)
		}
		if readErr == io.EOF {
			return nil
		}
	}
}

func cloneMessages(messages []Message) []Message {
	out := append([]Message(nil), messages...)
	for i := range out {
		out[i].Tokens = append([]int(nil), out[i].Tokens...)
		out[i].Prefix = append([]int(nil), out[i].Prefix...)
	}
	return out
}
