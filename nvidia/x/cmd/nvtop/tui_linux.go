package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
)

// runTUI runs the interactive interface on the terminal until the user
// quits or ctx is done.
func runTUI(ctx context.Context, src source, h *host, opts options) error {
	t, err := openTerminal(os.Stdin, os.Stdout)
	if err != nil {
		return err
	}
	defer t.restore()

	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGWINCH, syscall.SIGTSTP, syscall.SIGCONT)
	defer signal.Stop(sigs)

	keys := make(chan []key)
	go readKeys(os.Stdin, keys)

	devs := src.devices()
	u := newUI(devs, opts, os.Getpid(), time.Now)
	var scr *screen
	var buf []byte
	refresh := time.NewTimer(0)
	defer refresh.Stop()
	for !u.quit {
		select {
		case <-ctx.Done():
			return nil
		case <-refresh.C:
			gather(src, h, devs, !u.frozen())
			u.record()
			refresh.Reset(u.interval())
		case ks, ok := <-keys:
			if !ok {
				return nil
			}
			for _, k := range ks {
				u.handle(k)
			}
			if u.quit {
				return nil
			}
		case s := <-sigs:
			switch s {
			case syscall.SIGTSTP:
				// Leave the terminal as the shell expects it, then
				// stop as the default action would have. The stop
				// takes effect before kill returns, so on return the
				// process has been continued.
				t.restore()
				syscall.Kill(os.Getpid(), syscall.SIGSTOP)
				fallthrough
			case syscall.SIGCONT:
				// Back from a stop: the shell has reset the
				// terminal.
				if err := t.enter(); err != nil {
					return err
				}
			}
			u.redraw = true
		}
		w, hgt, err := t.size()
		if err != nil {
			return err
		}
		if scr == nil || u.redraw || scr.w != w || scr.h != hgt {
			scr = newScreen(w, hgt, !opts.noColor)
			u.redraw = false
		}
		u.render(scr)
		buf = scr.flush(buf[:0])
		if _, err := t.out.Write(buf); err != nil {
			return fmt.Errorf("write terminal: %w", err)
		}
	}
	return nil
}

// readKeys sends the keys read from r to keys until r fails.
func readKeys(r io.Reader, keys chan<- []key) {
	b := make([]byte, 256)
	for {
		n, err := r.Read(b)
		if n > 0 {
			keys <- parseKeys(b[:n])
		}
		if err != nil {
			close(keys)
			return
		}
	}
}

// A terminal is a terminal in the mode the interface needs: input
// unbuffered and not echoed, output on the alternate screen with the
// cursor hidden.
type terminal struct {
	in, out *os.File
	saved   unix.Termios
}

func openTerminal(in, out *os.File) (*terminal, error) {
	saved, err := unix.IoctlGetTermios(int(in.Fd()), unix.TCGETS)
	if err != nil {
		return nil, errors.New("standard input is not a terminal; use -s for a snapshot")
	}
	t := &terminal{in: in, out: out, saved: *saved}
	if err := t.enter(); err != nil {
		return nil, err
	}
	return t, nil
}

// enter puts the terminal in the mode the interface needs.
func (t *terminal) enter() error {
	if err := t.makeRaw(); err != nil {
		return err
	}
	t.out.WriteString("\x1b[?1049h\x1b[?25l")
	return nil
}

// makeRaw sets the mode ncurses's cbreak and noecho set: no line
// buffering and no echo, with signal keys still working.
func (t *terminal) makeRaw() error {
	tio := t.saved
	tio.Lflag &^= unix.ICANON | unix.ECHO
	tio.Cc[unix.VMIN] = 1
	tio.Cc[unix.VTIME] = 0
	if err := unix.IoctlSetTermios(int(t.in.Fd()), unix.TCSETS, &tio); err != nil {
		return fmt.Errorf("set terminal mode: %w", err)
	}
	return nil
}

// restore puts the terminal back as openTerminal found it.
func (t *terminal) restore() {
	t.out.WriteString("\x1b[0m\x1b[?25h\x1b[?1049l")
	unix.IoctlSetTermios(int(t.in.Fd()), unix.TCSETS, &t.saved)
}

func (t *terminal) size() (w, h int, err error) {
	ws, err := unix.IoctlGetWinsize(int(t.out.Fd()), unix.TIOCGWINSZ)
	if err != nil {
		return 0, 0, fmt.Errorf("get terminal size: %w", err)
	}
	return int(ws.Col), int(ws.Row), nil
}
