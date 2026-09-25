package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"time"
)

const version = "3.3.2-go"

// options are the command-line settings.
type options struct {
	delay      time.Duration // 0 if not set
	snapshot   bool
	loop       bool
	noPlot     bool
	noProcs    bool
	reverse    bool
	noColor    bool
	fahrenheit bool
	infoBar    bool
	ext        bool          // -x: the extensions nvtop does not have
	record     string        // file to record to; "-" records to stdout without the interface
	replay     string        // recording to replay
	remote     string        // comma-separated hosts to show instead of this one
	remoteCmd  string        // the nvtop to run on them, if not nvtop
	encodeHide time.Duration // negative keeps the meters shown
}

const helpText = `Available options:
  -d --delay        : Select the refresh rate (1 == 0.1s)
  -v --version      : Print the version and exit
  -p --no-plot      : Disable bar plot
  -P --no-processes : Disable process list
  -r --reverse-abs  : Reverse abscissa: plot the recent data left and older on the right
  -C --no-color     : No colors
  -f --freedom-unit : Use fahrenheit
  -i --gpu-info     : Show bar with additional GPU parameters
  -E --encode-hide  : Set encode/decode auto hide time in seconds (default 30s, negative = always on screen)
  -h --help         : Print help and exit
  -s --snapshot     : Output the current gpu stats without ncurses (useful for scripting)
  -l --loop         : Output the current gpu stats without ncurses in a loop
`

// extHelpText lists the flags nvtop does not have.
const extHelpText = `Extensions (not in nvtop):
  -x --extensions   : Show clock throttle reasons, energy, NVML events and process details
  -record FILE      : With -x, record every refresh to FILE as JSON lines (- for stdout, without the interface)
  -replay FILE      : With -x, replay a recording (Space pauses, n steps)
  -remote HOSTS     : With -x, show the GPUs of these comma-separated hosts, over ssh
  -remote-cmd CMD   : The nvtop to run on the remote hosts (default nvtop)
`

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

// run runs nvtop with the given arguments and returns the exit status.
func run(args []string, stdout, stderr io.Writer) int {
	opts, status, ok := parseFlags(args, stdout, stderr)
	if !ok {
		return status
	}

	var src source
	var h *host
	switch {
	case opts.replay != "":
		r, err := openReplay(opts.replay)
		if err != nil {
			fmt.Fprintf(stderr, "nvtop: %v\n", err)
			return 1
		}
		src = r
	case opts.remote != "":
		r, err := openRemote(strings.Split(opts.remote, ","), sshDialer(opts.remoteCmd, opts.delay))
		if err != nil {
			fmt.Fprintf(stderr, "nvtop: %v\n", err)
			return 1
		}
		src = r
	default:
		h = newHost(os.DirFS("/proc"))
		n, err := openNVML(h)
		if err != nil {
			fmt.Fprintf(stderr, "nvtop: %v\n", err)
			return 1
		}
		if opts.ext {
			n.extend()
		}
		src = n
	}
	if len(src.devices()) == 0 {
		src.close()
		fmt.Fprintln(stdout, "No GPU to monitor.")
		return 0
	}
	err := serve(src, h, opts, stdout)
	if cerr := src.close(); err == nil {
		err = cerr
	}
	if err != nil {
		fmt.Fprintf(stderr, "nvtop: %v\n", err)
		return 1
	}
	return 0
}

// serve shows, prints or records what src reads, as opts say, until the
// user quits or nvtop is signaled.
func serve(src source, h *host, opts options, stdout io.Writer) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGQUIT, syscall.SIGTERM)
	defer stop()
	switch {
	case opts.record == "-":
		return record(ctx, stdout, src, h, opts)
	case opts.snapshot || opts.loop:
		return snapshots(ctx, stdout, src, h, opts)
	}
	var rec *recorder
	if opts.record != "" {
		f, err := os.Create(opts.record)
		if err != nil {
			return err
		}
		defer f.Close()
		rec = newRecorder(f)
	}
	return runTUI(ctx, src, h, opts, rec)
}

// parseFlags parses args. If the command should stop, ok is false and
// status is its exit status.
func parseFlags(args []string, stdout, stderr io.Writer) (opts options, status int, ok bool) {
	fs := flag.NewFlagSet("nvtop", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "nvtop version %s\n%s%s", version, helpText, extHelpText)
	}
	boolVar := func(p *bool, short, long, usage string) {
		fs.BoolVar(p, short, false, usage)
		fs.BoolVar(p, long, false, usage)
	}
	delay := func(s string) error {
		n, err := strconv.ParseInt(s, 0, 64)
		if err != nil {
			return errors.New("the delay must be a positive value representing tenths of seconds")
		}
		if n < 0 {
			return errors.New("a negative delay requires a time machine")
		}
		opts.delay = time.Duration(min(max(n, 1), 999)) * 100 * time.Millisecond
		return nil
	}
	fs.Func("d", "refresh delay in tenths of a second", delay)
	fs.Func("delay", "refresh delay in tenths of a second", delay)
	var showVersion, help bool
	boolVar(&help, "h", "help", "print help and exit")
	boolVar(&showVersion, "v", "version", "print the version and exit")
	boolVar(&opts.snapshot, "s", "snapshot", "print one JSON snapshot and exit")
	boolVar(&opts.loop, "l", "loop", "print a JSON snapshot every refresh")
	boolVar(&opts.noPlot, "p", "no-plot", "do not draw the plot")
	boolVar(&opts.noProcs, "P", "no-processes", "do not list processes")
	boolVar(&opts.reverse, "r", "reverse-abs", "plot recent data on the left")
	boolVar(&opts.noColor, "C", "no-color", "do not use color")
	fs.BoolVar(&opts.noColor, "no-colour", false, "do not use color")
	boolVar(&opts.fahrenheit, "f", "freedom-unit", "show temperatures in Fahrenheit")
	boolVar(&opts.infoBar, "i", "gpu-info", "show a bar of more device information")
	boolVar(&opts.ext, "x", "extensions", "show what nvtop does not")
	fs.StringVar(&opts.record, "record", "", "with -x, record every refresh to `file`")
	fs.StringVar(&opts.replay, "replay", "", "with -x, replay a recording from `file`")
	fs.StringVar(&opts.remote, "remote", "", "with -x, show the GPUs of these comma-separated `hosts`")
	fs.StringVar(&opts.remoteCmd, "remote-cmd", "", "the nvtop `command` to run on the remote hosts")
	opts.encodeHide = 30 * time.Second
	hide := func(s string) error {
		f, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return fmt.Errorf("invalid format for encode/decode hide time: %s", s)
		}
		opts.encodeHide = time.Duration(f * float64(time.Second))
		return nil
	}
	fs.Func("E", "encode/decode auto-hide time in seconds", hide)
	fs.Func("encode-hide", "encode/decode auto-hide time in seconds", hide)

	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return opts, 0, false
		}
		return opts, 2, false
	}
	if help {
		fmt.Fprintf(stdout, "nvtop version %s\n%s%s", version, helpText, extHelpText)
		return opts, 0, false
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "nvtop: unexpected argument %q\n", fs.Arg(0))
		fs.Usage()
		return opts, 2, false
	}
	if showVersion {
		fmt.Fprintf(stdout, "nvtop version %s\n", version)
		return opts, 0, false
	}
	var bad string
	switch {
	case !opts.ext && opts.record != "":
		bad = "-record needs -x"
	case !opts.ext && opts.replay != "":
		bad = "-replay needs -x"
	case !opts.ext && opts.remote != "":
		bad = "-remote needs -x"
	case opts.remoteCmd != "" && opts.remote == "":
		bad = "-remote-cmd needs -remote"
	case opts.record != "" && opts.replay != "":
		bad = "-record and -replay cannot be used together"
	case opts.remote != "" && opts.replay != "":
		bad = "-remote and -replay cannot be used together"
	case opts.remote != "" && slices.Contains(strings.Split(opts.remote, ","), ""):
		bad = "-remote needs host names separated by commas"
	case opts.record != "" && (opts.snapshot || opts.loop):
		bad = "-record cannot be used with -s or -l"
	}
	if bad != "" {
		fmt.Fprintf(stderr, "nvtop: %s\n", bad)
		return opts, 2, false
	}
	return opts, 0, true
}

// snapshots prints a snapshot of every device after one refresh interval,
// and with -l keeps printing one every interval until ctx is done. The
// first refresh starts the counters that only a second one can turn into
// rates.
//
// When replaying, it prints the first frame, and with -l every frame
// after it, as fast as it can.
func snapshots(ctx context.Context, w io.Writer, src source, h *host, opts options) error {
	devs := src.devices()
	if p, ok := src.(player); ok {
		for {
			gather(src, h, devs, true)
			if err := writeSnapshot(w, devs, opts.fahrenheit, opts.noProcs); err != nil {
				return err
			}
			if !opts.loop || ctx.Err() != nil || !p.step() {
				return nil
			}
		}
	}
	delay := opts.delay
	if delay == 0 {
		delay = 100 * time.Millisecond
	}
	gather(src, h, devs, true)
	t := time.NewTicker(delay)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return nil
		case <-t.C:
		}
		gather(src, h, devs, true)
		if err := writeSnapshot(w, devs, opts.fahrenheit, opts.noProcs); err != nil {
			return err
		}
		if !opts.loop {
			return nil
		}
	}
}
