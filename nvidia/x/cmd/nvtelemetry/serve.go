package main

import (
	"context"
	"crypto/subtle"
	"errors"
	"flag"
	"io"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

func runServe(args []string, stdout, stderr io.Writer) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	return runServeContext(ctx, args, stdout, stderr)
}

func runServeContext(parent context.Context, args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	configFile := fs.String("config", "", "validated JSON configuration file")
	metricsConfig := fs.String("otlp-metrics-config", "", "optional transport-only OTLP metrics endpoint JSON")
	logsConfig := fs.String("otlp-logs-config", "", "optional transport-only OTLP diagnostic logs endpoint JSON")
	counterStart := fs.Int64("counter-start-unix-nanos", 0, "explicit counter epoch start for OTLP metrics (0 refuses counters)")
	address := fs.String("listen", "127.0.0.1:9464", "HTTP listening address")
	interval := fs.Duration("interval", time.Second, "sampling interval")
	duration := fs.Duration("duration", 0, "maximum serving duration (0 until signal or budget)")
	maxAge := fs.Duration("max-age", 5*time.Second, "maximum snapshot publication age")
	budget := fs.Int("budget", 100000, "maximum stream records before stopping")
	managedConfig := fs.String("managed-config", "", "explicit managed endpoint configuration JSON file")
	mock := fs.Bool("mock", false, "serve explicitly synthetic mock readings")
	uuids := fs.String("device-uuids", "", "exact UUID allowlist (required for live NVML)")
	tokenFile := fs.String("auth-token-file", "", "file containing an HTTP bearer token")
	cert := fs.String("tls-cert", "", "TLS certificate file")
	key := fs.String("tls-key", "", "TLS private key file")
	remote := fs.Bool("allow-remote", false, "allow a non-loopback listener with TLS and bearer authentication")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return errors.New("invalid serving options or bounds: extraneous arguments")
	}

	var parsed *ParsedConfig
	if *configFile != "" {
		var err error
		parsed, err = ParseConfigFile(*configFile)
		if err != nil {
			return err
		}
	} else {
		// Enforce bounds matching serve CLI flag requirements
		if *interval <= 0 || *duration < 0 || *maxAge <= 0 || *budget < 1 || *budget > 1<<20 {
			return errors.New("invalid serving options or bounds")
		}

		raw := ServeConfig{
			Listen:                *address,
			Interval:              interval.String(),
			MaxAge:                maxAge.String(),
			Duration:              duration.String(),
			Budget:                *budget,
			ShutdownTimeout:       "5s",
			Mock:                  *mock,
			ManagedConfig:         *managedConfig,
			CounterStartUnixNanos: *counterStart,
			OTLPMetricsConfig:     *metricsConfig,
			OTLPLogsConfig:        *logsConfig,
			AuthTokenFile:         *tokenFile,
			TLSCert:               *cert,
			TLSKey:                *key,
			AllowRemote:           *remote,
		}
		if *uuids != "" {
			raw.DeviceUUIDs = strings.Split(*uuids, ",")
		}

		var err error
		parsed, err = validateAndBuildParsedConfig(raw)
		if err != nil {
			return err
		}
	}

	agent, err := NewAgent(parsed, *configFile, stdout, stderr)
	if err != nil {
		return err
	}
	return agent.Start(parent)
}

func readBearerToken(path string) (string, error) {
	if path == "" {
		return "", nil
	}
	f, err := os.Open(path)
	if err != nil {
		return "", errors.New("open bearer token file failed")
	}
	defer f.Close()
	raw, err := io.ReadAll(io.LimitReader(f, 4097))
	if err != nil || len(raw) > 4096 {
		return "", errors.New("invalid bearer token file")
	}
	token := strings.TrimSpace(string(raw))
	for i := range token {
		if token[i] < 33 || token[i] > 126 {
			return "", errors.New("invalid bearer token")
		}
	}
	if token == "" || strings.ContainsAny(token, "\r\n\t ") {
		return "", errors.New("invalid bearer token")
	}
	return token, nil
}

func bearerHandler(next http.Handler, token string) http.Handler {
	if token == "" {
		return next
	}
	expected := []byte("Bearer " + token)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if subtle.ConstantTimeCompare([]byte(r.Header.Get("Authorization")), expected) != 1 {
			w.Header().Set("WWW-Authenticate", "Bearer")
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}
