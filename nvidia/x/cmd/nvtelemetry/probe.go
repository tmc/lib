package main

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	probeTimeout     = 2 * time.Second
	probeDialTimeout = time.Second
	probeMaxCABytes  = 1 << 20
)

type probeRoute struct {
	path string
	body string
}

func runProbe(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("probe", flag.ContinueOnError)
	fs.SetOutput(stderr)
	kind := fs.String("kind", "", "liveness or readiness")
	configFile := fs.String("config", "", "serve JSON configuration file")
	caFile := fs.String("ca-file", "", "PEM CA roots used to verify the local HTTPS listener")
	serverName := fs.String("server-name", "", "certificate DNS name or IP address")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 || *configFile == "" || *caFile == "" || !validProbeServerName(*serverName) {
		return errors.New("invalid probe options")
	}

	route, ok := probeRoutes[*kind]
	if !ok {
		return errors.New("invalid probe kind")
	}
	config, err := ParseConfigFile(*configFile)
	if err != nil {
		return errors.New("invalid probe configuration")
	}
	if err := probeEndpoint(context.Background(), config, *caFile, *serverName, route); err != nil {
		return err
	}
	return nil
}

var probeRoutes = map[string]probeRoute{
	"liveness":  {path: "/healthz", body: "ok\n"},
	"readiness": {path: "/readyz", body: "ready\n"},
}

func probeEndpoint(parent context.Context, config *ParsedConfig, caFile, serverName string, route probeRoute) error {
	if config == nil || config.BearerToken == "" || config.TLSCert == "" || config.TLSKey == "" || caFile == "" || !validProbeServerName(serverName) {
		return errors.New("probe requires configured bearer authentication and verified TLS")
	}
	address, err := probeAddress(config.Listen)
	if err != nil {
		return errors.New("probe listener address is invalid")
	}
	roots, err := loadProbeRoots(caFile)
	if err != nil {
		return errors.New("probe CA file is unavailable or invalid")
	}

	transport := &http.Transport{
		Proxy:                  nil,
		DialContext:            (&net.Dialer{Timeout: probeDialTimeout}).DialContext,
		TLSClientConfig:        &tls.Config{MinVersion: tls.VersionTLS12, RootCAs: roots, ServerName: serverName},
		TLSHandshakeTimeout:    probeDialTimeout,
		ResponseHeaderTimeout:  probeDialTimeout,
		MaxResponseHeaderBytes: 16 << 10,
		DisableKeepAlives:      true,
	}
	defer transport.CloseIdleConnections()
	client := &http.Client{
		Transport: transport,
		Timeout:   probeTimeout,
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}

	ctx, cancel := context.WithTimeout(parent, probeTimeout)
	defer cancel()
	u := url.URL{Scheme: "https", Host: address, Path: route.path}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return errors.New("probe request could not be created")
	}
	request.Header.Set("Authorization", "Bearer "+config.BearerToken)
	response, err := client.Do(request)
	if err != nil {
		return errors.New("probe request failed")
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return errors.New("probe endpoint is unhealthy")
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, int64(len(route.body)+1)))
	if err != nil || string(body) != route.body {
		return errors.New("probe endpoint response is invalid")
	}
	return nil
}

func loadProbeRoots(path string) (*x509.CertPool, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	pem, err := io.ReadAll(io.LimitReader(f, probeMaxCABytes+1))
	if err != nil || len(pem) == 0 || len(pem) > probeMaxCABytes {
		return nil, errors.New("invalid CA data")
	}
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(pem) {
		return nil, errors.New("invalid CA data")
	}
	return roots, nil
}

func probeAddress(listen string) (string, error) {
	host, portText, err := net.SplitHostPort(listen)
	if err != nil || host == "" && portText == "" {
		return "", fmt.Errorf("invalid listener")
	}
	port, err := strconv.ParseUint(portText, 10, 16)
	if err != nil || port == 0 {
		return "", fmt.Errorf("invalid listener port")
	}
	if host == "" {
		host = "127.0.0.1"
	} else {
		ip := net.ParseIP(host)
		if ip == nil {
			return "", fmt.Errorf("listener host must be an IP address")
		}
		if ip.IsUnspecified() {
			if ip.To4() != nil {
				host = "127.0.0.1"
			} else {
				host = "::1"
			}
		} else if !ip.IsLoopback() && !probeLocalIP(ip) {
			return "", fmt.Errorf("listener host is not assigned locally")
		}
	}
	return net.JoinHostPort(host, strconv.FormatUint(port, 10)), nil
}

func probeLocalIP(ip net.IP) bool {
	addresses, err := net.InterfaceAddrs()
	if err != nil {
		return false
	}
	for _, address := range addresses {
		var candidate net.IP
		switch addr := address.(type) {
		case *net.IPNet:
			candidate = addr.IP
		case *net.IPAddr:
			candidate = addr.IP
		}
		if candidate != nil && candidate.Equal(ip) {
			return true
		}
	}
	return false
}

func validProbeServerName(name string) bool {
	if name == "" || len(name) > 253 || strings.TrimSpace(name) != name {
		return false
	}
	if net.ParseIP(name) != nil {
		return true
	}
	for _, label := range strings.Split(name, ".") {
		if len(label) == 0 || len(label) > 63 || label[0] == '-' || label[len(label)-1] == '-' {
			return false
		}
		for _, c := range label {
			if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-') {
				return false
			}
		}
	}
	return true
}
