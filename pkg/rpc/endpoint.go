// Package rpc holds what every CortexDB gRPC client needs beside the
// generated stubs in rpc/v1.
package rpc

import (
	"crypto/tls"
	"fmt"
	"net"
	"net/url"
	"strings"

	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
)

// Endpoint turns a brain address into a dial target and its transport.
//
//   - host:port, or http://host:port: plaintext, for loopback, the LAN or a
//     tailnet, where the bearer token is the access control.
//   - https://host[:port]: TLS checked against the system roots, port 443 by
//     default, for a brain published behind a TLS-terminating proxy such as
//     cortexdb.superleo.cn.
//
// A path after the host is refused rather than dropped: gRPC has no base
// path, and an address that silently lost one would reach a different server
// than the one written down.
func Endpoint(addr string) (string, credentials.TransportCredentials, error) {
	addr = strings.TrimSpace(addr)
	if !strings.Contains(addr, "://") {
		return addr, insecure.NewCredentials(), nil
	}
	u, err := url.Parse(addr)
	if err != nil {
		return "", nil, fmt.Errorf("cortexdb endpoint %q: %w", addr, err)
	}
	if u.Path != "" && u.Path != "/" {
		return "", nil, fmt.Errorf("cortexdb endpoint %q: gRPC has no path; use scheme://host[:port]", addr)
	}
	host := u.Hostname()
	if host == "" {
		return "", nil, fmt.Errorf("cortexdb endpoint %q has no host", addr)
	}
	port := u.Port()
	switch strings.ToLower(u.Scheme) {
	case "https":
		if port == "" {
			port = "443"
		}
		return net.JoinHostPort(host, port), credentials.NewTLS(&tls.Config{ServerName: host, MinVersion: tls.VersionTLS12}), nil
	case "http":
		if port == "" {
			port = "80"
		}
		return net.JoinHostPort(host, port), insecure.NewCredentials(), nil
	}
	return "", nil, fmt.Errorf("cortexdb endpoint %q: scheme must be http or https", addr)
}
