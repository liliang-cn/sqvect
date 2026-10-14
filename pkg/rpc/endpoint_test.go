package rpc

import "testing"

func TestEndpointTargetsAndTransport(t *testing.T) {
	for _, c := range []struct {
		in, target, protocol string
	}{
		{"192.168.123.253:47821", "192.168.123.253:47821", "insecure"},
		{" 127.0.0.1:47821 ", "127.0.0.1:47821", "insecure"},
		{"http://10.0.0.5:47821", "10.0.0.5:47821", "insecure"},
		{"https://cortexdb.superleo.cn", "cortexdb.superleo.cn:443", "tls"},
		{"https://cortexdb.superleo.cn/", "cortexdb.superleo.cn:443", "tls"},
		{"https://brain.example.com:8443", "brain.example.com:8443", "tls"},
	} {
		target, creds, err := Endpoint(c.in)
		if err != nil {
			t.Fatalf("%q: %v", c.in, err)
		}
		if target != c.target || creds.Info().SecurityProtocol != c.protocol {
			t.Fatalf("%q: got %s over %s, want %s over %s", c.in, target, creds.Info().SecurityProtocol, c.target, c.protocol)
		}
	}
	for _, bad := range []string{"https://host/api", "ftp://host:21", "https://:443"} {
		if _, _, err := Endpoint(bad); err == nil {
			t.Fatalf("%q: want an error", bad)
		}
	}
}
