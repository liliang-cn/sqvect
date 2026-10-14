package connector

import (
	"testing"
	"unicode/utf8"
)

func TestMaskEmailPreservesUTF8AndHidesMalformedValues(t *testing.T) {
	for _, tc := range []struct{ input, want string }{
		{"张三@example.com", "张***@example.com"},
		{"éclair@example.com", "é***@example.com"},
		{"张@example.com", "***@example.com"},
		{"a@example.com", "***@example.com"},
		{"alice@example.com", "a***@example.com"},
		{"alice", "*****"},
		{"张三", "**"},
	} {
		t.Run(tc.input, func(t *testing.T) {
			got := MaskValue(PiiEmail, tc.input)
			if got != tc.want || !utf8.ValidString(got) {
				t.Fatalf("mask = %q (valid UTF-8 %v), want %q", got, utf8.ValidString(got), tc.want)
			}
		})
	}
}
