package authz

import (
	"os"
	"path/filepath"
	"testing"
)

const validTestKeyFile = `{"keys":[{"id":"test-reader","secret":"fixture-token","clearance":"read-only"}]}`

func TestKeyFileRejectsTrailingData(t *testing.T) {
	for name, suffix := range map[string]string{
		"second object":   `{}`,
		"second key file": validTestKeyFile,
		"null":            `null`,
		"invalid JSON":    `broken`,
		"comment":         `// comment`,
	} {
		t.Run(name, func(t *testing.T) {
			if ks, err := Parse([]byte(validTestKeyFile + "\n" + suffix)); err == nil || ks != nil {
				t.Fatalf("trailing data accepted: keys=%v, err=%v", ks, err)
			}
		})
	}
}

func TestKeyFileAllowsTrailingJSONWhitespace(t *testing.T) {
	for _, suffix := range []string{"", " ", "\n", "\r\n\t "} {
		ks, err := Parse([]byte(validTestKeyFile + suffix))
		if err != nil || ks.Len() != 1 {
			t.Fatalf("valid key file rejected with suffix %q: keys=%v, err=%v", suffix, ks, err)
		}
		if key, ok := ks.Lookup("fixture-token"); !ok || key.Clearance != ReadOnly {
			t.Fatal("valid key did not retain its clearance")
		}
	}
}

func TestLoadKeyFileRejectsTrailingData(t *testing.T) {
	path := filepath.Join(t.TempDir(), "keys.json")
	if err := os.WriteFile(path, []byte(validTestKeyFile+"\n{}"), 0o600); err != nil {
		t.Fatal(err)
	}
	if ks, err := Load(path); err == nil || ks != nil {
		t.Fatalf("trailing data accepted from file: keys=%v, err=%v", ks, err)
	}
}
