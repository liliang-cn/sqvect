package httpapi

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
)

const (
	reloadKeyA = `{"id":"a","secret":"a-secret","clearance":"read-write"}`
	reloadKeyB = `{"id":"b","secret":"b-secret","clearance":"read-only"}`
	reloadKeyC = `{"id":"c","secret":"c-secret","clearance":"read-only"}`
)

func reloadKeyFile(keys ...string) string { return `{"keys":[` + strings.Join(keys, ",") + `]}` }

// rewriteKeyFile writes body to path and moves its time on by step, so the
// change shows even within one tick of the file system clock.
func rewriteKeyFile(t *testing.T, path, body string, step time.Duration) {
	t.Helper()
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	at := time.Now().Add(step)
	if err := os.Chtimes(path, at, at); err != nil {
		t.Fatal(err)
	}
}

func TestAKeyRemovedFromTheKeyFileGets401WithoutARestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "keys.json")
	rewriteKeyFile(t, path, reloadKeyFile(reloadKeyA, reloadKeyB), 0)
	srv := newPolicyServer(t, Options{KeyFile: path})
	wantStatus(t, srv, http.MethodGet, "/v1/health", "a-secret", "", http.StatusOK)
	rewriteKeyFile(t, path, reloadKeyFile(reloadKeyB, reloadKeyC), time.Second)
	for secret, want := range map[string]int{
		"a-secret": http.StatusUnauthorized, // removed from the file
		"c-secret": http.StatusOK,           // added to it
		"b-secret": http.StatusOK,           // kept
	} {
		if got, body := do(t, srv, http.MethodGet, "/v1/health", secret, ""); got != want {
			t.Errorf("GET /v1/health as %s = %d, want %d: %s", secret, got, want, body)
		}
	}
}

// An emptied key file is a file that does not load: the keys in force stay,
// and a request without a key is still refused.
func TestAnEmptiedKeyFileDoesNotOpenTheRESTServer(t *testing.T) {
	path := filepath.Join(t.TempDir(), "keys.json")
	rewriteKeyFile(t, path, reloadKeyFile(reloadKeyA), 0)
	srv := newPolicyServer(t, Options{KeyFile: path})
	rewriteKeyFile(t, path, `{"keys":[]}`, time.Second)
	wantStatus(t, srv, http.MethodGet, "/v1/health", "", "", http.StatusUnauthorized)
	wantStatus(t, srv, http.MethodGet, "/v1/health", "a-secret", "", http.StatusOK)
}

// Options.Keys is a key set the caller built: it is served as given, and
// the file KeyFile names is never read.
func TestAPreloadedKeySetIsNeverRereadOverREST(t *testing.T) {
	path := filepath.Join(t.TempDir(), "keys.json")
	rewriteKeyFile(t, path, reloadKeyFile(reloadKeyC), 0)
	keys, err := authz.Parse([]byte(reloadKeyFile(reloadKeyA)))
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: keys, KeyFile: path})
	rewriteKeyFile(t, path, reloadKeyFile(reloadKeyB), time.Second)
	wantStatus(t, srv, http.MethodGet, "/v1/health", "a-secret", "", http.StatusOK)
	wantStatus(t, srv, http.MethodGet, "/v1/health", "b-secret", "", http.StatusUnauthorized)
	wantStatus(t, srv, http.MethodGet, "/v1/health", "c-secret", "", http.StatusUnauthorized)
}
