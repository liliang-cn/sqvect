package authz

import (
	"bytes"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const (
	keyA = `{"id":"a","secret":"a-secret","clearance":"read-write"}`
	keyB = `{"id":"b","secret":"b-secret","clearance":"read-only"}`
	keyC = `{"id":"c","secret":"c-secret","clearance":"read-only"}`
)

func keyFileOf(keys ...string) string { return `{"keys":[` + strings.Join(keys, ",") + `]}` }

// writeKeyFile writes body to path and gives it a time an hour back, moved
// on by step: two writes of the same size are still two changes, and the
// time is old enough for a stat of the file to be trusted.
func writeKeyFile(t testing.TB, path, body string, step time.Duration) {
	t.Helper()
	writeKeyFileAt(t, path, body, time.Now().Add(-time.Hour+step))
}

// writeKeyFileAt writes body to path in place and gives it the time at.
func writeKeyFileAt(t testing.TB, path, body string, at time.Time) {
	t.Helper()
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, at, at); err != nil {
		t.Fatal(err)
	}
}

// openTestKeySource opens a source on a file holding keys a and b, and
// records what it warns about.
func openTestKeySource(t *testing.T) (*KeySource, string, *[]error) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "keys.json")
	writeKeyFile(t, path, keyFileOf(keyA, keyB), 0)
	src, err := OpenKeySource(path)
	if err != nil {
		t.Fatal(err)
	}
	var warned []error
	src.Warn = func(err error) {
		if !src.mu.TryLock() {
			t.Error("Warn is called with the lock held")
		} else {
			src.mu.Unlock()
		}
		warned = append(warned, err)
	}
	return src, path, &warned
}

// currentWithoutTheLock calls Current while another holder keeps the lock
// of the rereads: it returns only when Current trusts the stat of the file.
func currentWithoutTheLock(t *testing.T, src *KeySource, why string) *KeySet {
	t.Helper()
	src.mu.Lock()
	defer src.mu.Unlock()
	done := make(chan *KeySet, 1)
	go func() { done <- src.Current() }()
	select {
	case keys := <-done:
		return keys
	case <-time.After(5 * time.Second):
		t.Fatalf("Current waits for the lock: %s", why)
		return nil
	}
}

// stoppedClock dates the reads of the sources opened in this test at the
// file's time plus after, whatever the real clock says.
func stoppedClock(t *testing.T, path string, after time.Duration) time.Time {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	at := info.ModTime()
	clock = func() time.Time { return at.Add(after) }
	t.Cleanup(func() { clock = time.Now })
	return at
}

func served(src *KeySource, secret string) bool {
	_, ok := src.Current().Lookup(secret)
	return ok
}

func TestAKeySourceServesAKeyAddedToTheFileAtTheNextLookup(t *testing.T) {
	src, path, warned := openTestKeySource(t)
	if served(src, "c-secret") {
		t.Fatal("c served before it was added")
	}
	writeKeyFile(t, path, keyFileOf(keyA, keyB, keyC), time.Second)
	if !served(src, "c-secret") {
		t.Error("a key added to the file is refused")
	}
	if !served(src, "a-secret") {
		t.Error("a key the file still holds is refused")
	}
	if src.Current() != src.Current() {
		t.Error("the new keys are not kept: each lookup parses the file again")
	}
	if len(*warned) != 0 {
		t.Errorf("a good change warned: %v", *warned)
	}
}

func TestAKeySourceRefusesAKeyRemovedFromTheFileAtTheNextLookup(t *testing.T) {
	src, path, _ := openTestKeySource(t)
	if !served(src, "a-secret") {
		t.Fatal("a refused before it was removed")
	}
	writeKeyFile(t, path, keyFileOf(keyB), time.Second)
	if served(src, "a-secret") {
		t.Error("a key removed from the file is still served")
	}
	if !served(src, "b-secret") {
		t.Error("a key the file still holds is refused")
	}
}

// A file that does not load leaves the previous keys in force: an emptied
// file does not open the server, a broken one does not lock everybody out.
// Each failed change is reported once, however many lookups follow it.
func TestABrokenKeyFileKeepsThePreviousKeysAndWarnsOncePerChange(t *testing.T) {
	for _, c := range []struct {
		name   string
		damage func(t *testing.T, path string)
	}{
		{"invalid JSON", func(t *testing.T, path string) { writeKeyFile(t, path, `{"keys":[`, time.Second) }},
		{"unknown field", func(t *testing.T, path string) {
			writeKeyFile(t, path, `{"keys":[{"id":"a","secret":"a-secret","clearence":"read-write"}]}`, time.Second)
		}},
		{"no keys", func(t *testing.T, path string) { writeKeyFile(t, path, `{"keys":[]}`, time.Second) }},
		{"empty file", func(t *testing.T, path string) { writeKeyFile(t, path, ``, time.Second) }},
		{"file removed", func(t *testing.T, path string) {
			if err := os.Remove(path); err != nil {
				t.Fatal(err)
			}
		}},
		{"a directory in its place", func(t *testing.T, path string) {
			if err := os.Remove(path); err != nil {
				t.Fatal(err)
			}
			if err := os.Mkdir(path, 0o700); err != nil {
				t.Fatal(err)
			}
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			src, path, warned := openTestKeySource(t)
			before := src.Current()
			c.damage(t, path)
			for range 3 {
				if got := src.Current(); got != before {
					t.Fatalf("a broken file replaced the keys: %v", got)
				}
			}
			if !served(src, "a-secret") || !served(src, "b-secret") {
				t.Error("the previous keys are no longer served")
			}
			if len(*warned) != 1 {
				t.Fatalf("warned %d times, want once: %v", len(*warned), *warned)
			}
			if msg := (*warned)[0].Error(); !strings.Contains(msg, path) || !strings.Contains(msg, "previous keys stay in force") {
				t.Errorf("warning %q does not name the file and what stays", msg)
			}
		})
	}
}

// A second failed change is reported again, and a fixed file is served.
func TestAKeySourceRecoversWhenABrokenFileIsFixed(t *testing.T) {
	src, path, warned := openTestKeySource(t)
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	src.Current()
	writeKeyFile(t, path, keyFileOf(keyC), time.Second)
	if !served(src, "c-secret") || served(src, "a-secret") {
		t.Fatal("the fixed file is not the one served")
	}
	// The same failure as the first one, after a success, is a new change.
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	src.Current()
	if len(*warned) != 2 {
		t.Fatalf("warned %d times over two failed changes: %v", len(*warned), *warned)
	}
	if !served(src, "c-secret") {
		t.Error("the keys of the fixed file were dropped")
	}
}

// A file rewritten with the text already in force keeps the same key set:
// nothing is parsed again.
func TestTheSameTextRewrittenIsNotParsedAgain(t *testing.T) {
	src, path, _ := openTestKeySource(t)
	before := src.Current()
	writeKeyFile(t, path, keyFileOf(keyA, keyB), time.Second)
	if got := src.Current(); got != before {
		t.Error("the same text was parsed again")
	}
	// Broken, then put back as it was: the keys in force are still the
	// ones it holds.
	writeKeyFile(t, path, `{"keys":[`, 2*time.Second)
	src.Current()
	writeKeyFile(t, path, keyFileOf(keyA, keyB), 3*time.Second)
	if got := src.Current(); got != before {
		t.Error("the text put back was parsed again")
	}
}

// A file that failed to load is not read again until it changes, once its
// time is old enough for the stat to be trusted.
func TestABrokenFileIsNotReadAgainUntilItChanges(t *testing.T) {
	src, path, warned := openTestKeySource(t)
	good := keyFileOf(keyA, keyC)
	writeKeyFile(t, path, strings.Repeat("x", len(good)), time.Second)
	src.Current()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(good), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, info.ModTime(), info.ModTime()); err != nil {
		t.Fatal(err)
	}
	if served(src, "c-secret") || !served(src, "b-secret") {
		t.Error("a file that failed to load was read again before it changed")
	}
	if len(*warned) != 1 {
		t.Errorf("warned %d times, want once: %v", len(*warned), *warned)
	}
}

// A static source serves the keys it was given and reads no file.
func TestAStaticKeySourceServesItsKeysAndReadsNoFile(t *testing.T) {
	ks, err := Parse([]byte(keyFileOf(keyA)))
	if err != nil {
		t.Fatal(err)
	}
	for _, keys := range []*KeySet{ks, nil} {
		src := StaticKeySource(keys)
		var warned []error
		src.Warn = func(err error) { warned = append(warned, err) }
		if src.Current() != keys || src.Current() != keys {
			t.Errorf("StaticKeySource(%v) serves another key set", keys)
		}
		if len(warned) != 0 {
			t.Errorf("a static source warned: %v", warned)
		}
	}
}

// A file replaced by rename, with the same size and time as the one it
// replaces, is still read: it is another file.
func TestAFileReplacedWithTheSameSizeAndTimeIsStillReread(t *testing.T) {
	src, path, _ := openTestKeySource(t)
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	next := path + ".next"
	// keyC is keyB with another id and secret of the same length.
	if err := os.WriteFile(next, []byte(keyFileOf(keyA, keyC)), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(next, info.ModTime(), info.ModTime()); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(next, path); err != nil {
		t.Fatal(err)
	}
	if after, _ := os.Stat(path); after.Size() != info.Size() || !after.ModTime().Equal(info.ModTime()) {
		t.Fatalf("the replacement differs in size or time: %d %v, want %d %v", after.Size(), after.ModTime(), info.Size(), info.ModTime())
	}
	if !served(src, "c-secret") || served(src, "b-secret") {
		t.Error("a file replaced with the same size and time was not read")
	}
}

func TestTheFirstLoadOfAKeySourceMustSucceed(t *testing.T) {
	dir := t.TempDir()
	broken := filepath.Join(dir, "broken.json")
	writeKeyFile(t, broken, `{"keys":[]}`, 0)
	for name, path := range map[string]string{
		"missing":     filepath.Join(dir, "missing.json"),
		"a directory": dir,
		"no keys":     broken,
	} {
		t.Run(name, func(t *testing.T) {
			src, err := OpenKeySource(path)
			if err == nil || src != nil {
				t.Fatalf("OpenKeySource(%s) = %v, %v; want an error", path, src, err)
			}
			// The error Load gives.
			if _, loadErr := Load(path); loadErr == nil || loadErr.Error() != err.Error() {
				t.Errorf("OpenKeySource says %q, Load says %v", err, loadErr)
			}
		})
	}
}

// Without Warn, a failed change goes to the standard logger.
func TestAKeySourceWithoutWarnLogsAFailedChange(t *testing.T) {
	src, path, _ := openTestKeySource(t)
	src.Warn = nil
	var out bytes.Buffer
	log.SetOutput(&out)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })
	writeKeyFile(t, path, `{"keys":[]}`, time.Second)
	src.Current()
	if !strings.Contains(out.String(), "previous keys stay in force") {
		t.Errorf("nothing logged for a failed change: %q", out.String())
	}
}

// What the reread costs a lookup: a stat of the key file. Run with
// go test -run '^$' -bench 'KeySet|KeySource' -benchmem ./pkg/authz/
func BenchmarkKeySetLookup(b *testing.B) {
	ks, err := Parse([]byte(keyFileOf(keyA, keyB)))
	if err != nil {
		b.Fatal(err)
	}
	for b.Loop() {
		ks.Lookup("b-secret")
	}
}

func BenchmarkKeySourceLookup(b *testing.B) {
	src := benchKeySource(b)
	for b.Loop() {
		src.Current().Lookup("b-secret")
	}
}

func BenchmarkKeySetLookupParallel(b *testing.B) {
	ks, err := Parse([]byte(keyFileOf(keyA, keyB)))
	if err != nil {
		b.Fatal(err)
	}
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			ks.Lookup("b-secret")
		}
	})
}

func BenchmarkKeySourceLookupParallel(b *testing.B) {
	src := benchKeySource(b)
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			src.Current().Lookup("b-secret")
		}
	})
}

func benchKeySource(b *testing.B) *KeySource {
	b.Helper()
	path := filepath.Join(b.TempDir(), "keys.json")
	writeKeyFile(b, path, keyFileOf(keyA, keyB), 0)
	src, err := OpenKeySource(path)
	if err != nil {
		b.Fatal(err)
	}
	return src
}

// ResolveSource takes what Resolve takes: a key file is the whole policy and
// is read again when it changes; without one, the legacy token, fixed.
func TestResolveSourcePicksThePolicyAsResolveDoes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "keys.json")
	writeKeyFile(t, path, keyFileOf(keyA), 0)
	src, err := ResolveSource(path, "legacy")
	if err != nil {
		t.Fatal(err)
	}
	if !served(src, "a-secret") || served(src, "legacy") {
		t.Error("the key file is not the whole policy")
	}
	writeKeyFile(t, path, keyFileOf(keyB), time.Second)
	if served(src, "a-secret") || !served(src, "b-secret") {
		t.Error("the key file is not read again")
	}
	if src, err := ResolveSource("", "legacy"); err != nil || !served(src, "legacy") {
		t.Errorf("the legacy token is not served: %v", err)
	}
	if src, err := ResolveSource("", ""); err != nil || src.Current().Enabled() {
		t.Errorf("no key file and no token is not a disabled policy: %v", err)
	}
	if _, err := ResolveSource(filepath.Join(t.TempDir(), "missing.json"), "legacy"); err == nil {
		t.Error("a key file that does not load fell back to the token")
	}
}

// A file rewritten in place with the same size and time as the last read —
// a secret rotated to one of the same length, within one tick of the file
// system clock — is still read: its time is too close to that read for the
// stat to be trusted, so the text is compared.
func TestASameSizeRewriteWithinTheTickOfTheLastReadIsServed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "keys.json")
	writeKeyFile(t, path, keyFileOf(keyA, keyB), 0)
	at := stoppedClock(t, path, 0)
	src, err := OpenKeySource(path)
	if err != nil {
		t.Fatal(err)
	}
	before, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	// keyC is keyB with another id and secret of the same length.
	writeKeyFileAt(t, path, keyFileOf(keyA, keyC), at)
	after, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if !os.SameFile(before, after) || after.Size() != before.Size() || !after.ModTime().Equal(before.ModTime()) {
		t.Fatalf("the rewrite changed the file, its size or its time")
	}
	if !served(src, "c-secret") || served(src, "b-secret") {
		t.Error("a same-size rewrite within the tick of the last read was not seen")
	}
}

// Once the file is older than the window, the stat is trusted again: the
// read that finds the same text records its time, and the lookups that
// follow take no lock.
func TestARecentFileIsComparedUntilItsTimeIsOldEnough(t *testing.T) {
	path := filepath.Join(t.TempDir(), "keys.json")
	writeKeyFile(t, path, keyFileOf(keyA, keyB), 0)
	at := stoppedClock(t, path, 0)
	src, err := OpenKeySource(path)
	if err != nil {
		t.Fatal(err)
	}
	before := src.Current()
	src.now = func() time.Time { return at.Add(time.Minute) }
	if src.Current() != before {
		t.Fatal("the same text was parsed again")
	}
	if currentWithoutTheLock(t, src, "the file is a minute older than the last read") != before {
		t.Error("another key set served")
	}
}

// A file changed and read once is then served from its stat.
func TestAChangedFileIsReadOnceThenServedFromItsStat(t *testing.T) {
	src, path, _ := openTestKeySource(t)
	writeKeyFile(t, path, keyFileOf(keyA, keyB, keyC), time.Second)
	keys := src.Current()
	if _, ok := keys.Lookup("c-secret"); !ok {
		t.Fatal("the change was not read")
	}
	if currentWithoutTheLock(t, src, "the file has not changed since the last read") != keys {
		t.Error("another key set served")
	}
}

// Every failed change is reported, also with no success between two of
// them, with two broken files of the same size, and with two of the same
// time and another size.
func TestEachFailedChangeInARowIsReported(t *testing.T) {
	src, path, warned := openTestKeySource(t)
	base := time.Now().Add(-time.Hour)
	for i, c := range []struct {
		body string
		at   time.Duration
	}{
		{`{"keys":[1`, time.Second},
		{`{"keys":[2`, 2 * time.Second},
		{`{"keys":[`, 3 * time.Second},
		{`{"keys":[12`, 3 * time.Second},
	} {
		writeKeyFileAt(t, path, c.body, base.Add(c.at))
		src.Current()
		src.Current()
		if len(*warned) != i+1 {
			t.Fatalf("after %d failed changes in a row, %d warnings: %v", i+1, len(*warned), *warned)
		}
	}
	if !served(src, "a-secret") {
		t.Error("the keys in force were dropped")
	}
}

// A stat that fails in another way is another failed change: the file
// missing, then a regular file where its directory was.
func TestAStatThatFailsAnotherWayIsReportedAgain(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "conf")
	if err := os.Mkdir(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "keys.json")
	writeKeyFile(t, path, keyFileOf(keyA, keyB), 0)
	src, err := OpenKeySource(path)
	if err != nil {
		t.Fatal(err)
	}
	var warned []error
	src.Warn = func(err error) { warned = append(warned, err) }
	if err := os.RemoveAll(dir); err != nil {
		t.Fatal(err)
	}
	src.Current()
	src.Current()
	if len(warned) != 1 {
		t.Fatalf("the file missing: %d warnings, want 1: %v", len(warned), warned)
	}
	writeKeyFile(t, dir, "not a directory", 0)
	src.Current()
	src.Current()
	if len(warned) != 2 {
		t.Fatalf("a file where the directory was: %d warnings, want 2: %v", len(warned), warned)
	}
	if !served(src, "a-secret") {
		t.Error("the keys in force were dropped")
	}
}

// The window is two seconds: a file whose time is 1.9 s before the last read
// is still read and compared, one 2.1 s before it is served from its stat.
func TestTheWindowIsTwoSeconds(t *testing.T) {
	for _, c := range []struct {
		name    string
		after   time.Duration
		trusted bool
	}{
		{"1.9 s", 1900 * time.Millisecond, false},
		{"2.1 s", 2100 * time.Millisecond, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "keys.json")
			writeKeyFile(t, path, keyFileOf(keyA, keyB), 0)
			at := stoppedClock(t, path, c.after)
			src, err := OpenKeySource(path)
			if err != nil {
				t.Fatal(err)
			}
			if c.trusted {
				if currentWithoutTheLock(t, src, "the file is "+c.name+" older than the last read") == nil {
					t.Error("no key set served")
				}
				return
			}
			// Read and compared: the opening read is within the window.
			before := src.Current()
			if _, ok := before.Lookup("b-secret"); !ok {
				t.Fatal("the keys of the file are not served")
			}
			writeKeyFileAt(t, path, keyFileOf(keyA, keyC), at)
			if !served(src, "c-secret") || served(src, "b-secret") {
				t.Error("a same-size rewrite " + c.name + " before the last read was not seen")
			}
		})
	}
}

// A KeySource that was never opened — the zero value, or nil — refuses
// every key rather than opening the server.
func TestAKeySourceNeverOpenedRefusesEveryKey(t *testing.T) {
	for name, src := range map[string]*KeySource{"zero": {}, "nil": nil} {
		keys := src.Current()
		if !keys.Enabled() {
			t.Errorf("%s source: a disabled policy, which lets every call through", name)
		}
		for _, secret := range []string{"", "a-secret"} {
			if _, ok := keys.Lookup(secret); ok {
				t.Errorf("%s source serves %q", name, secret)
			}
		}
	}
}

// A file rewritten in place with the time it had but another size is read
// again: the size alone tells the change, as when a key is removed.
func TestAFileRewrittenWithTheSameTimeButAnotherSizeIsReread(t *testing.T) {
	src, path, _ := openTestKeySource(t)
	before, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	writeKeyFileAt(t, path, keyFileOf(keyB), before.ModTime())
	after, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if !os.SameFile(before, after) || !after.ModTime().Equal(before.ModTime()) {
		t.Fatalf("the rewrite changed the file or its time")
	}
	if served(src, "a-secret") || !served(src, "b-secret") {
		t.Error("a file rewritten with another size was not read")
	}
}

// While the file is missing, or a directory stands in its place, a lookup
// serves the keys in force without taking the lock of the rereads.
func TestAMissingKeyFileIsNotLookedForUnderTheLock(t *testing.T) {
	for name, damage := range map[string]func(t *testing.T, path string){
		"removed": func(t *testing.T, path string) {
			if err := os.Remove(path); err != nil {
				t.Fatal(err)
			}
		},
		"a directory": func(t *testing.T, path string) {
			if err := os.Remove(path); err != nil {
				t.Fatal(err)
			}
			if err := os.Mkdir(path, 0o700); err != nil {
				t.Fatal(err)
			}
			old := time.Now().Add(-time.Hour)
			if err := os.Chtimes(path, old, old); err != nil {
				t.Fatal(err)
			}
		},
	} {
		t.Run(name, func(t *testing.T) {
			src, path, warned := openTestKeySource(t)
			before := src.Current()
			damage(t, path)
			src.Current()
			if currentWithoutTheLock(t, src, "the file is still "+name) != before {
				t.Error("another key set served")
			}
			if len(*warned) != 1 {
				t.Errorf("warned %d times, want once", len(*warned))
			}
		})
	}
}

// A file opened with a time older than the window is served from its stat
// from the first lookup: the opening read counts as the last read.
func TestAnOldFileIsServedFromItsStatFromTheFirstLookup(t *testing.T) {
	src, _, _ := openTestKeySource(t)
	keys := currentWithoutTheLock(t, src, "the file has not changed since it was opened")
	if _, ok := keys.Lookup("a-secret"); !ok {
		t.Error("the keys of the file are not served")
	}
}
