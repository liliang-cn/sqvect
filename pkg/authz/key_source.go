package authz

import (
	"bytes"
	"fmt"
	"log"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

// KeySource serves the keys of a key file as the file stands now: a key added
// to the file is served at the next lookup, a key removed from it is refused,
// without a restart.
//
// A change that does not load — a file that cannot be read, broken JSON, an
// unknown field, no keys at all — leaves the keys in force and is reported
// once, through Warn. An emptied file is one of those: it must not open the
// server, and a typo must not lock everybody out.
//
// Each Current costs one stat of the file, outside any lock. The file is read
// again when its time, its size or the file itself (a rename puts another
// one at the path) differs from the last read, and parsed again only when
// its text differs from the text of the keys in force. A stat is trusted
// only once the file's time is racyWindow older than the last read: until
// then, a rewrite of the same size within the same tick of the file system
// clock would look unchanged, so the file is read and its text compared.
//
// A KeySource comes from OpenKeySource, StaticKeySource or ResolveSource.
// The zero value, and a nil *KeySource, refuse every key.
type KeySource struct {
	path string
	// Warn receives one error per change of the file that fails to load.
	// Nil sends it to the standard logger. Set it before the first Current.
	Warn func(error)
	cur  atomic.Pointer[keySnapshot]
	// mu orders the rereads; failed is the change that failed last, the
	// file's time and size or the stat error, reported once. Empty after a
	// success.
	mu     sync.Mutex
	failed string
	now    func() time.Time
}

// racyWindow is how much older than the last read a file's time must be
// before its stat is trusted: two seconds, the coarsest clock of the common
// file systems (FAT).
const racyWindow = 2 * time.Second

// clock dates the reads of the key sources opened from then on.
var clock = time.Now

// keySnapshot is what Current serves until the file changes.
type keySnapshot struct {
	keys *KeySet
	// data is the text keys were parsed from; info is the file as last
	// read, which may be a later one that failed to load, and readAt the
	// time that read started. statErr is the error of a stat that failed,
	// with info nil.
	data    []byte
	info    os.FileInfo
	readAt  time.Time
	statErr string
}

// unchanged reports whether a stat of the file, as info or as err, shows
// the file of the snapshot, unchanged, and old enough for the stat to tell.
func (snap *keySnapshot) unchanged(info os.FileInfo, err error) bool {
	if err != nil {
		return err.Error() == snap.statErr
	}
	a := snap.info
	return a != nil && a.ModTime().Equal(info.ModTime()) && a.Size() == info.Size() && os.SameFile(a, info) &&
		a.ModTime().Before(snap.readAt.Add(-racyWindow))
}

// OpenKeySource loads a key file and serves it from then on. The first load
// must succeed, with the errors Load gives.
func OpenKeySource(path string) (*KeySource, error) {
	readAt := clock()
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read key file: %w", err)
	}
	keys, err := Parse(data)
	if err != nil {
		return nil, err
	}
	// A stat taken after the read: a change between the two has a time
	// within racyWindow of readAt, so the next Current reads it. A stat
	// that fails leaves info nil, which no stat matches.
	info, _ := os.Stat(path)
	s := &KeySource{path: path, now: clock}
	s.cur.Store(&keySnapshot{keys: keys, data: data, info: info, readAt: readAt})
	return s, nil
}

// StaticKeySource serves keys as given and never reads a file. A nil set
// is a disabled policy, as it is everywhere else.
func StaticKeySource(keys *KeySet) *KeySource {
	s := &KeySource{}
	s.cur.Store(&keySnapshot{keys: keys})
	return s
}

// ResolveSource is Resolve for a server that keeps running: a key file is
// served by a KeySource, and the legacy token by a static one.
func ResolveSource(keyFilePath, legacyToken string) (*KeySource, error) {
	if keyFilePath != "" {
		return OpenKeySource(keyFilePath)
	}
	return StaticKeySource(LegacyToken(legacyToken)), nil
}

// refuseAll is what a KeySource that was never opened serves.
var refuseAll = &KeySet{closed: true}

// Current returns the keys in force.
func (s *KeySource) Current() *KeySet {
	if s == nil {
		return refuseAll
	}
	snap := s.cur.Load()
	if snap == nil {
		return refuseAll
	}
	if s.path == "" {
		return snap.keys
	}
	info, err := os.Stat(s.path)
	if snap.unchanged(info, err) {
		return snap.keys
	}
	keys, warning := s.reread(info, err)
	if warning != nil {
		s.warn(warning)
	}
	return keys
}

// reread loads the file a stat saw change, as info or as statErr, and
// returns the keys in force and, for a failed change not reported yet, the
// warning to send once the lock is released.
func (s *KeySource) reread(info os.FileInfo, err error) (*KeySet, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	snap := s.cur.Load()
	next := &keySnapshot{keys: snap.keys, data: snap.data, info: info, readAt: s.now()}
	if err == nil {
		var data []byte
		if data, err = os.ReadFile(s.path); err == nil && !bytes.Equal(data, snap.data) {
			var keys *KeySet
			if keys, err = Parse(data); err == nil {
				next.keys, next.data = keys, data
			}
		}
	}
	// Stored on a failure too: a file that failed is not read again until
	// it changes, and a missing one is not looked for under the lock.
	if info == nil {
		next.statErr = err.Error()
	}
	s.cur.Store(next)
	if err == nil {
		s.failed = ""
		return next.keys, nil
	}
	key := "stat: " + next.statErr
	if info != nil {
		key = fmt.Sprintf("%s %d", info.ModTime(), info.Size())
	}
	if key == s.failed {
		return next.keys, nil
	}
	s.failed = key
	return next.keys, fmt.Errorf("key file %s: %w; the previous keys stay in force", s.path, err)
}

func (s *KeySource) warn(err error) {
	if s.Warn != nil {
		s.Warn(err)
		return
	}
	log.Printf("authz: %v", err)
}
