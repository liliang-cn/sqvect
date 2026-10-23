// Package authz is the key and scope model behind CortexDB's gRPC server.
//
// The server's documented deployment is a shared brain: many agents on many
// machines against one CortexDB. Until this package existed there was exactly
// one authorisation state — hold the bearer token and you have full read/write
// over every row every other agent ever wrote, so an agent on a laptop that
// only needs to recall its own project notes could delete somebody else's
// memory. A key here carries a clearance (read-only or read-write) and a scope
// confining which rows it may touch, so the blast radius of one leaked or
// misbehaving agent is the slice it was given rather than the whole brain.
//
// What this is not: transport security. CortexDB's gRPC transport is plaintext
// by design — loopback, a trusted LAN, or a Tailscale interface — so anyone who
// can read the wire reads the secrets along with the traffic and can then act
// as any key they have observed. Scoped keys reduce the blast radius between
// cooperating agents that already share a trusted network; they are not a
// defence against someone on that network. If the network is not trusted, put
// TLS or a tunnel underneath. Nothing in this package changes that.
package authz

import (
	"bytes"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// ErrDenied is the sentinel behind every refusal in this package. The gRPC
// layer maps it onto PERMISSION_DENIED; callers that only want to know whether
// something was an authorisation failure can use errors.Is instead of matching
// on message text.
var ErrDenied = errors.New("denied")

// Clearance is how much a key may do, independently of which rows it may see.
// There are two values on purpose: the moment clearance becomes a set of
// per-RPC grants, nobody can answer "what can this key do" by looking at it.
type Clearance string

const (
	// ReadOnly may invoke read RPCs only.
	ReadOnly Clearance = "read-only"
	// ReadWrite may invoke everything its scope allows.
	ReadWrite Clearance = "read-write"
)

// Valid reports whether c is one of the two clearances. There is no default:
// a key file that omits the clearance is rejected rather than assumed, because
// the assumption that would break quietly is "read-write".
func (c Clearance) Valid() bool { return c == ReadOnly || c == ReadWrite }

// The request field names a scope confines. These are proto field names, not Go
// ones, because that is what the interceptor reads off the wire.
const (
	FieldUserID     = "user_id"
	FieldScope      = "scope"
	FieldNamespace  = "namespace"
	FieldCollection = "collection"
)

// Scope confines a key to rows matching every field it sets. The zero Scope is
// unconfined and sees everything the clearance allows.
//
// These four fields are exactly the ones CortexDB requests already carry as
// proto fields, which is what makes the confinement decidable in an interceptor
// without the server first reading the row. Deliberately left out: anything
// that would need a query to evaluate — row ids, content predicates, time
// windows, per-RPC allow lists. A scope that cannot be decided from the request
// message alone is a scope that fails open somewhere, and a scope nobody can
// reason about is worse than no scope at all.
//
// Each field comes in a singular and a plural form. The singular confines the
// key to one value, which is what every key file written so far says. The
// plural confines it to any of several — the key of an agent that reads its
// team's collection and its own — and the two forms add up when both are
// given. A request still names exactly one value per field; the plural widens
// what the key may name, not what one request may ask for.
type Scope struct {
	UserID string `json:"user_id,omitempty"`
	// MemoryScope is the memory scope string (MemoryScopeUser, Session, …).
	// It is called "scope" on the wire; the Go field is not, because
	// Scope.Scope reads like a typo every time it appears.
	MemoryScope string `json:"scope,omitempty"`
	Namespace   string `json:"namespace,omitempty"`
	Collection  string `json:"collection,omitempty"`

	UserIDs      []string `json:"user_ids,omitempty"`
	MemoryScopes []string `json:"scopes,omitempty"`
	Namespaces   []string `json:"namespaces,omitempty"`
	Collections  []string `json:"collections,omitempty"`
}

// IsZero reports whether the scope confines nothing.
func (s Scope) IsZero() bool { return len(s.constraints()) == 0 }

// Permits reports whether the scope lets a key touch a row whose named field
// holds value. A field the scope does not confine permits anything. A confined
// field permits only its values, so an empty value — a row that predates
// scoping, or a field the row's kind does not have — is never permitted, which
// is the fail-closed answer the row checks rely on.
func (s Scope) Permits(field, value string) bool {
	for _, c := range s.constraints() {
		if c.field == field {
			return c.allows(value)
		}
	}
	return true
}

// constraint is one confinement: a request field name and the values the key
// may ask for under it.
type constraint struct {
	field  string
	values []string
}

// allows reports whether a request may name v under this field.
func (c constraint) allows(v string) bool {
	for _, want := range c.values {
		if v == want {
			return true
		}
	}
	return false
}

// String is how a denial spells the confinement: one value quoted as before,
// several as a list, so the operator reading the log sees the whole scope.
func (c constraint) String() string {
	if len(c.values) == 1 {
		return strconv.Quote(c.values[0])
	}
	quoted := make([]string, len(c.values))
	for i, v := range c.values {
		quoted[i] = strconv.Quote(v)
	}
	return "[" + strings.Join(quoted, ",") + "]"
}

// constraints lists the confinements in a stable order, so a denial names the
// same field every time for the same key and request. Singular and plural
// forms of a field merge into one constraint.
func (s Scope) constraints() []constraint {
	var out []constraint
	for _, c := range []struct {
		field string
		one   string
		many  []string
	}{
		{FieldUserID, s.UserID, s.UserIDs},
		{FieldScope, s.MemoryScope, s.MemoryScopes},
		{FieldNamespace, s.Namespace, s.Namespaces},
		{FieldCollection, s.Collection, s.Collections},
	} {
		var values []string
		if c.one != "" {
			values = append(values, c.one)
		}
		for _, v := range c.many {
			if v != "" && !contains(values, v) {
				values = append(values, v)
			}
		}
		if len(values) > 0 {
			out = append(out, constraint{field: c.field, values: values})
		}
	}
	return out
}

func contains(values []string, v string) bool {
	for _, w := range values {
		if w == v {
			return true
		}
	}
	return false
}

// validate refuses the one shape of scope that would read as a confinement and
// act as none: a plural list holding an empty string. An empty singular field
// is "unconfined" by design and stays so.
func (s Scope) validate() error {
	for _, l := range []struct {
		name   string
		values []string
	}{
		{"user_ids", s.UserIDs}, {"scopes", s.MemoryScopes},
		{"namespaces", s.Namespaces}, {"collections", s.Collections},
	} {
		for _, v := range l.values {
			if v == "" {
				return fmt.Errorf("scope %s holds an empty value", l.name)
			}
		}
	}
	return nil
}

// Key is one credential. The secret is the bearer token as it arrives on the
// wire; the id exists so a denial can be attributed and so a key can be revoked
// by name — deleting its entry from the key file is the revocation.
type Key struct {
	ID        string    `json:"id"`
	Secret    string    `json:"secret"`
	Clearance Clearance `json:"clearance"`
	Scope     Scope     `json:"scope,omitempty"`
}

// KeySet is the whole policy a server enforces. A nil or empty KeySet means no
// authentication at all, which is the historical behaviour of an unset token
// and is preserved deliberately: silently locking an open loopback deployment
// on upgrade would be a worse surprise than the open default. The one set
// without keys that enforces is the closed set a KeySource that was never
// opened serves: it refuses every key.
type KeySet struct {
	keys []Key
	// closed is a set with no key that still enforces: it refuses every
	// call. A KeySource that was never opened serves it.
	closed bool
}

// Enabled reports whether the set actually enforces anything.
func (ks *KeySet) Enabled() bool { return ks != nil && (len(ks.keys) > 0 || ks.closed) }

// Len is the number of keys, for tests and for startup logging.
func (ks *KeySet) Len() int {
	if ks == nil {
		return 0
	}
	return len(ks.keys)
}

// Lookup resolves a presented bearer secret to its key.
//
// Every key is compared and a match does not break the loop: returning early
// would leak, through timing, roughly where in the file a guessed secret sits.
// The comparison itself stays crypto/subtle.ConstantTimeCompare, as it was when
// there was one token and one comparison.
func (ks *KeySet) Lookup(secret string) (Key, bool) {
	if ks == nil {
		return Key{}, false
	}
	presented := []byte(secret)
	found := -1
	for i := range ks.keys {
		if subtle.ConstantTimeCompare(presented, []byte(ks.keys[i].Secret)) == 1 {
			found = i
		}
	}
	if found < 0 {
		return Key{}, false
	}
	return ks.keys[found], true
}

// NewKeySet validates keys and returns the policy they describe.
func NewKeySet(keys []Key) (*KeySet, error) {
	ids := make(map[string]struct{}, len(keys))
	secrets := make(map[string]struct{}, len(keys))
	for i, k := range keys {
		switch {
		case k.ID == "":
			return nil, fmt.Errorf("key %d: id is required", i)
		case k.Secret == "":
			return nil, fmt.Errorf("key %q: secret is required", k.ID)
		case !k.Clearance.Valid():
			return nil, fmt.Errorf("key %q: clearance must be %q or %q, got %q",
				k.ID, ReadOnly, ReadWrite, k.Clearance)
		}
		if _, dup := ids[k.ID]; dup {
			return nil, fmt.Errorf("key %q: duplicate id", k.ID)
		}
		if err := k.Scope.validate(); err != nil {
			return nil, fmt.Errorf("key %q: %w", k.ID, err)
		}
		// Two keys sharing a secret cannot be told apart on the wire, so which
		// scope a request got would depend on file order. Refuse instead.
		if _, dup := secrets[k.Secret]; dup {
			return nil, fmt.Errorf("key %q: shares a secret with an earlier key", k.ID)
		}
		ids[k.ID] = struct{}{}
		secrets[k.Secret] = struct{}{}
	}
	out := make([]Key, len(keys))
	copy(out, keys)
	return &KeySet{keys: out}, nil
}

// keyFile is the on-disk shape: {"keys": [...]}.
type keyFile struct {
	Keys []Key `json:"keys"`
}

// Parse reads a key file's contents.
func Parse(data []byte) (*KeySet, error) {
	var f keyFile
	dec := json.NewDecoder(bytes.NewReader(data))
	// Unknown fields are an error rather than a shrug: a misspelled "scopes" or
	// "clearence" would otherwise load as an unconfined read-write key, which
	// is the single worst way for this file to be wrong.
	dec.DisallowUnknownFields()
	if err := dec.Decode(&f); err != nil {
		return nil, fmt.Errorf("parse key file: %w", err)
	}
	// A key file is one JSON document, not a stream of concatenated values.
	if _, err := dec.Token(); err != io.EOF {
		return nil, errors.New("parse key file: unexpected trailing data")
	}
	if len(f.Keys) == 0 {
		return nil, errors.New("parse key file: no keys declared")
	}
	return NewKeySet(f.Keys)
}

// Load reads a key file from disk, once. A server that keeps running reads it
// through a KeySource, which reads it again when it changes.
func Load(path string) (*KeySet, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read key file: %w", err)
	}
	return Parse(data)
}

// LegacyKeyID names the implicit key synthesised from the single-token
// configuration, so denials and audit lines have something to name even for
// deployments that never wrote a key file.
const LegacyKeyID = "legacy-token"

// LegacyToken maps the historical CORTEXDB_GRPC_TOKEN deployment onto the key
// model: one full-access key, unconfined. Every deployment in the wild is that
// shape, and this mapping is the whole of the backward compatibility story —
// there is no second code path where the old token is treated differently.
func LegacyToken(token string) *KeySet {
	if token == "" {
		return nil
	}
	return &KeySet{keys: []Key{{
		ID:        LegacyKeyID,
		Secret:    token,
		Clearance: ReadWrite,
	}}}
}

// Resolve picks the policy for a server from its two configuration inputs.
//
// A key file, when given, is the entire policy: the legacy token is not also
// admitted alongside it. Honouring both would leave the environment variable as
// a master key outranking every scope in the file, which is exactly the hole
// the file exists to close.
func Resolve(keyFilePath, legacyToken string) (*KeySet, error) {
	if keyFilePath != "" {
		return Load(keyFilePath)
	}
	return LegacyToken(legacyToken), nil
}
