package authz

import (
	"errors"
	"strings"
	"testing"
)

// check_against_sources reads one collection, which the scope check forces
// to the key's. The sources it names are data: the facade holds them to the
// collection, so a source of another collection comes back missing, exactly
// as a source that does not exist.
func TestAConfinedKeyChecksOnlyItsCollection(t *testing.T) {
	notes := Key{ID: "notes", Secret: "n", Clearance: ReadOnly, Scope: Scope{Collection: "notes"}}
	hermes := Key{ID: "hermes", Secret: "h", Clearance: ReadOnly, Scope: Scope{UserID: "hermes"}}
	sentences := `"sentences":[{"text":"Orpheus earns 5.","sources":["hr-salaries-2026","chunk:hr-salaries-2026:000"]}],"equivalents":{"rgpd":["gdpr"]}`
	for _, c := range []struct {
		key  Key
		args string
		want string
	}{
		{notes, `{"collection":"secrets",` + sentences + `}`, `asks for "secrets"`},
		{notes, `{` + sentences + `}`, "leaves collection unset"},
		{notes, `{"collection":"notes","limit":3}`, "takes no limit argument"},
		{hermes, `{"collection":"notes",` + sentences + `}`, "takes no user_id argument"},
	} {
		if err := c.key.AuthorizeToolArgs("check_against_sources", []byte(c.args)); !errors.Is(err, ErrDenied) || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s %s: %v, want a denial saying %q", c.key.ID, c.args, err, c.want)
		}
	}
	if err := notes.AuthorizeToolArgs("check_against_sources", []byte(`{"collection":"notes",`+sentences+`}`)); err != nil {
		t.Errorf("notes in its own collection: %v", err)
	}
}
