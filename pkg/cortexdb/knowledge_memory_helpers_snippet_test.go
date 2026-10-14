package cortexdb

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestCompactSnippetCutsOnACharacterBoundary(t *testing.T) {
	long := strings.Repeat("配合现代化控制面高可用", 30)
	got := compactSnippet(long)
	if !utf8.ValidString(got) {
		t.Fatalf("snippet is not valid UTF-8: %q", got)
	}
	if n := utf8.RuneCountInString(got); n != 180 {
		t.Fatalf("snippet has %d characters, want 180", n)
	}
	if !strings.HasSuffix(got, "...") {
		t.Fatalf("a cut snippet ends in an ellipsis: %q", got)
	}
	if short := "  高可用\n  集群 "; compactSnippet(short) != "高可用 集群" {
		t.Fatalf("short content: %q", compactSnippet(short))
	}
}
