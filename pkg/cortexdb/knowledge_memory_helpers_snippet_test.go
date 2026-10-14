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

func TestClipStringNeverSplitsACharacter(t *testing.T) {
	text := strings.Repeat("实验机见 drbd-lab-vms，", 40)
	for max := 1; max < 200; max++ {
		got := clipString(text, max)
		if !utf8.ValidString(got) {
			t.Fatalf("max %d: not valid UTF-8: %q", max, got)
		}
		if len(got) > max {
			t.Fatalf("max %d: %d bytes", max, len(got))
		}
	}
	if got := clipString("short", 100); got != "short" {
		t.Fatalf("short text: %q", got)
	}
}
