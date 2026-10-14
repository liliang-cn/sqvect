package graphflow

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestSummarizeCutsOnACharacterBoundary(t *testing.T) {
	got := summarize(strings.Repeat("宽吞吐", 50), 100)
	if !utf8.ValidString(got) || utf8.RuneCountInString(got) != 103 {
		t.Fatalf("summary: %q", got)
	}
}
