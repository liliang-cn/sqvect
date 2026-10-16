package cortexdb

import (
	"strings"
	"testing"
	"unicode"
)

func TestSplitGraphRAGTextSentenceAware(t *testing.T) {
	// Multi-sentence text over the word budget must split on sentence
	// boundaries, never mid-sentence.
	text := "Alpha bravo charlie delta echo. Foxtrot golf hotel india juliet. Kilo lima mike november oscar. Papa quebec romeo sierra tango."
	chunks := splitGraphRAGText(text, 10, 0)
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d: %v", len(chunks), chunks)
	}
	for _, c := range chunks {
		c = strings.TrimSpace(c)
		if c == "" {
			t.Fatalf("empty chunk")
		}
		// Each chunk should end at a sentence terminator (no mid-sentence cut).
		last := c[len(c)-1]
		if last != '.' && last != '!' && last != '?' {
			t.Errorf("chunk not sentence-aligned (ends %q): %q", string(last), c)
		}
		if got := len(strings.Fields(c)); got > 10 {
			t.Errorf("chunk exceeds word budget (%d): %q", got, c)
		}
	}
}

func TestSplitGraphRAGTextOverlap(t *testing.T) {
	text := "One two three four five. Six seven eight nine ten. Eleven twelve thirteen fourteen fifteen."
	chunks := splitGraphRAGText(text, 10, 5)
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}
	// With overlap, the start of chunk 2 should repeat the tail sentence of chunk 1.
	if !strings.Contains(chunks[1], "Six seven eight nine ten.") {
		t.Errorf("expected overlap sentence carried into chunk 2, got: %q", chunks[1])
	}
}

func TestSplitGraphRAGTextOversizedSentence(t *testing.T) {
	// A single sentence longer than the budget must still be split (word window).
	long := strings.Repeat("word ", 30) // 30 words, no terminator
	chunks := splitGraphRAGText(long, 10, 2)
	if len(chunks) < 2 {
		t.Fatalf("oversized single sentence should be word-windowed, got %d chunks", len(chunks))
	}
	for _, c := range chunks {
		if got := len(strings.Fields(c)); got > 10 {
			t.Errorf("word-window chunk exceeds budget (%d)", got)
		}
	}
}

func TestSplitGraphRAGTextShortReturnsOne(t *testing.T) {
	if got := splitGraphRAGText("just a short sentence.", 120, 0); len(got) != 1 {
		t.Errorf("short text should be one chunk, got %d", len(got))
	}
	if got := splitGraphRAGText("   ", 120, 0); got != nil {
		t.Errorf("blank text should be nil, got %v", got)
	}
}

// Chinese text with few spaces is one long "word" run, so the character limit
// slices it. Sliced at a fixed width it left a last chunk of ")" and windows
// that began mid-word; such scraps embed near everything and topped recall for
// short queries.
func TestCharLimitLeavesNoScrapsAndCutsAtBoundaries(t *testing.T) {
	var b strings.Builder
	for i := 0; b.Len() < 4000; i++ {
		b.WriteString("长期记忆分层架构：工作记忆、情节记忆和语义记忆各司其职（working, episodic and semantic memory）。")
	}
	text := b.String() + ")"
	for _, overlap := range []int{0, 20} {
		chunks := splitGraphRAGText(text, 120, overlap)
		if len(chunks) < 2 {
			t.Fatalf("overlap %d: expected several chunks, got %d", overlap, len(chunks))
		}
		for i, c := range chunks {
			n := len([]rune(c))
			if n < 120 {
				t.Errorf("overlap %d: chunk %d is a %d-rune scrap: %q", overlap, i, n, c)
			}
			if n > 480+120 {
				t.Errorf("overlap %d: chunk %d has %d runes, over the limit plus tail", overlap, i, n)
			}
		}
		last := []rune(chunks[len(chunks)-1])
		if string(last[len(last)-1]) != ")" {
			t.Errorf("overlap %d: the text's end was lost", overlap)
		}
		for i, c := range chunks[:len(chunks)-1] {
			r := []rune(c)
			// Ends on punctuation, or on a whole word that stands between spaces in the text.
			fields := strings.Fields(c)
			if end := r[len(r)-1]; !unicode.IsPunct(end) && !strings.Contains(text, " "+fields[len(fields)-1]+" ") {
				t.Errorf("overlap %d: chunk %d ends mid-word with %q", overlap, i, string(end))
			}
		}
	}
}
