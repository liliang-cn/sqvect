package cortexdb

import (
	"database/sql"
	"math"
	"strings"
	"testing"
	"unicode/utf8"
)

// coverage compares the folded forms of the question's words and of the row,
// and a synonym that is a question word without its accents is no new term.
func TestRarityCoverageComparesWordsWithoutAccents(t *testing.T) {
	q := newRarityQuery("liberte economique", nil)
	s := newRarityScorer(q, map[string]int{}, 10)
	if m := s.coverage("La liberté économique est garantie."); m.share < rarityWhole || m.words != 2 {
		t.Errorf("unaccented question on accented text: %+v, want all of it covered", m)
	}
	q = newRarityQuery("liberté économique", nil)
	s = newRarityScorer(q, map[string]int{}, 10)
	if m := s.coverage("LA LIBERTE ECONOMIQUE EST GARANTIE."); m.share < rarityWhole || m.words != 2 {
		t.Errorf("accented question on unaccented text: %+v, want all of it covered", m)
	}
	for _, pair := range [][2]string{{"liberté", "liberte"}, {"liberte", "liberté"}} {
		if q := newRarityQuery(pair[0], []string{pair[1]}); len(q.extras) != 0 {
			t.Errorf("question %q, synonym %q: extras = %q; the same word with other accents is not a synonym", pair[0], pair[1], q.extras)
		}
	}
	if q := newRarityQuery("hochschule", []string{"universität", "universitat"}); len(q.extras) != 1 {
		t.Errorf("one synonym typed two ways: extras = %q, want one", q.extras)
	}
	if q := newRarityQuery("universität universitat basel", nil); len(q.words) != 2 || q.words[0] != "universität" {
		t.Errorf("one question word typed two ways: words = %q, want [universität basel]", q.words)
	}
	for _, pair := range [][2]string{{"universitat", "Die Universität Zürich"}, {"universität", "Die Universitat Zurich"}} {
		s = newRarityScorer(newRarityQuery("hochschule", []string{pair[0]}), map[string]int{}, 10)
		// The row holds the keyword and not the question word: it is credited
		// the keyword's half weight, the question word being as rare.
		if m := s.coverage(pair[1]); m.words != 1 || math.Abs(m.share-rarityExtraWeight) > 1e-9 {
			t.Errorf("keyword %q on %q: %+v, want one word at share %.2f", pair[0], pair[1], m, rarityExtraWeight)
		}
	}
	// Each keyword is compared in its own folded form, after the question's
	// words: the row holds both keywords and neither question word.
	s = newRarityScorer(newRarityQuery("hochschule basel", []string{"universität", "zürich"}), map[string]int{}, 10)
	if m := s.coverage("Die Universitat Zurich"); m.words != 2 {
		t.Errorf("two keywords on \"Die Universitat Zurich\": %+v, want two words", m)
	}
}

// foldAccents folds what the word index folds: the Latin letters FTS5
// unicode61 reads without their diacritic, the combining marks it removes,
// and nothing else.
func TestFoldAccentsFoldsWhatTheWordIndexFolds(t *testing.T) {
	for in, want := range map[string]string{
		"àáâãäåçèéêëìíîïñòóôõöùúûüýÿ":  "aaaaaaceeeeiiiinooooouuuuyy",
		"āăąćčďēęěğīįľńňőřśšşťūůűźżžſ": "aaaccdeeegiilnnorssstuuuzzzs",
		"ß æ ø œ đ ł ð þ ı":            "ß æ ø œ đ ł ð þ ı",
		"șțǎǐǒǔơưȁ":                    "staiououa",
		"hiến pháp việt nam":           "hiến phap việt nam",
		"café ẍy":                    "cafe xy",
		"plain ascii, 123":             "plain ascii, 123",
		"cortexdb的使用场景":                "cortexdb的使用场景",
	} {
		if got := foldAccents(in); got != want {
			t.Errorf("foldAccents(%q) = %q, want %q", in, got, want)
		}
	}
}

// Every letter of Latin-1, Latin Extended-A and B and Latin Extended
// Additional, and every combining diacritical mark, as the word index reads
// it: each is indexed between two x's, read back through fts5vocab, and
// compared with the words foldAccents leaves of the same text lowercased.
// chunks_fts and messages_fts use FTS5's default tokenizer, unicode61, as the
// table here does.
func TestFoldAccentsMatchesTheWordIndex(t *testing.T) {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer db.Close()
	db.SetMaxOpenConns(1)
	for _, stmt := range []string{
		`CREATE VIRTUAL TABLE words USING fts5(content)`,
		`CREATE VIRTUAL TABLE terms USING fts5vocab(words, instance)`,
	} {
		if _, err := db.Exec(stmt); err != nil {
			t.Fatalf("%s: %v", stmt, err)
		}
	}
	var runes []rune
	for _, block := range [][2]rune{{0xc0, 0x250}, {0x300, 0x370}, {0x1e00, 0x1f00}} {
		for r := block[0]; r < block[1]; r++ {
			if utf8.ValidRune(r) {
				runes = append(runes, r)
			}
		}
	}
	for _, r := range runes {
		if _, err := db.Exec(`INSERT INTO words(rowid, content) VALUES (?, ?)`, int64(r), "x"+string(r)+"x"); err != nil {
			t.Fatalf("index %U: %v", r, err)
		}
	}
	rows, err := db.Query(`SELECT doc, term FROM terms ORDER BY doc, offset`)
	if err != nil {
		t.Fatalf("read terms: %v", err)
	}
	defer rows.Close()
	index := map[rune][]string{}
	for rows.Next() {
		var doc int64
		var term string
		if err := rows.Scan(&doc, &term); err != nil {
			t.Fatalf("scan: %v", err)
		}
		index[rune(doc)] = append(index[rune(doc)], term)
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("rows: %v", err)
	}
	differ := 0
	for _, r := range runes {
		want := strings.Join(index[r], " ")
		got := strings.Join(strings.FieldsFunc(foldAccents(strings.ToLower("x"+string(r)+"x")), func(c rune) bool { return !isWordRune(c) }), " ")
		if got != want {
			differ++
			t.Errorf("%U %q: index reads %q, foldAccents %q", r, string(r), want, got)
		}
	}
	if differ > 0 || len(index) != len(runes) {
		t.Fatalf("%d of %d runes differ; %d read back", differ, len(runes), len(index))
	}
}
