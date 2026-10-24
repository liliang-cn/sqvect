package cortexdb

import (
	"context"
	"fmt"
	"math"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/liliang-cn/cortexdb/v2/pkg/core"
	"github.com/liliang-cn/cortexdb/v2/pkg/sqldialect"
)

// Keyword hits scored by how much of the question they cover, weighted by how
// rare each covered term is.
//
// bm25 scores a row against one query string, and the lexical paths run
// several — the question, its keywords OR-ed together, each alternate query —
// keeping each row's best. So one matched term was enough for a high score,
// and a common one did as well as any: on the shared brain "CortexDB
// 的使用场景是？" returned a DRBD lecture note that shares only 使用场景 and a
// knowledge note that mentions CortexDB in passing ("原计划存 CortexDB"), and
// in "CortexDB 是什么，能用来做什么" a memory matching only 是什么 scored 0.82
// and outranked one about CortexDB. Nothing in the score said how much of the
// question a row answers, or whether the words it shares are words every
// seventh memory contains.
//
// The score here does. A question is broken into units — its words, and for
// CJK text the characters its bigrams cover (see lexical_cjk.go) — and each
// unit weighs its inverse document frequency in the store being searched:
//
//	coverage = Σ idf(units the row contains) / Σ idf(units of the question)
//
// A row that shares only the term a third of the store contains covers next
// to nothing; one that holds the question's rare terms covers most of it.
// Document frequencies are counted with one substring scan of the searched
// set — a bucket of memories, a collection of chunks — which is the same SQL
// on SQLite and PostgreSQL (ILIKE there for LIKE) and needs no index of its
// own. A CJK term of two characters, which the trigram index cannot see, is
// counted the same way, or over the bigram path's candidates when it ran.
//
// Keywords and alternate queries a caller supplies are synonyms, not more
// question: a row that contains one is credited at half its weight, without
// enlarging what the question asks for.

// rarityExtraWeight is the credit a supplied keyword earns, as a share of
// what the same term would earn as a word of the question.
const rarityExtraWeight = 0.5

// rarityCJKCharWeight scales a CJK character's weight so that a two-character
// word, which covers two characters, weighs about as much as one English word.
const rarityCJKCharWeight = 0.5

// rarityMaxWords bounds the word units taken from the question and from the
// supplied keywords; CJK bigrams are already bounded by cjkMaxTerms.
const rarityMaxWords = 24

// rarityBM25Share is how much of a keyword hit's relevance the row's own
// bm25 (or bigram BM25) decides: enough to order rows that cover the same
// terms, too little to lift a row over one that covers more of the question.
const rarityBM25Share = 0.2

// rarityQuery is a question broken into the units coverage is counted in.
type rarityQuery struct {
	words []string // non-CJK words of the question, lowercased
	cjk   cjkQuery // CJK bigrams of the question and the characters they cover
	// extras are keyword and alternate-query terms the question itself does
	// not contain: lowercased words and CJK bigrams.
	extras []string
	// common are the words of the alternate queries that are common in their
	// match sets (see variantWords). As extras they add their weight, but they
	// are not separate words of a row.
	common map[string]bool
}

// newRarityQuery breaks a question, and the synonyms supplied with it, into
// coverage units.
func newRarityQuery(query string, synonyms []string) rarityQuery {
	q := rarityQuery{words: rarityWords(query, rarityMaxWords), cjk: cjkQueryTerms(query)}
	have := make(map[string]struct{}, len(q.words)+len(q.cjk.terms))
	for _, w := range q.words {
		have[w] = struct{}{}
	}
	for _, t := range q.cjk.terms {
		have[t] = struct{}{}
	}
	joined := strings.Join(synonyms, " ")
	add := func(term string) {
		if _, ok := have[term]; ok || len(q.extras) >= rarityMaxWords {
			return
		}
		have[term] = struct{}{}
		q.extras = append(q.extras, term)
	}
	for _, w := range rarityWords(joined, rarityMaxWords) {
		add(w)
	}
	for _, t := range cjkQueryTerms(joined).terms {
		add(t)
	}
	return q
}

// rarityWords are the non-CJK words of text, lowercased, without stopwords
// and single letters — the same tokens the keyword expansion searches for.
func rarityWords(text string, limit int) []string {
	text = strings.Map(func(r rune) rune {
		if isCJKRune(r) {
			return ' '
		}
		return r
	}, text)
	words := lexicalQueryKeywords(text)
	if len(words) > limit {
		words = words[:limit]
	}
	return words
}

// empty reports a question with nothing to count coverage in: function words
// and stopwords only. Every row then covers all of it.
func (q rarityQuery) empty() bool {
	return len(q.words) == 0 && len(q.cjk.terms) == 0 && len(q.extras) == 0
}

// patterns are the distinct strings whose document frequency the score needs.
func (q rarityQuery) patterns() []string {
	out := make([]string, 0, len(q.words)+len(q.cjk.terms)+len(q.extras))
	out = append(out, q.words...)
	out = append(out, q.cjk.terms...)
	out = append(out, q.extras...)
	return out
}

// patternsBesideBigrams are the patterns other than the question's own CJK
// bigrams, which a caller that runs the bigram path counts there.
func (q rarityQuery) patternsBesideBigrams() []string {
	out := make([]string, 0, len(q.words)+len(q.extras))
	out = append(out, q.words...)
	out = append(out, q.extras...)
	return out
}

// rarityScorer is a rarityQuery weighed against one searched set.
type rarityScorer struct {
	q      rarityQuery
	wordW  []float64 // weight of q.words[i]
	charW  []float64 // weight of each CJK content character of the question
	extraW []float64 // weight of q.extras[i]
	total  float64   // what the whole question weighs
	words  int       // separate words in the question, counted as rarityMatch.words
}

// rarityPriorRows are rows assumed to exist beside the searched set, holding
// none of the question's terms. Inverse document frequency means nothing in a
// store of three memories, where every word that occurs at all occurs in a
// third of it and weighs a fifth of a word that does not; a new user's store
// is exactly that size. Against a few thousand rows the prior changes next to
// nothing.
const rarityPriorRows = 100

// rarityIDF is BM25's inverse document frequency over the searched set and
// the prior rows: always positive, low for a term most rows contain, highest
// for one that none does.
func rarityIDF(df, n int) float64 {
	if df < 0 {
		df = 0
	}
	if n < df {
		n = df
	}
	n += rarityPriorRows
	return math.Log(1 + (float64(n)-float64(df)+0.5)/(float64(df)+0.5))
}

// newRarityScorer weighs q's units by their document frequency df in a set
// of n rows.
//
// A CJK character takes the lowest idf among the bigrams that cover it and
// occur in the set at all. The lowest, because a character belongs to one
// word, and a bigram that straddles two words (用场 in 使用场景) is rare for
// being nonsense, not for being specific. Only those that occur, for the same
// reason: a straddling bigram no row contains says nothing. A character whose
// every bigram is absent from the set belongs to a word the store has never
// seen, and weighs what such a word does — it is part of the question that no
// row can answer, which is what keeps a question about something the store
// does not hold from being "covered" by the one word it shares.
func newRarityScorer(q rarityQuery, df map[string]int, n int) *rarityScorer {
	s := &rarityScorer{q: q}
	for _, w := range q.words {
		weight := rarityIDF(df[w], n)
		s.wordW = append(s.wordW, weight)
		s.total += weight
	}
	if q.cjk.chars > 0 {
		s.charW = make([]float64, q.cjk.chars)
		seen := make([]bool, q.cjk.chars)
		for t, term := range q.cjk.terms {
			if df[term] == 0 {
				continue
			}
			weight := rarityIDF(df[term], n)
			for _, pos := range q.cjk.covers[t] {
				if !seen[pos] || weight < s.charW[pos] {
					s.charW[pos] = weight
					seen[pos] = true
				}
			}
		}
		unseen := rarityIDF(0, n)
		for pos := range s.charW {
			if !seen[pos] {
				s.charW[pos] = unseen
			}
			s.charW[pos] *= rarityCJKCharWeight
			s.total += s.charW[pos]
		}
	}
	for _, e := range q.extras {
		s.extraW = append(s.extraW, rarityExtraWeight*rarityIDF(df[e], n))
	}
	s.words = len(q.words)
	for pos, start := range q.cjk.fragmentStart {
		if start {
			s.words++
		} else if pos > 0 && runLength(q.cjk.fragmentStart, pos) == cjkMinSpan {
			s.words++
		}
	}
	// A question of function words only, asked with keywords: the keywords
	// are what it asks, at full weight.
	if s.total == 0 {
		for i := range s.extraW {
			s.extraW[i] /= rarityExtraWeight
			s.total += s.extraW[i]
		}
		s.words = len(s.extraW)
	}
	return s
}

// runLength is how many characters of its fragment end at pos, pos included.
func runLength(fragmentStart []bool, pos int) int {
	n := 1
	for i := pos; i > 0 && !fragmentStart[i]; i-- {
		n++
	}
	return n
}

// rarityMatch is what a row covers of a question.
type rarityMatch struct {
	// share is the part of the question, by weight, the row contains, in [0, 1].
	share float64
	// words counts the separate words of the question it contains, the way
	// the bigram path counts them (see cjkMinSpan): each question word or
	// keyword, each run of covered CJK characters, and a run of cjkMinSpan
	// characters or more as two.
	words int
}

// coverage is what content covers of the question. A question with nothing
// to weigh is covered whole by every row.
func (s *rarityScorer) coverage(content string) rarityMatch {
	if s.total <= 0 {
		return rarityMatch{share: 1, words: s.words}
	}
	lower := strings.ToLower(content)
	var m rarityMatch
	var got float64
	for i, w := range s.q.words {
		if containsWord(lower, w) {
			got += s.wordW[i]
			m.words++
		}
	}
	if len(s.charW) > 0 {
		covered := make([]bool, len(s.charW))
		for t, term := range s.q.cjk.terms {
			if !strings.Contains(content, term) {
				continue
			}
			for _, pos := range s.q.cjk.covers[t] {
				covered[pos] = true
			}
		}
		run := 0
		for pos, c := range covered {
			if s.q.cjk.fragmentStart[pos] {
				run = 0
			}
			if !c {
				run = 0
				continue
			}
			got += s.charW[pos]
			run++
			if run == 1 || run == cjkMinSpan {
				m.words++
			}
		}
	}
	for i, e := range s.q.extras {
		var hit bool
		if isCJKTerm(e) {
			hit = strings.Contains(content, e)
		} else {
			hit = containsWord(lower, e)
		}
		if hit {
			got += s.extraW[i]
			if !s.q.common[e] {
				m.words++
			}
		}
	}
	m.share = min(1, got/s.total)
	return m
}

// answers reports whether a match is evidence of an answer: it covers at
// least floor of the question, and more than one of its words — unless the
// question is one word, or the row covers all of it.
//
// One shared word is not an answer, however rare. 如何学会弹吉他 shares 会弹
// with every memory about a dialog that 会弹出, and the coverage of that one
// bigram is real; that it is one word, and not the one asked about, is what
// gives it away. The bigram path has always held CJK rows to this (see
// cjkMinSpan); it now holds for words and keywords too.
func (s *rarityScorer) answers(m rarityMatch, floor float64) bool {
	if m.share <= 0 || m.share < floor {
		return false
	}
	return m.words >= 2 || s.words <= 1 || m.share >= rarityWhole
}

// rarityWhole is the share above which a row covers the whole question, up to
// floating-point error.
const rarityWhole = 0.999

func isCJKTerm(term string) bool {
	r, _ := utf8.DecodeRuneInString(term)
	return isCJKRune(r)
}

// containsWord reports whether lower contains word as a word, or for a word
// of four letters or more as the start of one — "deploy" in "deployed", as
// the prefix forms of the keyword search match — but never inside one: "ha"
// is not in "that". Both sides are already lowercased.
func containsWord(lower, word string) bool {
	if word == "" {
		return false
	}
	prefixOK := utf8.RuneCountInString(word) >= 4
	for from := 0; from < len(lower); {
		i := strings.Index(lower[from:], word)
		if i < 0 {
			return false
		}
		start := from + i
		end := start + len(word)
		before, _ := utf8.DecodeLastRuneInString(lower[:start])
		after, _ := utf8.DecodeRuneInString(lower[end:])
		if (start == 0 || !isWordRune(before)) && (prefixOK || end == len(lower) || !isWordRune(after)) {
			return true
		}
		from = start + 1
	}
	return false
}

// isWordRune is a rune that continues a word for containsWord: a letter or
// digit outside the CJK scripts, which run into Latin text unspaced
// ("CortexDB的使用场景") and do not make it one word.
func isWordRune(r rune) bool {
	return (unicode.IsLetter(r) || unicode.IsDigit(r)) && !isCJKRune(r)
}

// corpusStats describes a searched set for the rarity score: how many rows it
// has, their mean length in characters, and how many rows contain each term.
type corpusStats struct {
	n      int
	avgLen float64
	df     map[string]int
}

// memoryBucketStats is corpusStats for one memory bucket.
type memoryBucketStats = corpusStats

// scanCorpusStats measures the rows that from/where select in one scan,
// counting the rows that contain each pattern.
//
// It is a substring count, so a short word's count includes rows where it is
// part of a longer one ("ha" in "that"). That only ever overstates how common
// a term is, which errs toward weighing it less.
func (db *DB) scanCorpusStats(ctx context.Context, column, from, where string, whereArgs []any, patterns []string) (corpusStats, error) {
	like := column + ` LIKE ? ESCAPE '\'`
	if d := db.Dialect(); d != nil && d.Kind() == sqldialect.Postgres {
		// LIKE is case-sensitive there and case-insensitive (for ASCII) in
		// SQLite; ILIKE makes the two count the same rows.
		like = column + ` ILIKE ? ESCAPE '\'`
	}
	var b strings.Builder
	b.WriteString("SELECT COUNT(*), COALESCE(AVG(LENGTH(" + column + ")), 0)")
	args := make([]any, 0, len(patterns)+len(whereArgs))
	for _, p := range patterns {
		b.WriteString(", COALESCE(SUM(CASE WHEN " + like + " THEN 1 ELSE 0 END), 0)")
		args = append(args, core.SubstringPattern(p))
	}
	b.WriteString(" FROM " + from)
	if where != "" {
		b.WriteString(" WHERE " + where)
	}
	args = append(args, whereArgs...)

	var n int64
	var avgLen float64
	counts := make([]int64, len(patterns))
	dest := make([]any, 0, len(counts)+2)
	dest = append(dest, &n, &avgLen)
	for i := range counts {
		dest = append(dest, &counts[i])
	}
	if err := db.queryRow(ctx, b.String(), args...).Scan(dest...); err != nil {
		return corpusStats{}, fmt.Errorf("count term frequencies: %w", err)
	}
	stats := corpusStats{n: int(n), avgLen: avgLen, df: make(map[string]int, len(patterns))}
	for i, p := range patterns {
		stats.df[p] = int(counts[i])
	}
	return stats, nil
}

// memoryBucketStatsFor measures one memory bucket, counting the patterns
// given. The CJK bigrams of a question are not among them: the bigram path
// counts those over its candidates (see searchMemoryCJK).
func (db *DB) memoryBucketStatsFor(ctx context.Context, bucketID string, patterns []string) (memoryBucketStats, error) {
	return db.scanCorpusStats(ctx, "content", "messages", "session_id = ?", []any{bucketID}, patterns)
}

// chunkRarityScorer weighs q against the knowledge chunks of a collection, or
// of every collection when it is empty.
func (db *DB) chunkRarityScorer(ctx context.Context, collection string, q rarityQuery) (*rarityScorer, error) {
	if q.empty() {
		return newRarityScorer(q, nil, 0), nil
	}
	from := "embeddings e"
	where := ""
	var args []any
	if collection != "" {
		from = "embeddings e LEFT JOIN collections c ON e.collection_id = c.id"
		where = "c.name = ?"
		args = append(args, collection)
	}
	stats, err := db.scanCorpusStats(ctx, "e.content", from, where, args, q.patterns())
	if err != nil {
		return nil, err
	}
	return newRarityScorer(q, stats.df, stats.n), nil
}

// rarityRelevance is a keyword hit's relevance: its coverage of the question,
// with the row's own bm25 relevance (in [0,1)) deciding among rows that cover
// the same.
func rarityRelevance(coverage, bm25 float64) float64 {
	return coverage * (1 - rarityBM25Share + rarityBM25Share*bm25)
}
