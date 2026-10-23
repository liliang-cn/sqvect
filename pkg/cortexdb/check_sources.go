package cortexdb

import (
	"context"
	"errors"
	"fmt"
	"index/suffixarray"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"

	"github.com/liliang-cn/cortexdb/v2/pkg/core"
)

// The bounds of one check: a synthesis is checked sentence by sentence, each
// sentence against the few passages it cites. Each distinct source costs a
// few reads and its folded text an index, and each lookup of a fact or of
// one of its equivalents in one cited source costs the bytes of its folded
// form, so they are bounded per call as well. The folded bounds hold
// whatever NFKD writes: ﷺ is 3 bytes stored and 33 folded. A fact, a key
// or a value is at most 256 bytes once folded.
const (
	maxCheckSentences         = 500
	maxCheckSources           = 50
	maxCheckDistinctSources   = 200
	maxCheckSourceBytes       = 4 << 20
	maxCheckFoldedSourceBytes = 8 << 20
	maxCheckEquivalents       = 1000
	maxCheckAlternatives      = 20
	maxCheckEquivalentBytes   = 64 << 10
	maxCheckFormLength        = 256
	maxCheckLookups           = 1 << 20
	maxCheckLookupBytes       = 64 << 20
)

// CheckAgainstSourcesRequest asks, for each sentence, which of its facts no
// source it cites contains.
type CheckAgainstSourcesRequest struct {
	// Collection, when set, is the collection every source must be in; a
	// source of another collection is missing.
	Collection string            `json:"collection,omitempty"`
	Sentences  []SourcedSentence `json:"sentences"`
	// Equivalents maps a fact to what the sources may write it as, another
	// language's name for instance: {"rgpd": ["gdpr"]}. The engine holds no
	// such list of its own.
	Equivalents map[string][]string `json:"equivalents,omitempty"`
}

// SourcedSentence is one sentence and the sources it cites, each a document
// id or a chunk id. Sources may be left out: a sentence with no source does
// not hold.
type SourcedSentence struct {
	Text    string   `json:"text"`
	Sources []string `json:"sources,omitempty"`
}

// SentenceCheck is the verdict on one sentence. Missing lists its facts that
// no source contains, SourcesMissing the sources that could not be read. A
// sentence holds when it cites a source, every source was read, and no fact
// is missing.
type SentenceCheck struct {
	Holds          bool     `json:"holds"`
	Missing        []string `json:"missing"`
	SourcesMissing []string `json:"sources_missing"`
}

// CheckAgainstSourcesResponse has one result per sentence, in the order of
// the request.
type CheckAgainstSourcesResponse struct {
	Results []SentenceCheck `json:"results"`
	Counts  CheckCounts     `json:"counts"`
}

// CheckCounts counts the sentences checked and those that held.
type CheckCounts struct {
	Sentences int `json:"sentences"`
	Hold      int `json:"hold"`
}

// CheckAgainstSources checks each sentence against the text of the sources it
// cites: every number and every marked word of the sentence must be in one of
// them. Deterministic, no model. It says what is missing, not whether the
// sentence is true.
func (db *DB) CheckAgainstSources(ctx context.Context, req CheckAgainstSourcesRequest) (CheckAgainstSourcesResponse, error) {
	if len(req.Sentences) > maxCheckSentences {
		return CheckAgainstSourcesResponse{}, fmt.Errorf("cortexdb: check against sources: at most %d sentences per call", maxCheckSentences)
	}
	var ids []string
	for _, s := range req.Sentences {
		if len(s.Sources) > maxCheckSources {
			return CheckAgainstSourcesResponse{}, fmt.Errorf("cortexdb: check against sources: at most %d sources per sentence", maxCheckSources)
		}
		ids = append(ids, s.Sources...)
	}
	ids = documentIDsOf(ids)
	if len(ids) > maxCheckDistinctSources {
		return CheckAgainstSourcesResponse{}, fmt.Errorf("cortexdb: check against sources: at most %d distinct sources per call", maxCheckDistinctSources)
	}
	alternatives, err := foldEquivalents(req.Equivalents)
	if err != nil {
		return CheckAgainstSourcesResponse{}, fmt.Errorf("cortexdb: check against sources: %w", err)
	}
	texts, err := db.sourceTexts(ctx, req.Collection, ids)
	if err != nil {
		return CheckAgainstSourcesResponse{}, fmt.Errorf("cortexdb: check against sources: %w", err)
	}
	sources, err := foldSources(texts)
	if err != nil {
		return CheckAgainstSourcesResponse{}, fmt.Errorf("cortexdb: check against sources: %w", err)
	}
	budget := newCheckBudget()
	resp := CheckAgainstSourcesResponse{Results: make([]SentenceCheck, 0, len(req.Sentences)), Counts: CheckCounts{Sentences: len(req.Sentences)}}
	for _, s := range req.Sentences {
		if err := ctx.Err(); err != nil {
			return CheckAgainstSourcesResponse{}, err
		}
		check := SentenceCheck{SourcesMissing: []string{}}
		var cited []*checkSource
		for _, id := range s.Sources {
			source, ok := sources[id]
			if !ok {
				if !slices.Contains(check.SourcesMissing, id) {
					check.SourcesMissing = append(check.SourcesMissing, id)
				}
				continue
			}
			if !slices.Contains(cited, source) {
				cited = append(cited, source)
			}
		}
		if check.Missing, err = missingFacts(s.Text, cited, alternatives, budget); err != nil {
			return CheckAgainstSourcesResponse{}, fmt.Errorf("cortexdb: check against sources: %w", err)
		}
		check.Holds = len(s.Sources) > 0 && len(check.SourcesMissing) == 0 && len(check.Missing) == 0
		if check.Holds {
			resp.Counts.Hold++
		}
		resp.Results = append(resp.Results, check)
	}
	return resp, nil
}

// sourceTexts is the one read of a check: the text of each source, by chunk
// id or by document id. A source is left out, and so missing, when it does
// not exist, when its document is expired or hidden by the review gate, when
// it is outside the collection asked for, or outside the caller's scope.
// The text kept is bounded per call in stored bytes, counted on the sources
// kept alone, so that a refusal says nothing of a source the caller cannot
// read.
func (db *DB) sourceTexts(ctx context.Context, collection string, ids []string) (map[string]string, error) {
	// loadToolChunks leaves out a chunk of an expired or hidden document.
	chunks, err := db.GraphRAGTools().loadToolChunks(ctx, nil, ids, false, 0)
	if err != nil {
		return nil, err
	}
	type source struct{ text, collection string }
	found := make(map[string]source, len(ids))
	for _, c := range chunks {
		emb, err := db.store.GetByID(ctx, c.ID)
		if err != nil {
			return nil, err
		}
		found[c.ID] = source{c.Content, emb.Collection}
	}
	var documents []string
	for _, id := range ids {
		if _, isChunk := found[id]; isChunk {
			continue
		}
		doc, err := db.store.GetDocument(ctx, id)
		if errors.Is(err, core.ErrNotFound) {
			continue
		}
		if err != nil {
			return nil, err
		}
		docCollection, err := db.knowledgeCollection(ctx, id)
		if err != nil {
			return nil, err
		}
		found[id] = source{doc.Content, docCollection}
		documents = append(documents, id)
	}
	// A document read by id is held to the expiry and the review gate here,
	// as its chunks are in loadToolChunks.
	hidden := db.hiddenAmong(ctx, nil, documents)
	texts := make(map[string]string, len(found))
	size := 0
	for id, src := range found {
		_, isHidden := hidden[id]
		if isHidden || collection != "" && src.collection != collection || !knowledgeReadPermitted(ctx, src.collection) {
			continue
		}
		if size += len(src.text); size > maxCheckSourceBytes {
			return nil, fmt.Errorf("at most %d bytes (4 MB) of source text per call", maxCheckSourceBytes)
		}
		texts[id] = src.text
	}
	return texts, nil
}

// checkSource is the text of one source, folded and indexed once per call,
// however many sentences cite it: its tokens that hold a digit, for the
// facts that hold one, and a suffix array, for the words. scanned counts the
// bytes of the forms looked up in it.
type checkSource struct {
	numbers map[string]bool
	text    string
	index   *suffixarray.Index
	words   map[string]bool
	scanned int
}

// foldSources folds the text of each source, bounds the folded bytes of the
// call as it goes, and only then indexes them.
func foldSources(texts map[string]string) (map[string]*checkSource, error) {
	sources := make(map[string]*checkSource, len(texts))
	size := 0
	for id, text := range texts {
		folded, ok := foldAtMost(text, maxCheckFoldedSourceBytes-size)
		if !ok {
			return nil, fmt.Errorf("at most %d bytes (8 MB) of source text per call, once folded", maxCheckFoldedSourceBytes)
		}
		size += len(folded)
		sources[id] = &checkSource{numbers: map[string]bool{}, text: folded, words: map[string]bool{}}
	}
	for id, source := range sources {
		for _, token := range numberTokens(texts[id]) {
			source.numbers[foldFact(token)] = true
		}
		source.index = suffixarray.New([]byte(source.text))
	}
	return sources, nil
}

// checkForm is a fact or an equivalent, folded, with whether it holds a
// digit, decided once.
type checkForm struct {
	text  string
	digit bool
}

func formOf(folded string) checkForm {
	return checkForm{folded, strings.ContainsFunc(folded, unicode.IsDigit)}
}

// holds reports whether the source contains a form. A form that holds a
// digit is found only whole, as a token of the source; any other is found
// inside a longer word. A lookup reads its form a bounded number of times.
func (s *checkSource) holds(form checkForm) bool {
	s.scanned += len(form.text)
	if form.digit {
		return s.numbers[form.text]
	}
	found, seen := s.words[form.text]
	if !seen {
		found = len(s.index.Lookup([]byte(form.text), 1)) > 0
		s.words[form.text] = found
	}
	return found
}

// checkBudget is what a call may still look up: a number of lookups, and
// the bytes of their forms.
type checkBudget struct{ lookups, bytes int }

func newCheckBudget() *checkBudget {
	return &checkBudget{maxCheckLookups, maxCheckLookupBytes}
}

// take takes one lookup of form from the budget, or says why it cannot.
func (b *checkBudget) take(form checkForm) error {
	if b.lookups == 0 {
		return fmt.Errorf("at most %d lookups of facts in sources per call: cite fewer sources or pass fewer equivalents", maxCheckLookups)
	}
	if len(form.text) > b.bytes {
		return fmt.Errorf("at most %d bytes of facts looked up in sources per call, once folded: cite fewer sources, pass fewer equivalents or shorten the sentences", maxCheckLookupBytes)
	}
	b.lookups--
	b.bytes -= len(form.text)
	return nil
}

// foldEquivalents bounds the equivalents of a request and folds them once,
// keys and values alike (equivalentPiece). The values are bounded per fact
// once folded, so that spellings of one key share the bound. An empty value
// stands for nothing.
func foldEquivalents(equivalents map[string][]string) (map[string][]checkForm, error) {
	if len(equivalents) > maxCheckEquivalents {
		return nil, fmt.Errorf("at most %d equivalents per call", maxCheckEquivalents)
	}
	alternatives := make(map[string][]checkForm, len(equivalents))
	size := 0
	tooBig := fmt.Errorf("at most %d bytes of equivalents per call", maxCheckEquivalentBytes)
	for fact, alts := range equivalents {
		if size += len(fact); size > maxCheckEquivalentBytes {
			return nil, tooBig
		}
		key, err := equivalentPiece(fact)
		if err != nil {
			return nil, err
		}
		for _, alt := range alts {
			if size += len(alt); size > maxCheckEquivalentBytes {
				return nil, tooBig
			}
			if alt, err = equivalentPiece(alt); err != nil {
				return nil, err
			}
			if alt != "" {
				if alternatives[key] = append(alternatives[key], formOf(alt)); len(alternatives[key]) > maxCheckAlternatives {
					return nil, fmt.Errorf("at most %d values per fact, once its spellings are folded", maxCheckAlternatives)
				}
			}
		}
	}
	return alternatives, nil
}

// equivalentPiece trims, folds and bounds a key or a value of the
// equivalents: a name or a number, at most 256 bytes once trimmed and
// folded, since a lookup costs the folded bytes. Punctuation and white space
// at its ends go, so a value of a space stands for nothing. One that holds a
// digit is cut as a sentence is and must give one piece.
func equivalentPiece(s string) (string, error) {
	for trimmed := ""; trimmed != s; {
		trimmed, s = s, strings.TrimSpace(trimPiece(s))
	}
	if strings.ContainsFunc(s, unicode.IsDigit) {
		var pieces []string
		eachPiece(s, func(piece string) { pieces = append(pieces, piece) })
		if len(pieces) != 1 {
			return "", fmt.Errorf("an equivalent that holds a digit is found only whole, so it is one piece: %q", s)
		}
		s = pieces[0]
	}
	folded, ok := foldAtMost(s, maxCheckFormLength)
	if !ok {
		return "", fmt.Errorf("at most %d bytes per equivalent key or value, once trimmed and folded", maxCheckFormLength)
	}
	return folded, nil
}

// missingFacts lists the facts of sentence that none of the cited sources
// holds, directly or through one of its alternatives, in the order of the
// sentence. A fact longer than 256 bytes once folded is missing with no
// lookup. Each lookup in one source is taken from the budget before it is
// made, and the call is refused at the first that does not fit. Pure: the
// texts are read by sourceTexts and indexed by foldSources, the
// alternatives folded by foldEquivalents.
func missingFacts(sentence string, cited []*checkSource, alternatives map[string][]checkForm, budget *checkBudget) ([]string, error) {
	missing := []string{}
	var refused error
	held := func(form checkForm) bool {
		return slices.ContainsFunc(cited, func(s *checkSource) bool {
			if refused = budget.take(form); refused != nil {
				return true
			}
			return s.holds(form)
		})
	}
	for _, fact := range sentenceFacts(sentence) {
		folded, short := foldAtMost(fact, maxCheckFormLength)
		if !short || !held(formOf(folded)) && !slices.ContainsFunc(alternatives[folded], held) {
			missing = append(missing, fact)
		}
		if refused != nil {
			return nil, refused
		}
	}
	return missing, nil
}

// sentenceFacts lists the facts of a sentence, each once, in its order: its
// tokens that hold a digit (numberTokens), and its marked words
// (isWordFact) among the rest. The first word is no fact when only its
// initial is a capital, since every sentence starts with one.
func sentenceFacts(sentence string) []string {
	var facts []string
	// A fact too long to fold is known by its text, which no folded form can
	// equal: a folded form folds to itself, so it would be short.
	seen := map[string]bool{}
	add := func(fact string) {
		key, short := foldAtMost(fact, maxCheckFormLength)
		if !short {
			key = fact
		}
		if !seen[key] {
			seen[key] = true
			facts = append(facts, fact)
		}
	}
	// A sentence that starts in Han, kana or hangul has no capital to discount.
	at := strings.IndexFunc(sentence, unicode.IsLetter)
	letter, _ := utf8.DecodeRuneInString(sentence[max(at, 0):])
	first := !isCJK(letter)
	eachPiece(sentence, func(piece string) {
		if strings.ContainsFunc(piece, unicode.IsDigit) {
			add(piece)
			first = false
			return
		}
		for _, token := range factTokens(piece) {
			if isWordFact(token) && (!first || !onlyInitialCapital(token)) {
				add(token)
			}
			first = false
		}
	})
	return facts
}

// numberTokens lists the tokens of a text that hold a digit, each a fact
// found only whole.
func numberTokens(text string) []string {
	var tokens []string
	eachPiece(text, func(piece string) {
		if strings.ContainsFunc(piece, unicode.IsDigit) {
			tokens = append(tokens, piece)
		}
	})
	return tokens
}

// eachPiece cuts a text at white space and at Han, kana and hangul, which
// are written without spaces, and gives each piece stripped of the sentence
// punctuation at its ends. Whatever else a piece holds stays in it: 12/03,
// 12:30, GPT-4, −5, 10⁶, 1'234.56, 5% are each one piece. A space within a
// line joins two pieces after a digit and before a group of exactly three
// digits, as in 1 234,56, and a single Han, kana or hangul rune between two
// digits does not cut, as in 2025年3月12日.
func eachPiece(text string, yield func(string)) {
	for _, line := range strings.FieldsFunc(text, isLineBreak) {
		fields := strings.FieldsFunc(line, unicode.IsSpace)
		for i := 0; i < len(fields); i++ {
			first := i
			for i+1 < len(fields) && endsWithDigit(fields[i]) && startsWithGroupOfThree(fields[i+1]) {
				i++
			}
			// Joined once, so a long run costs its length and no more.
			for _, piece := range cutCJK(strings.Join(fields[first:i+1], " ")) {
				if piece = trimPiece(piece); piece != "" {
					yield(piece)
				}
			}
		}
	}
}

func isLineBreak(r rune) bool {
	return strings.ContainsRune("\n\r\v\f\u0085\u2028\u2029", r)
}

func endsWithDigit(s string) bool {
	r, _ := utf8.DecodeLastRuneInString(s)
	return unicode.IsDigit(r)
}

// startsWithGroupOfThree reports whether s starts with exactly three digits.
func startsWithGroupOfThree(s string) bool {
	end := strings.IndexFunc(s, func(r rune) bool { return !unicode.IsDigit(r) })
	if end < 0 {
		end = len(s)
	}
	return utf8.RuneCountInString(s[:end]) == 3
}

// cutCJK cuts a field at each run of Han, kana or hangul, except a single
// rune with a digit on each side: the 年 and 月 of 2025年3月12日.
func cutCJK(field string) []string {
	var pieces []string
	start := 0
	for i := 0; i < len(field); {
		r, size := utf8.DecodeRuneInString(field[i:])
		if !isCJK(r) {
			i += size
			continue
		}
		end := len(field)
		if n := strings.IndexFunc(field[i:], func(r rune) bool { return !isCJK(r) }); n >= 0 {
			end = i + n
		}
		before, _ := utf8.DecodeLastRuneInString(field[:i])
		after, _ := utf8.DecodeRuneInString(field[end:])
		if end > i+size || !unicode.IsDigit(before) || !unicode.IsDigit(after) {
			pieces = append(pieces, field[start:i])
			start = end
		}
		i = end
	}
	return append(pieces, field[start:])
}

// trimPiece strips a piece of the punctuation at its ends: at its start,
// opening brackets and quotes, ¿ ¡ and the CJK sentence marks; at its end,
// the sentence punctuation (sentenceMarks), closing brackets and quotes.
// Format characters go at either end. Any other sign stays: .5 is not 5,
// and 5% is not 5.
func trimPiece(piece string) string {
	return strings.TrimRightFunc(strings.TrimLeftFunc(piece, isOpening), isClosing)
}

// sentenceMarks is the sentence punctuation, in Latin, CJK, Arabic and
// Devanagari forms.
const sentenceMarks = ".,;:!?…\"'。、，．！？；：｡､،؛؟۔।॥"

func isOpening(r rune) bool {
	return unicode.In(r, unicode.Ps, unicode.Pi, unicode.Pf, unicode.Cf) || strings.ContainsRune("\"'¿¡、。，；：！？", r)
}

func isClosing(r rune) bool {
	return unicode.In(r, unicode.Pe, unicode.Pi, unicode.Pf, unicode.Cf) || strings.ContainsRune(sentenceMarks, r)
}

// factTokens cuts a piece with no digit into runs of letters, marks, points
// and hyphens, with the points and hyphens trimmed from their ends.
func factTokens(piece string) []string {
	var tokens []string
	var run []rune
	flush := func() {
		if token := strings.Trim(string(run), ".-"); token != "" {
			tokens = append(tokens, token)
		}
		run = run[:0]
	}
	for _, r := range piece {
		if unicode.IsLetter(r) || unicode.IsMark(r) || r == '.' || r == '-' {
			run = append(run, r)
		} else {
			flush()
		}
	}
	flush()
	return tokens
}

// isCJK reports Han, kana and hangul, with the Common marks written inside
// or between kana words: ー ・, their half-width forms ｰ ･, ゛ ゜, 〆, the
// wave dash 〜, ゠ and the repeat marks 〱 〲. The half-width voicing marks
// ﾞ ﾟ fold away as marks. The full-width tilde ～ folds to ~ and stays in a
// piece, as ~ does, so a fact glued to it is missed.
func isCJK(r rune) bool {
	return unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul) || strings.ContainsRune(kanaMarks, r)
}

const kanaMarks = "\u30fc\u30fb\uff70\uff65\u309b\u309c\u3006\u301c\u30a0\u3031\u3032"

// isWordFact reports whether a word with no digit is worth checking: a name
// or an acronym. It has a capital after its first letter, or an initial
// capital and four letters or more.
func isWordFact(token string) bool {
	first, size := utf8.DecodeRuneInString(token)
	capitalAfter := strings.ContainsFunc(token[size:], unicode.IsUpper)
	return capitalAfter || unicode.IsUpper(first) && utf8.RuneCountInString(token) >= 4
}

// onlyInitialCapital reports whether a token is a capital followed by small
// letters alone.
func onlyInitialCapital(token string) bool {
	for i, r := range token {
		if i == 0 && !unicode.IsUpper(r) || i > 0 && !unicode.IsLower(r) {
			return false
		}
	}
	return true
}

// foldFact folds a fact, its equivalents and the source text alike: case
// and accents (NFKD, marks dropped), format characters dropped, ß to ss, a
// decimal comma to a point, every dash and the minus sign to -, and ’ ‘ to '. Superscripts and the
// other number signs that are not digits are kept as they are: 10⁶ is not
// 106.
func foldFact(s string) string {
	folded, _ := foldAtMost(s, -1)
	return folded
}

// foldAtMost is foldFact that stops, and reports false, once the folded form
// passes limit bytes, so that what NFKD writes is bounded as it is written.
// A negative limit is none.
func foldAtMost(s string, limit int) (string, bool) {
	var b strings.Builder
	over := func() bool { return limit >= 0 && b.Len() > limit }
	var nfkd norm.Iter
	for s = strings.ToLower(s); s != ""; {
		end := strings.IndexFunc(s, isNumberSign)
		if end < 0 {
			end = len(s)
		}
		for nfkd.InitString(norm.NFKD, s[:end]); !nfkd.Done(); {
			for _, r := range string(nfkd.Next()) {
				switch {
				case r < utf8.RuneSelf && r != ',':
					b.WriteByte(byte(r))
				case unicode.In(r, unicode.Mn, unicode.Cf):
				case r == 'ß':
					b.WriteString("ss")
				case r == ',':
					b.WriteByte('.')
				case unicode.Is(unicode.Pd, r) || r == '\u2212':
					b.WriteByte('-')
				case r == '\u2019' || r == '\u2018':
					b.WriteByte('\'')
				default:
					b.WriteRune(r)
				}
			}
			if over() {
				return "", false
			}
		}
		if end < len(s) {
			r, size := utf8.DecodeRuneInString(s[end:])
			b.WriteRune(r)
			end += size
		}
		if over() {
			return "", false
		}
		s = s[end:]
	}
	return b.String(), true
}

func isNumberSign(r rune) bool {
	return unicode.Is(unicode.No, r)
}
