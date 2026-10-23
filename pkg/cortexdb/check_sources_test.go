package cortexdb

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// sameFacts compares two fact lists as sets, so a test names what it means
// and not the order the extraction meets it in.
func sameFacts(t *testing.T, what string, got, want []string) {
	t.Helper()
	g, w := slices.Clone(got), slices.Clone(want)
	slices.Sort(g)
	slices.Sort(w)
	if !slices.Equal(g, w) {
		t.Errorf("%s: facts %q, want %q", what, got, want)
	}
}

// missingFrom is missingFacts on raw texts and equivalents, folded and
// indexed as a call does it, every text cited.
func missingFrom(sentence string, texts []string, equivalents map[string][]string) []string {
	byID := make(map[string]string, len(texts))
	for i, text := range texts {
		byID[fmt.Sprint(i)] = text
	}
	var cited []*checkSource
	for _, source := range folded(byID) {
		cited = append(cited, source)
	}
	alternatives, err := foldEquivalents(equivalents)
	if err != nil {
		panic(err)
	}
	missing, err := missingFacts(sentence, cited, alternatives, newCheckBudget())
	if err != nil {
		panic(err)
	}
	return missing
}

// folded is foldSources on texts within its bound.
func folded(texts map[string]string) map[string]*checkSource {
	sources, err := foldSources(texts)
	if err != nil {
		panic(err)
	}
	return sources
}

// countingContext counts the calls to Err, and fails from the call failAt
// on when failAt is set.
type countingContext struct {
	context.Context
	calls, failAt int
}

func (c *countingContext) Err() error {
	if c.calls++; c.failAt > 0 && c.calls >= c.failAt {
		return context.Canceled
	}
	return c.Context.Err()
}

// The facts of a sentence are its tokens that hold a digit, and its marked
// words: a capital after the first letter, or an initial capital on a word
// of four letters or more. Common words are not checked this way.
func TestAFactIsANumberOrAMarkedWord(t *testing.T) {
	sameFacts(t, "marked words",
		sentenceFacts("Le modèle MusicLM de DeepMind bat GPT-4, llama_3 et un LLM x86 à Oslo avec 1,5 point."),
		[]string{"GPT-4", "llama_3", "x86", "1,5", "MusicLM", "DeepMind", "LLM", "Oslo"})
	sameFacts(t, "a short capitalised word", sentenceFacts("Il dit que Rio est loin."), nil)
}

// A word with no digit is cut at anything but letters, marks, points and
// hyphens, so a plus sign joins two names and each is checked. A point or a
// hyphen at either end of a word is trimmed: "DeepMind-" in "DeepMind- und
// Google-Teams" is DeepMind.
func TestATokenIsCutAtAPlusAndTrimmedOfEdgeHyphens(t *testing.T) {
	sameFacts(t, "a trailing hyphen", sentenceFacts("Die DeepMind- und Google-Teams arbeiten."), []string{"DeepMind", "Google-Teams"})
	sameFacts(t, "a leading hyphen", sentenceFacts("Die Teams von -DeepMind arbeiten."), []string{"Teams", "DeepMind"})
	if got := missingFrom("Le duo DeepMind+Google sort.", []string{"DeepMind and Google"}, nil); len(got) != 0 {
		t.Errorf("missing %q, want none: a plus sign joins two names", got)
	}
}

// A fact is listed once, under the first form the sentence writes it in:
// Müller and MULLER are one fact.
func TestAFactIsListedOnceUnderItsFoldedForm(t *testing.T) {
	sameFacts(t, "one fact", sentenceFacts("Er kennt Müller und MULLER."), []string{"Müller"})
}

// A sentence starts with a capital whether or not its first word is a name,
// so the first word is no fact when only its initial is a capital. An
// acronym in first place stays one.
func TestTheFirstWordIsNotAFactWhenOnlyItsInitialIsCapital(t *testing.T) {
	sameFacts(t, "a capitalised first word", sentenceFacts("Lyria arrive en 2025."), []string{"2025"})
	sameFacts(t, "an acronym first", sentenceFacts("IBM publie Granite."), []string{"IBM", "Granite"})
	sameFacts(t, "a quoted first word", sentenceFacts("« Lyria » arrive."), nil)
	sameFacts(t, "a first word with a digit", sentenceFacts("Llama3 tops the chart."), []string{"Llama3"})
	sameFacts(t, "a word after a first number", sentenceFacts("2025 Lyria sortit."), []string{"2025", "Lyria"})
	sameFacts(t, "a first word with a mark", sentenceFacts("Ka\u0331ren arrive."), []string{"Ka\u0331ren"})
}

// Case and accents are folded on the sentence, the sources and the
// equivalents alike: Müller is MULLER, Straße is STRASSE, either way round.
func TestFoldingIgnoresCaseAndAccentsOnBothSides(t *testing.T) {
	for _, c := range []struct{ sentence, source string }{
		{"Der Bericht nennt Müller und die Straße.", "BERICHT: MULLER, STRASSE"},
		{"Der BERICHT nennt MULLER und die STRASSE.", "Bericht über Müller in der Straße"},
	} {
		if got := missingFrom(c.sentence, []string{c.source}, nil); len(got) != 0 {
			t.Errorf("%q against %q: missing %q", c.sentence, c.source, got)
		}
	}
	// A mark with no precomposed letter stays in its word, and folds away.
	sameFacts(t, "a combining mark", sentenceFacts("Le texte cite Ka\u0331ren."), []string{"Ka\u0331ren"})
	if got := missingFrom("Le texte cite Ka\u0331ren.", []string{"about karen"}, nil); len(got) != 0 {
		t.Errorf("missing %q, want none", got)
	}
	if got := missingFrom("Der Bericht nennt Mueller.", []string{"Bericht über Müller"}, nil); !slices.Equal(got, []string{"Mueller"}) {
		t.Errorf("Mueller is not Müller unless the request says so: missing %q", got)
	}
}

// A token that holds a digit is one fact, the whole run between white
// space, stripped of the sentence punctuation at its ends. Everything else
// in it stays: separators, slashes, colons, dashes, signs, superscripts,
// letters. It is found only whole, as a token of the source after the same
// folding: each form holds against itself and misses against a changed
// part.
func TestATokenWithADigitIsOneFactFoundOnlyWhole(t *testing.T) {
	sameFacts(t, "a date and a time", sentenceFacts("Signé le 12/03/2025 à 12:30, avec GPT‑4 et −5 %."), []string{"12/03/2025", "12:30", "GPT‑4", "−5"})
	sameFacts(t, "an amount", sentenceFacts("Es kostet 1.234,56 Euro."), []string{"1.234,56", "Euro"})
	sameFacts(t, "a group of three after a space", sentenceFacts("Le prix est de 1 234,56 euros."), []string{"1 234,56"})
	sameFacts(t, "quotes and brackets", sentenceFacts("En « 2025 » (et 2026) puis “2027”."), []string{"2025", "2026", "2027"})
	for _, c := range []struct{ form, changed string }{
		{"1,234.56", "1,234.57"}, {"1.234,56", "1.234,57"}, {"2,500.75", "2,500.76"},
		{"1'234.56", "7'234.56"}, {"1’234.56", "7’234.56"}, {"1‘234.56", "7‘234.56"},
		{"1·234.56", "7·234.56"}, {"١٬٢٣٤", "٢٬٢٣٤"}, {"1´234.56", "7´234.56"},
		{"1 234,56", "7 234,56"}, {"1 234,56", "7 234,56"}, {"1 234,56", "7 234,56"},
		{"1 234,56", "7 234,56"}, {"1 234", "7 234"}, {"1　234", "7　234"},
		{"12/03/2025", "13/03/2025"}, {"12.03.2025", "13.03.2025"}, {"2025-03-12", "2025-03-13"},
		{"12:30", "13:30"}, {"1/4", "3/4"}, {"3–5", "3–6"}, {"3.5", "3.6"},
		{"−5", "5"}, {"5", "−5"}, {"-5", "+5"}, {"10⁶", "10⁹"}, {"10", "10³"}, {"5", ".5"},
		{"GPT-4", "GPT-40"}, {"GPT‐4", "GPT‐3"}, {"GPT‑4", "GPT-3"}, {"GPT–4", "GPT–40"}, {"GPT-4", "GPT-4o"},
		{"x86", "x87"}, {"Llama3", "Llama2"}, {"4o", "14o"}, {"2025-08", "2025-081"},
		{"7", "17"}, {"7", "7.5"}, {"3", "3.5"}, {"1", "1,000"}, {"1", "1'234"}, {"234", "1'234"}, {"234", "1’234"},
		{"1", "1 234"}, {"234", "1 234"}, {"4", "GPT-4"}, {"4", "GPT‐4"}, {"4", "GPT‑4"}, {"4", "GPT–4"},
		{"4", "4-bit"}, {"3", "Llama3"}, {"86", "x86"}, {"2025", "2025-08"}, {"1,234", "1234"}, {"1'234", "1 234"}, {"10⁶", "106"},
		{"5%", "5‰"}, {"5%", "5"}, {"5′", "5″"}, {"٥٪", "٧٪"}, {"12§", "12"}, {"٢٣٤", "١ ٢٣٤"},
		{"2025年3月12日", "2025年3月13日"}, {"3時30分", "3時31分"}, {"2025", "2025年3月"},
	} {
		sentence := "Le chiffre est " + c.form + "."
		if got := missingFrom(sentence, []string{sentence}, nil); len(got) != 0 {
			t.Errorf("%q against itself: missing %q, want none", c.form, got)
		}
		source := "Le chiffre est " + c.changed + ", pour un article et douze autres."
		if got := missingFrom(sentence, []string{source}, nil); len(got) != 1 {
			t.Errorf("%q against %q: missing %q, want it", c.form, c.changed, got)
		}
	}
	// The leading group of an amount is no fact of its own, so it is not
	// found standing alone elsewhere in the source.
	for _, c := range []struct{ sentence, source string }{
		{"Le prix est de CHF 1'234.56.", "Le prix est de CHF 7'234.56, pour 1 article."},
		{"Le prix est de 1 234,56 euros.", "Le prix est de 7 234,56 euros, pour 1 article."},
		{"Signed on 12/03/2025.", "Signed on 13/03/2025 by 12 people."},
		{"The meeting is at 12:30.", "The meeting is at 13:30 for 12 people."},
		{"It costs 1/4 of it.", "It costs 3/4 of it, for 1 item."},
		{"The model GPT\u20114 is out.", "GPT-3 is out, with 4 more."},
		{"Revenue rose 5%.", "We opened 5 shops; revenue rose 7%."},
		{"2025年3月12日に署名。", "2025年3月13日に12人が署名。"},
	} {
		if got := missingFrom(c.sentence, []string{c.source}, nil); len(got) != 1 {
			t.Errorf("%q against %q: missing %q, want one fact", c.sentence, c.source, got)
		}
	}
	// One folding on both sides: every dash and the minus sign are -, ' and
	// ’ are one, a decimal comma is a point (so 1,234 is 1.234), full-width
	// digits are digits, and the sentence punctuation at a token's ends is
	// no part of it.
	for _, c := range []struct{ form, source string }{
		{"GPT‑4", "GPT-4"}, {"GPT-4", "GPT‐4"}, {"−5", "-5"}, {"3–5", "3-5"},
		{"1'234.56", "1’234.56"}, {"1,5", "1.5"}, {"1,234", "1.234"}, {"２０２５", "2025"},
		{"2025", "« 2025 »"}, {"2025", "(2025)"}, {"2025", "“2025”,"}, {"2025", "2025..."}, {"2025", "\"2025\""}, {"2025", "»2025«"},
		{"1 234", "1 234"}, {"10²", "10²"}, {"1'234.56", "1‘234.56"},
		{"2025", "¿2025?"}, {"2025", "¡2025!"}, {"2025", "、2025。"}, {"2025", "：2025；"}, {"2025", "2025、"},
		{"2025", "\u200f2025\u200e"}, {"2025", "\ufeff2025"}, {"2025", "20\u200b25"}, {"2025", "2025.\u200f"}, {"2025", "\u200f(2025)"},
		{"2025", "年2025"}, {"2025", "2025年"}, {"2025", "x年2025"},
	} {
		sentence := "Le chiffre est " + c.form + "."
		if got := missingFrom(sentence, []string{"Le chiffre est " + c.source + " ici."}, nil); len(got) != 0 {
			t.Errorf("%q against %q: missing %q, want none", c.form, c.source, got)
		}
	}
	// A space within a line joins two pieces only after a digit and before a
	// group of exactly three digits.
	for _, source := range []string{"1 2345 articles", "1 23 articles", "en 1 7 modèles", "1\n234 articles",
		"1\u2028234 articles", "1\u2029234 articles", "1\u0085234 articles", "1\r234 articles",
		"1\v234 articles", "1\f234 articles"} {
		if got := missingFrom("Il y a 1 article.", []string{source}, nil); len(got) != 0 {
			t.Errorf("1 against %q: missing %q, want none", source, got)
		}
	}
	// A number sign at the start of a piece is kept outside NFKD as well.
	if got := missingFrom("Il vaut ²5.", []string{"Il vaut 25."}, nil); !slices.Equal(got, []string{"²5"}) {
		t.Errorf("²5 against 25: missing %q, want ²5", got)
	}
	// The full-width tilde folds to ~ and stays in its piece: a number
	// glued to it is missed, which fails safe.
	if got := missingFrom("ア～2台。", []string{"2台。"}, nil); !slices.Equal(got, []string{"～2"}) {
		t.Errorf("a number after ～: missing %q, want ～2", got)
	}
	// The join reads the field just before the group: "1 234x 567" is two
	// pieces, 1 234x and 567.
	if got := missingFrom("Il y a 567 articles.", []string{"1 234x 567"}, nil); len(got) != 0 {
		t.Errorf("567 after 1 234x: missing %q, want none", got)
	}
	if got := missingFrom("Le prix est 234.", []string{"234 euros"}, nil); len(got) != 0 {
		t.Errorf("234 after a word: missing %q, want none", got)
	}
	// Han, kana and hangul cut a piece, except between two digits: the CJK
	// marks of a date or a time are part of it, and a CJK sentence mark
	// before a number is stripped.
	sameFacts(t, "a CJK date", sentenceFacts("2025年3月12日に署名。"), []string{"2025年3月12"})
	sameFacts(t, "two runes between digits", sentenceFacts("2025年に12人が署名した。"), []string{"2025", "12"})
	for _, c := range []struct{ sentence, source string }{
		{"2025年に12人が署名した。", "署名は、2025年。12人が来た。"},
		{"価格は500円。", "価格：500円"},
		{"売上は12%増えた。", "前年比。12%増"},
		{"サーバー2台を買った。", "2台のサーバーを買った。"},
		{"東京・2025年に。", "2025年に東京で。"},
		{"ｻｰﾊﾞｰ2台を買った。", "2台を買った。"},
		{"東京･2025年に。", "2025年に東京で。"},
		{"締め〆2025年に。", "2025年に締め。"},
		{"ガ゛2台。", "2台。"},
		{"パ゜2台。", "2台。"},
		{"ｶﾞﾟ2台。", "2台。"},
		{"ﾊﾞ2台。", "2台。"},
		{"ア〜2台。", "2台。"},
		{"ア゠2台。", "2台。"},
		{"ア〱2台。", "2台。"},
		{"ア〲2台。", "2台。"},
		{"10時〜12時に。", "12時に開始、10時から"},
	} {
		if got := missingFrom(c.sentence, []string{c.source}, nil); len(got) != 0 {
			t.Errorf("%q against %q: missing %q, want none", c.sentence, c.source, got)
		}
	}
	// Each sentence mark is stripped at the end of a piece, and each opening
	// mark at its start.
	for _, mark := range ".,;:!?…\"'。、，．！？；：｡､،؛؟۔।॥)]}»”’" {
		if got := missingFrom("Le chiffre est 2025.", []string{"en 2025" + string(mark) + " ici"}, nil); len(got) != 0 {
			t.Errorf("2025%c: missing %q, want none", mark, got)
		}
	}
	for _, mark := range "\"'¿¡、。，；：！？([{«“‘" {
		if got := missingFrom("Le chiffre est 2025.", []string{"en " + string(mark) + "2025 ici"}, nil); len(got) != 0 {
			t.Errorf("%c2025: missing %q, want none", mark, got)
		}
	}
	// A format character is no part of a word either: a soft hyphen folds away.
	if got := missingFrom("Le Microsoft est là.", []string{"Micro\u00adsoft"}, nil); len(got) != 0 {
		t.Errorf("a soft hyphen: missing %q, want none", got)
	}
	// A superscript is no digit: "5 km²" states 5, not 2.
	sameFacts(t, "km²", sentenceFacts("Area is 5 km²."), []string{"5"})
	// A word with no digit is found inside a longer one.
	if got := missingFrom("Le modèle Lyria sort.", []string{"lyrias"}, nil); len(got) != 0 {
		t.Errorf("Lyria against lyrias: missing %q, want none", got)
	}
}

// A run of groups of three joined across spaces is built once, not grown
// group by group: 10,000 groups take a few dozen allocations, not one per
// group.
func TestALongRunOfGroupsIsJoinedOnce(t *testing.T) {
	for _, groups := range []int{10, 10_000} {
		text := "1" + strings.Repeat(" 234", groups)
		var pieces []string
		allocs := testing.AllocsPerRun(3, func() {
			pieces = pieces[:0]
			eachPiece(text, func(piece string) { pieces = append(pieces, piece) })
		})
		if len(pieces) != 1 || pieces[0] != text || allocs > 64 {
			t.Errorf("%d groups: %d pieces, %.0f allocations, want one piece in at most 64", groups, len(pieces), allocs)
		}
	}
}

// Each source is folded and indexed on its own, and a sentence reads only
// the ones it cites.
func TestASentenceReadsOnlyTheSourcesItCites(t *testing.T) {
	sources := folded(map[string]string{"a": "a 7", "b": "b 8 Zorg", "c": "c 9"})
	for _, c := range []struct {
		cited []string
		want  []string
	}{
		{[]string{"a"}, []string{"8", "9", "Zorgblat"}},
		{[]string{"b"}, []string{"7", "9", "Zorgblat"}},
		{[]string{"a", "c"}, []string{"8", "Zorgblat"}},
	} {
		var cited []*checkSource
		for _, id := range c.cited {
			cited = append(cited, sources[id])
		}
		if got, err := missingFacts("Les valeurs 7, 8 et 9 du Zorgblat.", cited, nil, newCheckBudget()); err != nil || !slices.Equal(got, c.want) {
			t.Errorf("citing %v: missing %q (%v), want %q", c.cited, got, err, c.want)
		}
	}
	// An equivalent never reaches across two sources.
	equivalents, _ := foldEquivalents(map[string][]string{"Zorgblat": {"zorg\nblat"}})
	two := folded(map[string]string{"x": "price 5 zorg", "y": "blat bar"})
	if got, err := missingFacts("Le Zorgblat coûte 5.", []*checkSource{two["x"], two["y"]}, equivalents, newCheckBudget()); err != nil || !slices.Equal(got, []string{"Zorgblat"}) {
		t.Errorf("missing %q (%v), want Zorgblat", got, err)
	}
}

// Each lookup of a fact or of one of its equivalents in one cited source is
// counted against the bound of the call, and the call is refused past it.
// A source cited twice by a sentence is looked up once.
func TestTheLookupsAreCountedAndBounded(t *testing.T) {
	sources := folded(map[string]string{"a": "a", "b": "b"})
	cited := []*checkSource{sources["a"], sources["b"]}
	equivalents, _ := foldEquivalents(map[string][]string{"Zorgblat": {"zorg", "blat"}})
	// 7 and 8 in two sources, Zorgblat and its two equivalents in two
	// sources: 4 + 6 lookups.
	budget := &checkBudget{10, maxCheckLookupBytes}
	if got, err := missingFacts("Les valeurs 7 et 8 du Zorgblat.", cited, equivalents, budget); err != nil || len(got) != 3 || budget.lookups != 0 {
		t.Errorf("10 lookups allowed: missing %q, %v, %d left, want 3 facts, no error, none left", got, err, budget.lookups)
	}
	budget = &checkBudget{9, maxCheckLookupBytes}
	if _, err := missingFacts("Les valeurs 7 et 8 du Zorgblat.", cited, equivalents, budget); err == nil || !strings.Contains(err.Error(), "at most 1048576 lookups") {
		t.Errorf("9 lookups allowed: %v, want a refusal", err)
	}
	// A held fact stops at the first source that holds it.
	budget = &checkBudget{10, maxCheckLookupBytes}
	if _, err := missingFacts("Le chiffre 7.", []*checkSource{folded(map[string]string{"s": "7"})["s"], sources["a"]}, nil, budget); err != nil || budget.lookups != 9 {
		t.Errorf("a held fact: %v, %d left, want 9", err, budget.lookups)
	}
	// Each lookup costs the bytes of its folded form: 7, 8 (1 byte each),
	// zorgblat (8), zorg (4) and blat (4), each in two sources, make 36.
	budget = &checkBudget{maxCheckLookups, 36}
	if _, err := missingFacts("Les valeurs 7 et 8 du Zorgblat.", cited, equivalents, budget); err != nil || budget.bytes != 0 {
		t.Errorf("36 bytes allowed: %v, %d left, want none left", err, budget.bytes)
	}
	budget = &checkBudget{maxCheckLookups, 35}
	if _, err := missingFacts("Les valeurs 7 et 8 du Zorgblat.", cited, equivalents, budget); err == nil || !strings.Contains(err.Error(), "at most 67108864 bytes of facts looked up") {
		t.Errorf("35 bytes allowed: %v, want a refusal", err)
	}
}

// A lookup costs the bytes of its folded form, and the bytes looked up in a
// call are bounded however much NFKD writes: a fact longer than 256 bytes
// once folded, such as one word with a megabyte of ﷺ, is missing with no
// lookup, and the equivalents at their caps, each value of seven ﷺ, are
// refused once the bytes would pass the bound, the sources read no further
// than it.
func TestTheBytesLookedUpAreBoundedOnceFolded(t *testing.T) {
	texts := map[string]string{}
	for i := range maxCheckSources {
		texts[fmt.Sprint(i)] = "a"
	}
	var cited []*checkSource
	scanned := func() int {
		n := 0
		for _, s := range cited {
			n += s.scanned
		}
		return n
	}
	fresh := func() {
		cited = cited[:0]
		for _, s := range folded(texts) {
			cited = append(cited, s)
		}
	}
	fresh()
	word := "Zorgblat" + strings.Repeat("\ufdfa", 1<<20/3)
	budget := newCheckBudget()
	if got, err := missingFacts("Le "+word+" et le Zorgblat.", cited, nil, budget); err != nil || len(got) != 2 || got[0] != word || scanned() != 8*len(cited) || budget.lookups != maxCheckLookups-len(cited) {
		t.Errorf("a word of a megabyte of ﷺ: missing %d facts (%v), %d bytes scanned, %d lookups taken, want it and Zorgblat missing, with %d lookups of 8 bytes", len(got), err, scanned(), maxCheckLookups-budget.lookups, len(cited))
	}
	// The equivalents at their caps: as many keys as 64 KiB holds, 20
	// values each, every value of seven ﷺ.
	equivalents := map[string][]string{}
	var keys []string
	for size := 0; ; {
		key := fmt.Sprintf("Qkey%c%c", 'a'+len(keys)/26, 'a'+len(keys)%26)
		var values []string
		add := len(key)
		for v := range maxCheckAlternatives {
			values = append(values, fmt.Sprintf("w%03d", v)+strings.Repeat("\ufdfa", 7))
			add += len(values[v])
		}
		if size += add; size > maxCheckEquivalentBytes {
			break
		}
		equivalents[key] = values
		keys = append(keys, key)
	}
	alternatives, err := foldEquivalents(equivalents)
	if err != nil {
		t.Fatal(err)
	}
	fresh()
	budget = newCheckBudget()
	sentence := "Les " + strings.Join(keys, " ") + "."
	for err = nil; err == nil; {
		_, err = missingFacts(sentence, cited, alternatives, budget)
	}
	if !strings.Contains(err.Error(), "at most 67108864 bytes of facts looked up") || budget.lookups == 0 {
		t.Errorf("%d keys of 20 values: %v, %d lookups left, want the bytes refused first", len(keys), err, budget.lookups)
	}
	if spent := maxCheckLookupBytes - budget.bytes; scanned() != spent || spent > maxCheckLookupBytes || spent < maxCheckLookupBytes-maxCheckFormLength {
		t.Errorf("the equivalents: %d bytes scanned, %d taken, want them equal and within 256 of the bound", scanned(), spent)
	}
}

// A fact is at most 256 bytes once folded: one of 256 is looked up and
// found, one of 257, or of eight ﷺ (264), is missing even from a source
// that holds it, with no lookup.
func TestAFactIsAtMost256BytesOnceFolded(t *testing.T) {
	for _, c := range []struct {
		word    string
		missing bool
	}{
		{"Z" + strings.Repeat("v", maxCheckFormLength-1), false},
		{"Z" + strings.Repeat("v", maxCheckFormLength), true},
		{"Z" + strings.Repeat("é", maxCheckFormLength-1), false},
		{"Zv" + strings.Repeat("\ufdfa", 7), false},
		{"Zv" + strings.Repeat("\ufdfa", 8), true},
		{"Zv" + strings.Repeat("7\ufdfa", 7), false},
		{"Zv" + strings.Repeat("7\ufdfa", 8), true},
		{"7" + strings.Repeat("v", maxCheckFormLength-3) + "²", false},
		{"7" + strings.Repeat("v", maxCheckFormLength-2) + "²", true},
	} {
		source := folded(map[string]string{"s": "x " + c.word + " y"})["s"]
		budget := newCheckBudget()
		got, err := missingFacts("Le "+c.word+".", []*checkSource{source}, nil, budget)
		if err != nil || (len(got) == 1) != c.missing || (budget.lookups == maxCheckLookups) != c.missing {
			t.Errorf("a fact of %d bytes stored: missing %d (%v), %d lookups, want missing %v", len(c.word), len(got), err, maxCheckLookups-budget.lookups, c.missing)
		}
	}
	// Two long facts are listed once each, by their text.
	long := "Z" + strings.Repeat("v", maxCheckFormLength)
	sameFacts(t, "two long facts", sentenceFacts(long+" "+long+" "+long+"w."), []string{long, long + "w"})
}

// The fold stops as soon as it passes its bound, within the run it is
// writing: four megabytes of ﷺ, which NFKD would write as 46, are refused
// after a few times the bound in allocations, not after the whole fold.
func TestTheFoldStopsAsItPassesItsBound(t *testing.T) {
	// The bound is in bytes written, and a bound of 0 is a bound.
	for _, c := range []struct {
		text  string
		limit int
		ok    bool
	}{
		{"ab", 2, true}, {"abc", 2, false}, {"", 0, true}, {"a", 0, false}, {"a", -1, true}, {"\ufdfa", 32, false}, {"\ufdfa", 33, true},
	} {
		if _, ok := foldAtMost(c.text, c.limit); ok != c.ok {
			t.Errorf("%q within %d bytes: %v, want %v", c.text, c.limit, ok, c.ok)
		}
	}
	ligatures := strings.Repeat("\ufdfa", 4<<20/3)
	for _, limit := range []int{maxCheckFormLength, 1 << 20} {
		least := uint64(math.MaxUint64)
		for range 3 {
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			_, ok := foldAtMost(ligatures, limit)
			runtime.ReadMemStats(&after)
			if ok {
				t.Fatalf("4 MB of ﷺ folded within %d bytes", limit)
			}
			least = min(least, after.TotalAlloc-before.TotalAlloc)
		}
		if least > uint64(8*limit+64<<10) {
			t.Errorf("a fold bounded at %d bytes allocated %d bytes, want it to stop near its bound", limit, least)
		}
	}
}

// The source text of a call is bounded once folded, before it is indexed:
// 8 MB, with ﷺ counted as the 33 bytes it folds to.
func TestTheSourceTextIsBoundedOnceFolded(t *testing.T) {
	big := strings.Repeat("a", maxCheckFoldedSourceBytes-33)
	if _, err := foldSources(map[string]string{"big": big, "ligature": "\ufdfa"}); err != nil {
		t.Errorf("%d bytes folded: %v", maxCheckFoldedSourceBytes, err)
	}
	if _, err := foldSources(map[string]string{"big": big, "ligature": "\ufdfa", "byte": "a"}); err == nil || !strings.Contains(err.Error(), "at most 8388608 bytes (8 MB) of source text per call, once folded") {
		t.Errorf("%d bytes folded: %v, want a refusal", maxCheckFoldedSourceBytes+1, err)
	}
}

// The equivalents of a call are bounded: keys, values per key and bytes. A
// value is trimmed like a piece, and a value that holds a digit must be one
// piece, since it is found only whole.
func TestTheEquivalentsAreBoundedAndTrimmed(t *testing.T) {
	many := map[string][]string{}
	for i := range maxCheckEquivalents {
		many[fmt.Sprint("k", i)] = []string{"v"}
	}
	if _, err := foldEquivalents(many); err != nil {
		t.Errorf("%d keys: %v", len(many), err)
	}
	many["one-more"] = []string{"v"}
	if _, err := foldEquivalents(many); err == nil || !strings.Contains(err.Error(), "at most 1000 equivalents") {
		t.Errorf("%d keys: %v, want a refusal", len(many), err)
	}
	values := slices.Repeat([]string{"v"}, maxCheckAlternatives)
	if _, err := foldEquivalents(map[string][]string{"k": values}); err != nil {
		t.Errorf("%d values: %v", len(values), err)
	}
	if _, err := foldEquivalents(map[string][]string{"k": append(values, "v")}); err == nil || !strings.Contains(err.Error(), "at most 20 values per fact") {
		t.Errorf("%d values: %v, want a refusal", len(values)+1, err)
	}
	// The bound holds per fact once folded: keys that fold to one fact
	// share it.
	if _, err := foldEquivalents(map[string][]string{"Zorgblat": values[:10], "ZORGBLAT": values[:10]}); err != nil {
		t.Errorf("20 values over two spellings: %v", err)
	}
	if _, err := foldEquivalents(map[string][]string{"Zorgblat": values[:10], "ZORGBLAT": values[:10], "zo\u0301rgblat": values[:1]}); err == nil || !strings.Contains(err.Error(), "at most 20 values per fact") {
		t.Errorf("21 values over three spellings: %v, want a refusal", err)
	}
	// An empty value stands for nothing and counts for nothing.
	if _, err := foldEquivalents(map[string][]string{"k": append(slices.Clone(values), "", "« »")}); err != nil {
		t.Errorf("%d values and two empty ones: %v", len(values), err)
	}
	// A key or a value is at most 256 bytes: a name or a number, not a text.
	edge := strings.Repeat("v", maxCheckFormLength)
	for _, c := range []struct {
		equivalents map[string][]string
		refused     bool
	}{
		{map[string][]string{edge: {"v"}}, false},
		{map[string][]string{edge + "v": {"v"}}, true},
		{map[string][]string{"k": {edge}}, false},
		{map[string][]string{"k": {edge + "v"}}, true},
	} {
		_, err := foldEquivalents(c.equivalents)
		if refused := err != nil && strings.Contains(err.Error(), "at most 256 bytes per equivalent key or value"); refused != c.refused || !refused && err != nil {
			t.Errorf("a key or value of %d bytes: %v, want refused %v", len(edge)+btoi(c.refused), err, c.refused)
		}
	}
	// The 256 bytes are counted once trimmed and folded, since a lookup
	// costs the folded form: seven ﷺ (21 bytes) fold to 231 and eight to
	// 264, and 200 é (400 bytes) fold to 200, as do spaces around a value.
	for _, c := range []struct {
		value   string
		refused bool
	}{
		{strings.Repeat("\ufdfa", 7), false},
		{strings.Repeat("\ufdfa", 8), true},
		{strings.Repeat("\ufdfa", 85), true},
		{strings.Repeat("é", 200), false},
		{strings.Repeat(" ", 150) + edge + strings.Repeat(" ", 150), false},
	} {
		for _, equivalents := range []map[string][]string{{c.value: {"v"}}, {"k": {c.value}}} {
			_, err := foldEquivalents(equivalents)
			if refused := err != nil && strings.Contains(err.Error(), "at most 256 bytes per equivalent key or value, once trimmed and folded"); refused != c.refused || !refused && err != nil {
				t.Errorf("%q, %d bytes stored, %d folded: %v, want refused %v", c.value[:min(len(c.value), 12)], len(c.value), len(foldFact(c.value)), err, c.refused)
			}
		}
	}
	if got := missingFrom("Le RGPD.", []string{"the gdpr"}, map[string][]string{"RGPD": {strings.Repeat(" ", 300) + "gdpr"}}); len(got) != 0 {
		t.Errorf("a value of 304 bytes trimmed to 4: missing %q, want none", got)
	}
	// The bytes of the call are bounded, keys and values: 256 keys of 4
	// bytes with values of 252 make 65,536, and one byte more is refused.
	sized := func(extra int) map[string][]string {
		m := map[string][]string{}
		for i := range 256 {
			m[fmt.Sprintf("k%03d", i)] = []string{strings.Repeat("v", 252)}
		}
		m["k000"] = []string{strings.Repeat("v", 252+extra)}
		return m
	}
	if _, err := foldEquivalents(sized(0)); err != nil {
		t.Errorf("%d bytes: %v", maxCheckEquivalentBytes, err)
	}
	if _, err := foldEquivalents(sized(1)); err == nil || !strings.Contains(err.Error(), "at most 65536 bytes of equivalents") {
		t.Errorf("%d bytes: %v, want a refusal", maxCheckEquivalentBytes+1, err)
	}
	// A key counts with no value as well: 256 keys of 256 bytes, then one
	// more byte.
	keys := map[string][]string{}
	for i := range 256 {
		keys[fmt.Sprintf("%03d", i)+strings.Repeat("v", maxCheckFormLength-3)] = nil
	}
	if _, err := foldEquivalents(keys); err != nil {
		t.Errorf("keys of %d bytes with no value: %v", maxCheckEquivalentBytes, err)
	}
	keys["z"] = nil
	if _, err := foldEquivalents(keys); err == nil || !strings.Contains(err.Error(), "at most 65536 bytes of equivalents") {
		t.Errorf("keys of %d bytes with no value: %v, want a refusal", maxCheckEquivalentBytes+1, err)
	}
	// A key is trimmed and cut as a value is.
	if got := missingFrom("Le GPT-4.", []string{"gpt4"}, map[string][]string{"« GPT-4. »": {"gpt4"}}); len(got) != 0 {
		t.Errorf("a trimmed key: missing %q, want none", got)
	}
	if got := missingFrom("Il vaut 5.", []string{"Il vaut 2016"}, map[string][]string{"5": {"« 2016. »"}}); len(got) != 0 {
		t.Errorf("a trimmed value: missing %q, want none", got)
	}
	if got := missingFrom("Il vaut 5.", []string{"2016年に"}, map[string][]string{"5": {"2016年"}}); len(got) != 0 {
		t.Errorf("a value cut to its piece: missing %q, want none", got)
	}
	if got := missingFrom("Le RGPD.", []string{"the gdpr"}, map[string][]string{"RGPD": {"(GDPR).", "« (DSGVO) »"}}); len(got) != 0 {
		t.Errorf("a trimmed word: missing %q, want none", got)
	}
	if got := missingFrom("Le RGPD.", []string{"the dsgvo"}, map[string][]string{"RGPD": {"« (DSGVO) »"}}); len(got) != 0 {
		t.Errorf("a word trimmed twice: missing %q, want none", got)
	}
	if got := missingFrom("Le RGPD.", []string{"a b"}, map[string][]string{"RGPD": {" ", "« »"}}); !slices.Equal(got, []string{"RGPD"}) {
		t.Errorf("a value of white space: missing %q, want RGPD", got)
	}
	if _, err := foldEquivalents(map[string][]string{"GDPR": {"Regulation 2016/679"}}); err == nil || !strings.Contains(err.Error(), "one piece") {
		t.Errorf("a value with a digit and a space: %v, want a refusal", err)
	}
	if _, err := foldEquivalents(map[string][]string{"RGPD 2016": {"GDPR"}}); err == nil || !strings.Contains(err.Error(), "one piece") {
		t.Errorf("a key with a digit and a space: %v, want a refusal", err)
	}
	if got := missingFrom("Le RGPD.", []string{"Regulation 2016/679"}, map[string][]string{"RGPD": {"2016/679"}}); len(got) != 0 {
		t.Errorf("a value with a digit: missing %q, want none", got)
	}
}

// What a fact may be written as in another language comes from the request,
// folded like the rest, under the same rule as the fact. An empty
// equivalent stands for nothing.
func TestAnEquivalentFromTheRequestStandsForTheFact(t *testing.T) {
	equivalents := map[string][]string{"RGPD": {"", "DSGVO", "GDPR"}}
	if got := missingFrom("Le RGPD s'applique.", []string{"The GDPR applies."}, equivalents); len(got) != 0 {
		t.Errorf("missing %q, want none: GDPR stands for RGPD", got)
	}
	if got := missingFrom("Le RGPD s'applique.", []string{"The rules apply."}, equivalents); !slices.Equal(got, []string{"RGPD"}) {
		t.Errorf("missing %q, want RGPD", got)
	}
	// The key and its values are folded: an accented key stands for the
	// plain fact, an accented value for the plain source.
	if got := missingFrom("Er kennt Muller.", []string{"He knows Miller."}, map[string][]string{"Müller": {"Miller"}}); len(got) != 0 {
		t.Errorf("accented key: missing %q, want none", got)
	}
	if got := missingFrom("Er wohnt in Genf.", []string{"Il habite à Geneve."}, map[string][]string{"Genf": {"Genève"}}); len(got) != 0 {
		t.Errorf("accented value: missing %q, want none", got)
	}
	// An equivalent is found only whole, as the fact is: GPT4 is not in GPT40.
	if got := missingFrom("Le modèle GPT-4 sort.", []string{"GPT40 and 4 more"}, map[string][]string{"GPT-4": {"GPT4"}}); !slices.Equal(got, []string{"GPT-4"}) {
		t.Errorf("an equivalent inside a longer number: missing %q, want GPT-4", got)
	}
}

// The engine knows no language: without an equivalent RGPD is not GDPR, and
// a capitalised word after a full stop inside the sentence is a fact.
func TestTheEngineHoldsNoWordList(t *testing.T) {
	if got := missingFrom("Le RGPD s'applique.", []string{"The GDPR applies."}, nil); !slices.Equal(got, []string{"RGPD"}) {
		t.Errorf("missing %q, want RGPD", got)
	}
	sameFacts(t, "a second sentence", sentenceFacts("La CNIL a dit. Elle confirme."), []string{"CNIL", "Elle"})
}

// Han, kana and hangul are written without spaces, so a run of them is cut
// from the Latin and the digits beside it and is never a fact: what stays
// checkable is the numbers and the Latin names.
func TestHanKanaAndHangulRunsAreNoFactAndAreSplitFromLatin(t *testing.T) {
	sameFacts(t, "han", sentenceFacts("谷歌于2025年发布Lyria。"), []string{"2025", "Lyria"})
	sameFacts(t, "kana", sentenceFacts("グーグルはGeminiを発表した"), []string{"Gemini"})
	sameFacts(t, "hangul", sentenceFacts("구글은Gemini를 발표했다"), []string{"Gemini"})
	if got := missingFrom("谷歌于2025年发布Lyria。", []string{"谷歌在2025年发布了Lyria模型。"}, nil); len(got) != 0 {
		t.Errorf("missing %q, want none", got)
	}
}

// Full-width digits and letters are the ASCII ones, in the sentence and in
// the sources.
func TestFullwidthFormsAreASCIIOnBothSides(t *testing.T) {
	sameFacts(t, "full width", sentenceFacts("Le modèle sort en ２０２５ avec ＧＰＴ."), []string{"２０２５", "ＧＰＴ"})
	if got := missingFrom("Le modèle GPT sort en 2025.", []string{"ＧＰＴ ２０２５"}, nil); len(got) != 0 {
		t.Errorf("full-width source: missing %q, want none", got)
	}
}

func checkBrain(t *testing.T, name string, opts ...Option) (*DB, *GraphRAGToolbox, context.Context) {
	t.Helper()
	db := openReviewBrain(t, name, opts...)
	return db, db.GraphRAGTools(), context.Background()
}

func ingestSource(t *testing.T, tools *GraphRAGToolbox, ctx context.Context, req ToolIngestDocumentRequest) {
	t.Helper()
	if _, err := tools.IngestDocument(ctx, req); err != nil {
		t.Fatalf("ingest %s: %v", req.DocumentID, err)
	}
}

func checkOne(t *testing.T, db *DB, ctx context.Context, collection, text string, sources ...string) SentenceCheck {
	t.Helper()
	resp, err := db.CheckAgainstSources(ctx, CheckAgainstSourcesRequest{Collection: collection, Sentences: []SourcedSentence{{Text: text, Sources: sources}}})
	if err != nil {
		t.Fatalf("check %q: %v", text, err)
	}
	return resp.Results[0]
}

const lyria = "Lyria RealTime was released by Google DeepMind in 2025, in August."

// A source is a document by its id or a chunk by its id; the results follow
// the order of the request and the counts say how many held.
func TestCheckAgainstSourcesReadsDocumentsAndChunks(t *testing.T) {
	db, tools, ctx := checkBrain(t, "check-read")
	ingestSource(t, tools, ctx, ToolIngestDocumentRequest{DocumentID: "a", Content: lyria, Collection: "c1"})
	resp, err := db.CheckAgainstSources(ctx, CheckAgainstSourcesRequest{Collection: "c1", Sentences: []SourcedSentence{
		{Text: "Google DeepMind released Lyria RealTime in 2025.", Sources: []string{"a"}},
		{Text: "Google DeepMind released Lyria RealTime in 2025.", Sources: []string{"chunk:a:000"}},
		{Text: "Google released MusicLM in 2023.", Sources: []string{"a", "chunk:a:000"}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	body, _ := json.Marshal(resp)
	want := `{"results":[{"holds":true,"missing":[],"sources_missing":[]},{"holds":true,"missing":[],"sources_missing":[]},` +
		`{"holds":false,"missing":["MusicLM","2023"],"sources_missing":[]}],"counts":{"sentences":3,"hold":2}}`
	if string(body) != want {
		t.Errorf("got  %s\nwant %s", body, want)
	}
	// Another source read in the same call does not count for a sentence
	// that does not cite it.
	ingestSource(t, tools, ctx, ToolIngestDocumentRequest{DocumentID: "m", Content: "Google released MusicLM in 2023.", Collection: "c1"})
	resp, err = db.CheckAgainstSources(ctx, CheckAgainstSourcesRequest{Collection: "c1", Sentences: []SourcedSentence{
		{Text: "Google released MusicLM in 2023.", Sources: []string{"a"}},
		{Text: "Google released MusicLM in 2023.", Sources: []string{"m"}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.Results; got[0].Holds || !slices.Equal(got[0].Missing, []string{"MusicLM", "2023"}) || !got[1].Holds {
		t.Errorf("got %+v, want the first to miss MusicLM and 2023 and the second to hold", got)
	}
}

// A missing source is named once, however often the sentence cites it.
func TestAMissingSourceIsNamedAndTheSentenceDoesNotHold(t *testing.T) {
	db, tools, ctx := checkBrain(t, "check-missing")
	ingestSource(t, tools, ctx, ToolIngestDocumentRequest{DocumentID: "a", Content: lyria, Collection: "c1"})
	got := checkOne(t, db, ctx, "c1", "Google DeepMind released Lyria.", "a", "nope", "chunk:a:009", "nope")
	if got.Holds || !slices.Equal(got.SourcesMissing, []string{"nope", "chunk:a:009"}) || len(got.Missing) != 0 {
		t.Errorf("got %+v, want a sentence that does not hold, naming nope and chunk:a:009", got)
	}
}

func TestASentenceWithNoSourceDoesNotHold(t *testing.T) {
	db, _, ctx := checkBrain(t, "check-none")
	if got := checkOne(t, db, ctx, "", "it is so."); got.Holds || len(got.Missing) != 0 || len(got.SourcesMissing) != 0 {
		t.Errorf("got %+v, want a sentence with no fact and no source that does not hold", got)
	}
}

// A source of another collection is missing, exactly as a source that does
// not exist; with no collection, every collection is read.
func TestASourceOfAnotherCollectionIsMissing(t *testing.T) {
	db, tools, ctx := checkBrain(t, "check-collection")
	ingestSource(t, tools, ctx, ToolIngestDocumentRequest{DocumentID: "b", Content: lyria, Collection: "c2"})
	got := checkOne(t, db, ctx, "c1", "Google DeepMind released Lyria.", "b", "chunk:b:000")
	if got.Holds || !slices.Equal(got.SourcesMissing, []string{"b", "chunk:b:000"}) {
		t.Errorf("collection c1: got %+v, want both sources of c2 missing", got)
	}
	if got := checkOne(t, db, ctx, "", "Google DeepMind released Lyria.", "b", "chunk:b:000"); !got.Holds {
		t.Errorf("no collection: got %+v, want it to hold", got)
	}
}

// An expired document, and under the review gate a proposed one, is not a
// source, by its id or by its chunk's.
func TestAnExpiredOrProposedSourceIsMissing(t *testing.T) {
	db, tools, ctx := checkBrain(t, "check-expired")
	ingestSource(t, tools, ctx, ToolIngestDocumentRequest{DocumentID: "old", Content: lyria, ExpiresAt: strPtr(pastInstant())})
	ingestSource(t, tools, ctx, ToolIngestDocumentRequest{DocumentID: "live", Content: lyria})
	got := checkOne(t, db, ctx, "", "Google DeepMind released Lyria.", "old", "chunk:old:000", "live")
	if got.Holds || !slices.Equal(got.SourcesMissing, []string{"old", "chunk:old:000"}) {
		t.Errorf("expired: got %+v, want old and its chunk missing", got)
	}

	db, tools, ctx = checkBrain(t, "check-proposed", WithReviewGate())
	ingestSource(t, tools, ctx, ToolIngestDocumentRequest{DocumentID: "p", Content: lyria, ReviewStatus: ReviewProposed})
	ingestSource(t, tools, ctx, ToolIngestDocumentRequest{DocumentID: "ok", Content: lyria})
	got = checkOne(t, db, ctx, "", "Google DeepMind released Lyria.", "p", "chunk:p:000", "ok")
	if got.Holds || !slices.Equal(got.SourcesMissing, []string{"p", "chunk:p:000"}) {
		t.Errorf("proposed: got %+v, want p and its chunk missing", got)
	}
}

// Under a caller scope, a source in a collection the caller may not read is
// missing, by its id or by its chunk's, even when the request names no
// collection.
func TestACallerScopeOutsideTheCollectionSeesNoSource(t *testing.T) {
	db, tools, ctx := checkBrain(t, "check-scope")
	ingestSource(t, tools, ctx, ToolIngestDocumentRequest{DocumentID: "mine", Content: lyria, Collection: "c1"})
	ingestSource(t, tools, ctx, ToolIngestDocumentRequest{DocumentID: "theirs", Content: lyria, Collection: "c2"})
	scoped := WithCallerScope(ctx, CallerScope{Key: "c1-key", Knowledge: func(c string) bool { return c == "c1" }})
	got := checkOne(t, db, scoped, "", "Google DeepMind released Lyria.", "mine", "chunk:mine:000", "theirs", "chunk:theirs:000")
	if got.Holds || !slices.Equal(got.SourcesMissing, []string{"theirs", "chunk:theirs:000"}) {
		t.Errorf("got %+v, want the sources of c2 missing", got)
	}
}

func TestTheCheckIsBounded(t *testing.T) {
	db, _, ctx := checkBrain(t, "check-bounds")
	many := make([]SourcedSentence, maxCheckSentences+1)
	if _, err := db.CheckAgainstSources(ctx, CheckAgainstSourcesRequest{Sentences: many}); err == nil || !strings.Contains(err.Error(), "at most 500 sentences") {
		t.Errorf("%d sentences: %v, want a refusal", len(many), err)
	}
	if _, err := db.CheckAgainstSources(ctx, CheckAgainstSourcesRequest{Sentences: many[:maxCheckSentences]}); err != nil {
		t.Errorf("%d sentences: %v", maxCheckSentences, err)
	}
	sources := make([]string, maxCheckSources+1)
	if _, err := db.CheckAgainstSources(ctx, CheckAgainstSourcesRequest{Sentences: []SourcedSentence{{Sources: sources}}}); err == nil || !strings.Contains(err.Error(), "at most 50 sources") {
		t.Errorf("%d sources: %v, want a refusal", len(sources), err)
	}
	// Distinct sources are bounded per call: 200 are read, 201 refused.
	// The same source cited by every sentence counts once.
	var spread []SourcedSentence
	for i := range maxCheckDistinctSources / maxCheckSources {
		ids := make([]string, maxCheckSources)
		for j := range ids {
			ids[j] = fmt.Sprintf("s%d-%d", i, j)
		}
		spread = append(spread, SourcedSentence{Sources: ids})
	}
	if _, err := db.CheckAgainstSources(ctx, CheckAgainstSourcesRequest{Sentences: spread}); err != nil {
		t.Errorf("%d distinct sources: %v", maxCheckDistinctSources, err)
	}
	spread = append(spread, SourcedSentence{Sources: []string{"one-more", "s0-0"}})
	if _, err := db.CheckAgainstSources(ctx, CheckAgainstSourcesRequest{Sentences: spread}); err == nil || !strings.Contains(err.Error(), "at most 200 distinct sources") {
		t.Errorf("%d distinct sources: %v, want a refusal", maxCheckDistinctSources+1, err)
	}
	// The source text read is bounded per call, in stored bytes, counted on
	// the sources the caller may read: Hangul, which NFKD triples, is
	// counted as stored, and an expired source is not counted.
	setContent := func(id, content string, expired bool) {
		t.Helper()
		req := ToolIngestDocumentRequest{DocumentID: id, Content: lyria}
		if expired {
			req.ExpiresAt = strPtr(pastInstant())
		}
		ingestSource(t, db.GraphRAGTools(), ctx, req)
		if _, err := db.exec(ctx, "UPDATE documents SET content = ? WHERE id = ?", content, id); err != nil {
			t.Fatal(err)
		}
	}
	check := func(ids ...string) (CheckAgainstSourcesResponse, error) {
		return db.CheckAgainstSources(ctx, CheckAgainstSourcesRequest{Sentences: []SourcedSentence{{Text: "Il y a 1 article.", Sources: ids}}})
	}
	setContent("half", strings.Repeat("a", maxCheckSourceBytes/2), false)
	setContent("half-more", strings.Repeat("b", maxCheckSourceBytes/2), false)
	if _, err := check("half", "half-more"); err != nil {
		t.Errorf("4 MB of source text: %v", err)
	}
	setContent("byte", "c", false)
	if _, err := check("half", "half-more", "byte"); err == nil || !strings.Contains(err.Error(), "at most 4194304 bytes (4 MB) of source text per call") {
		t.Errorf("4 MB and a byte of source text: %v, want a refusal", err)
	}
	setContent("hangul", strings.Repeat("한", maxCheckSourceBytes/6), false)
	if _, err := check("hangul", "half"); err != nil {
		t.Errorf("Hangul counted as stored: %v", err)
	}
	// The folded text is bounded as well: 900 KB of ﷺ fold to 9.9 MB.
	setContent("ligatures", strings.Repeat("\ufdfa", 300_000), false)
	if _, err := check("ligatures"); err == nil || !strings.Contains(err.Error(), "check against sources: at most 8388608 bytes (8 MB) of source text per call, once folded") {
		t.Errorf("900 KB of ﷺ: %v, want a refusal", err)
	}
	setContent("old-big", strings.Repeat("a", maxCheckSourceBytes+1), true)
	if resp, err := check("old-big"); err != nil || !slices.Equal(resp.Results[0].SourcesMissing, []string{"old-big"}) {
		t.Errorf("an expired source of more than 4 MB: %+v, %v, want it missing and not counted", resp, err)
	}
	// A sentence is looked up in the sources it cites alone: a ledger of
	// nearly 4 MB that another sentence cites costs it nothing, and a ledger
	// of digits costs a lookup per fact, not per match.
	ledger := strings.Repeat("1;2;3;5\n", (maxCheckSourceBytes-100)/8)
	setContent("ledger", ledger, false)
	setContent("small", "3 articles, 1 table, 2 chaises, 5 lampes", false)
	resp, err := db.CheckAgainstSources(ctx, CheckAgainstSourcesRequest{Sentences: []SourcedSentence{
		{Text: "Il y a 3 articles, 1 table, 2 chaises, 5 lampes.", Sources: []string{"small"}},
		{Text: "Les lignes disent 1;2;3;5, puis 1, 11 et 111.", Sources: []string{"ledger"}},
	}})
	if err != nil || !resp.Results[0].Holds || !slices.Equal(resp.Results[1].Missing, []string{"1", "11", "111"}) {
		t.Errorf("a ledger beside a small source: %+v, %v", resp, err)
	}
	// Missing numbers, each counted against the lookups of the call.
	numbers := func(n int) string {
		var b strings.Builder
		for i := range n {
			fmt.Fprintf(&b, " %d", 1000+i)
		}
		return "Les valeurs" + b.String() + "."
	}
	// A sentence that cites one source 50 times looks it up once: 500
	// sentences of 50 missing numbers make 25,000 lookups, not 1,250,000.
	same := make([]SourcedSentence, maxCheckSentences)
	for i := range same {
		same[i] = SourcedSentence{Text: numbers(50), Sources: slices.Repeat([]string{"small"}, maxCheckSources)}
	}
	if _, err := db.CheckAgainstSources(ctx, CheckAgainstSourcesRequest{Sentences: same}); err != nil {
		t.Errorf("one source cited %d times: %v", maxCheckSentences*maxCheckSources, err)
	}
	// Each source once, whatever the order: 500 sentences of 700 missing
	// numbers citing small, ledger, small, ledger make 700,000 lookups, not
	// 1,050,000 or more.
	for i := range same {
		same[i] = SourcedSentence{Text: numbers(700), Sources: []string{"small", "ledger", "small", "ledger"}}
	}
	if _, err := db.CheckAgainstSources(ctx, CheckAgainstSourcesRequest{Sentences: same}); err != nil {
		t.Errorf("two sources cited twice each: %v", err)
	}
	// 100,000 equivalents for one fact are refused before any lookup.
	alternatives := make([]string, 100_000)
	for i := range alternatives {
		alternatives[i] = fmt.Sprint("v", i)
	}
	for i := range same {
		same[i].Text = "Le Zorgblat est là."
	}
	if _, err := db.CheckAgainstSources(ctx, CheckAgainstSourcesRequest{Sentences: same, Equivalents: map[string][]string{"Zorgblat": alternatives}}); err == nil || !strings.Contains(err.Error(), "at most 20 values per fact") {
		t.Errorf("100,000 equivalents: %v, want a refusal", err)
	}
	// The lookups of a call are bounded: 500 sentences citing 50 sources
	// each make 1,025,000 lookups with 41 missing numbers, and are
	// answered, and 1,050,000 with 42, and are refused.
	var fifty []string
	for i := range maxCheckSources {
		id := fmt.Sprint("p", i)
		setContent(id, "rien", false)
		fifty = append(fifty, id)
	}
	for _, c := range []struct {
		numbers int
		refused bool
	}{{41, false}, {42, true}} {
		for i := range same {
			same[i] = SourcedSentence{Text: numbers(c.numbers), Sources: fifty}
		}
		_, err := db.CheckAgainstSources(ctx, CheckAgainstSourcesRequest{Sentences: same})
		if refused := err != nil && strings.Contains(err.Error(), "at most 1048576 lookups of facts in sources per call"); refused != c.refused || !refused && err != nil {
			t.Errorf("%d missing numbers in 50 sources, 500 times: %v, want refused %v", c.numbers, err, c.refused)
		}
	}
}

// The context is checked before each sentence: three sentences ask it two
// more times than one does, and a context that ends on the last sentence
// ends the call.
func TestTheSentenceLoopHeedsTheContext(t *testing.T) {
	db, tools, ctx := checkBrain(t, "check-context")
	ingestSource(t, tools, ctx, ToolIngestDocumentRequest{DocumentID: "a", Content: lyria})
	sentences := func(n int) CheckAgainstSourcesRequest {
		return CheckAgainstSourcesRequest{Sentences: slices.Repeat([]SourcedSentence{{Text: "Lyria arrive.", Sources: []string{"a"}}}, n)}
	}
	one, three := &countingContext{Context: ctx}, &countingContext{Context: ctx}
	if _, err := db.CheckAgainstSources(one, sentences(1)); err != nil {
		t.Fatal(err)
	}
	if _, err := db.CheckAgainstSources(three, sentences(3)); err != nil {
		t.Fatal(err)
	}
	if three.calls-one.calls != 2 {
		t.Errorf("the context asked %d times for three sentences and %d for one, want two more", three.calls, one.calls)
	}
	ending := &countingContext{Context: ctx, failAt: three.calls}
	if resp, err := db.CheckAgainstSources(ending, sentences(3)); !errors.Is(err, context.Canceled) || len(resp.Results) != 0 {
		t.Errorf("a context ended on the last sentence: %+v, %v, want context.Canceled", resp, err)
	}
}

// A source that cannot be read is an error, never a missing source.
func TestACheckWhoseSourcesCannotBeReadFails(t *testing.T) {
	for _, c := range []struct {
		name, breakIt, source string
	}{
		{"documents", "ALTER TABLE documents RENAME COLUMN content TO content_hidden", "a"},
		{"chunk collection", "ALTER TABLE collections RENAME COLUMN name TO name_hidden", "chunk:a:000"},
		{"document collection", "ALTER TABLE collections RENAME COLUMN name TO name_hidden", "a"},
	} {
		db, tools, ctx := checkBrain(t, "check-broken")
		ingestSource(t, tools, ctx, ToolIngestDocumentRequest{DocumentID: "a", Content: lyria, Collection: "c1"})
		if _, err := db.exec(ctx, c.breakIt); err != nil {
			t.Fatal(err)
		}
		if _, err := db.CheckAgainstSources(ctx, CheckAgainstSourcesRequest{Sentences: []SourcedSentence{{Text: "x", Sources: []string{c.source}}}}); err == nil {
			t.Errorf("%s: no error", c.name)
		}
	}
	db, _, ctx := checkBrain(t, "check-closed")
	_ = db.Close()
	if _, err := db.CheckAgainstSources(ctx, CheckAgainstSourcesRequest{Sentences: []SourcedSentence{{Text: "x", Sources: []string{"a"}}}}); err == nil {
		t.Error("closed brain: no error")
	}
	// Equivalents past their bound are refused before any source is read.
	tooMany := map[string][]string{"x": slices.Repeat([]string{"v"}, maxCheckAlternatives+1)}
	if _, err := db.CheckAgainstSources(ctx, CheckAgainstSourcesRequest{Sentences: []SourcedSentence{{Text: "x", Sources: []string{"a"}}}, Equivalents: tooMany}); err == nil || !strings.Contains(err.Error(), "values per fact") {
		t.Errorf("closed brain, too many equivalents: %v, want the equivalents refused first", err)
	}
}

// The tool is declared, dispatched and decodes its request, through Call
// and over MCP.
func TestCheckAgainstSourcesIsATool(t *testing.T) {
	db, tools, ctx := checkBrain(t, "check-tool")
	ingestSource(t, tools, ctx, ToolIngestDocumentRequest{DocumentID: "a", Content: lyria})
	out, err := tools.Call(ctx, "check_against_sources", json.RawMessage(`{"sentences":[{"text":"Le RGPD vise DeepMind.","sources":["a"]}],"equivalents":{"rgpd":["Lyria"]}}`))
	if err != nil {
		t.Fatal(err)
	}
	if got := out.(CheckAgainstSourcesResponse); !got.Results[0].Holds {
		t.Errorf("got %+v", got)
	}
	if _, err := tools.Call(ctx, "check_against_sources", json.RawMessage(`{"sentences":3}`)); err == nil {
		t.Error("a bad request decoded")
	}

	clientTransport, serverTransport := mcp.NewInMemoryTransports()
	serverSession, err := db.NewMCPServer(MCPServerOptions{}).Connect(ctx, serverTransport, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = serverSession.Close() }()
	session, err := mcp.NewClient(&mcp.Implementation{Name: "check-test"}, nil).Connect(ctx, clientTransport, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = session.Close() }()
	result, err := session.CallTool(ctx, &mcp.CallToolParams{Name: "check_against_sources", Arguments: map[string]any{
		"sentences": []any{map[string]any{"text": "Le modèle MusicLM sort.", "sources": []any{"a"}}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if payload, _ := json.Marshal(result.StructuredContent); !strings.Contains(string(payload), `"missing":["MusicLM"]`) {
		t.Errorf("over MCP: %s", payload)
	}
	// A sentence with no source is accepted over MCP as through Call, and
	// does not hold.
	result, err = session.CallTool(ctx, &mcp.CallToolParams{Name: "check_against_sources", Arguments: map[string]any{
		"sentences": []any{map[string]any{"text": "it is so."}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if payload, _ := json.Marshal(result.StructuredContent); result.IsError || !strings.Contains(string(payload), `"holds":false`) {
		t.Errorf("no source over MCP: %s", payload)
	}
	for _, def := range tools.Definitions() {
		if def.Name != "check_against_sources" {
			continue
		}
		items, _ := json.Marshal(def.InputSchema["properties"].(map[string]any)["sentences"].(map[string]any)["items"].(map[string]any)["required"])
		if string(items) != `["text"]` {
			t.Errorf("declared required fields of a sentence: %s, want only text", items)
		}
	}
}

func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}
