package cortexdb

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"slices"
	"strings"
	"sync"

	"github.com/liliang-cn/cortexdb/v2/pkg/core"
	"github.com/liliang-cn/cortexdb/v2/pkg/observability"
	"github.com/liliang-cn/cortexdb/v2/pkg/sqldialect"
)

// A variant is often a sentence ("Wer legt das Gehalt der Richter fest?"),
// and its function words would find every row of its language through the OR
// query. Which words those are is read from the rows the variant reaches,
// one word at a time: a word of a variant is common when it is in
// variantCommonShare of the rows that the variant's other words reach, and
// those rows are variantMatchFloor or more. A function word is in most rows
// that the rest of its sentence reaches, whatever share of the store its
// language is; a content word is in a few. With fewer rows than that, the
// counts cannot tell the two apart, and the word is undecided: it does not
// join the OR query and counts as a word of a row, as before this rule. So a
// variant that reaches the store through one word (a sentence in a language
// the store lacks) searches as it did. A variant of one or two words, often an
// alias ("k8s kubernetes"), is not measured: each of its words would be
// measured on the rows of the other alone, so an alias that rows name with its
// other spelling would make that spelling common. It searches as it did. No
// word list, and the same rule for every language.
//
// The rows counted are all the rows of the scope searched, proposed,
// rejected and expired ones and those the caller may not read among them.
// They only sort the variant's words; what the search returns passes the
// gate as before.
const (
	variantCommonShare = 0.5
	variantMatchFloor  = 3
	// variantMatchCap bounds the rows read for one variant. They are read in
	// rowid order on SQLite (the order rows were stored) and in id order on
	// PostgreSQL, not by rank, so past the cap the shares are those of the
	// first rows in that order, not of the best matches.
	variantMatchCap = 1000
)

// variantCommon reports whether a word that df of matched rows contain is
// common.
func variantCommon(df, matched int) bool {
	return matched >= variantMatchFloor && float64(df) >= variantCommonShare*float64(matched)
}

// variantWords are the words of a search's alternate queries, sorted by the
// rule above.
type variantWords struct {
	kept   []string        // decided and not common: join the OR query (see lexicalSearchQueries)
	common map[string]bool // common among the rows the rest of their variant reaches
}

// commonBeside is v's common words, less the words of the caller's own
// synonyms (keywords, entity names): the rule is for the words of alternate
// queries only, and a keyword counts as it always has.
func (v variantWords) commonBeside(synonyms []string) map[string]bool {
	if len(v.common) == 0 {
		return nil
	}
	own := lexicalQueryKeywords(strings.Join(synonyms, " "))
	common := make(map[string]bool, len(v.common))
	for w := range v.common {
		if !slices.Contains(own, w) {
			common[w] = true
		}
	}
	return common
}

// variantWordsOf sorts the words of each alternate query. match returns the
// content of the rows a variant's OR query finds; the rows that the other
// words of the variant reach are those among them that hold one of those
// words. When a match fails, no variant word joins and the search runs as it
// did before.
func variantWordsOf(alternateQueries []string, match func(orQuery string) ([]string, error)) variantWords {
	v := variantWords{common: map[string]bool{}}
	decided := map[string]bool{}
	for _, alternate := range alternateQueries {
		words := lexicalQueryKeywords(alternate)
		if len(words) < 3 {
			continue
		}
		contents, err := match(strings.Join(formatFTSKeywords(words, false), " OR "))
		if err != nil {
			log.Printf("cortexdb: alternate-query words left out of the keyword query: %v", err)
			return variantWords{}
		}
		held := make([][]bool, len(contents)) // held[r][i]: row r holds words[i]
		count := make([]int, len(contents))   // how many of the words row r holds
		for r, c := range contents {
			lower := strings.ToLower(c)
			held[r] = make([]bool, len(words))
			for i, w := range words {
				if containsWord(lower, w) {
					held[r][i] = true
					count[r]++
				}
			}
		}
		for i, w := range words {
			matched, df := 0, 0
			for r := range held {
				if held[r][i] {
					if count[r] > 1 {
						matched++
						df++
					}
				} else if count[r] > 0 {
					matched++
				}
			}
			if matched < variantMatchFloor {
				continue // undecided
			}
			decided[w] = true
			if variantCommon(df, matched) {
				v.common[w] = true
			}
		}
	}
	for _, w := range alternateQueryWords(alternateQueries) {
		if decided[w] && !v.common[w] {
			v.kept = append(v.kept, w)
		}
	}
	return v
}

// variantMatchSQL selects the content of the rows of table (aliased a, with
// join) that orQuery matches, at most limit of them, in rowid order on SQLite
// and id order on PostgreSQL. Only the content is read, and no rank is
// computed.
func variantMatchSQL(d sqldialect.Dialect, orQuery, index, table, join, where string, whereArgs []any, limit int) (string, []any) {
	if d != nil && d.Kind() == sqldialect.Postgres {
		cond, args, _ := core.PostgresLexicalCondition("a.content", orQuery)
		return `SELECT a.content FROM ` + table + join + ` WHERE ` + cond + ` AND ` + where + ` ORDER BY a.id LIMIT ?`,
			append(append(args, whereArgs...), limit)
	}
	index = core.CJKAwareIndex(index, orQuery)
	return `SELECT a.content FROM ` + index + ` JOIN ` + table + ` ON a.rowid = ` + index + `.rowid` + join + ` WHERE ` + index + ` MATCH ? AND ` + where + ` ORDER BY ` + index + `.rowid LIMIT ?`,
		append(append([]any{orQuery}, whereArgs...), limit)
}

// variantMatch runs variantMatchSQL.
func (db *DB) variantMatch(ctx context.Context, querySQL string, args []any) ([]string, error) {
	rows, err := db.query(ctx, querySQL, args...)
	if err != nil {
		return nil, fmt.Errorf("match variant: %w", err)
	}
	var contents []string
	var content sql.NullString
	err = eachRow(rows, func() error {
		err := rows.Scan(&content)
		contents = append(contents, content.String)
		return err
	})
	return contents, err
}

// chunkVariantWords sorts the words of the alternate queries of a search over
// the chunks of collection, or of every collection when it is empty.
func (db *DB) chunkVariantWords(ctx context.Context, collection string, alternateQueries []string) variantWords {
	return db.chunkVariantWordsUpTo(ctx, collection, alternateQueries, variantMatchCap)
}

func (db *DB) chunkVariantWordsUpTo(ctx context.Context, collection string, alternateQueries []string, limit int) variantWords {
	join, where, args := "", "1 = 1", []any(nil)
	if collection != "" {
		join, where, args = " JOIN collections c ON a.collection_id = c.id", "c.name = ?", []any{collection}
	}
	return db.memoVariantWords(ctx, "chunks\x00"+collection, alternateQueries, func(orQuery string) ([]string, error) {
		querySQL, queryArgs := variantMatchSQL(db.Dialect(), orQuery, "chunks_fts", "embeddings a", join, where, args, limit)
		return db.variantMatch(ctx, querySQL, queryArgs)
	})
}

// memoryVariantWords sorts the words of the alternate queries of a search
// over one memory bucket.
func (db *DB) memoryVariantWords(ctx context.Context, bucketID string, alternateQueries []string) variantWords {
	return db.memoryVariantWordsUpTo(ctx, bucketID, alternateQueries, variantMatchCap)
}

func (db *DB) memoryVariantWordsUpTo(ctx context.Context, bucketID string, alternateQueries []string, limit int) variantWords {
	return db.memoVariantWords(ctx, "memory\x00"+bucketID, alternateQueries, func(orQuery string) ([]string, error) {
		querySQL, args := variantMatchSQL(db.Dialect(), orQuery, "messages_fts", "messages a", "", "a.session_id = ?", []any{bucketID}, limit)
		return db.variantMatch(ctx, querySQL, args)
	})
}

// A search measures its variants once: SearchKnowledge sorts them on the
// tool path and again in its floor, SearchTextOnly once for each over-fetch.
// withVariantMemo gives a search a memo that every measure under it shares.
type variantMemoKey struct{}

type variantMemo struct {
	mu   sync.Mutex
	done map[string]variantWords
}

func withVariantMemo(ctx context.Context) context.Context {
	if _, ok := ctx.Value(variantMemoKey{}).(*variantMemo); ok {
		return ctx
	}
	return context.WithValue(ctx, variantMemoKey{}, &variantMemo{done: map[string]variantWords{}})
}

// memoVariantWords is variantWordsOf over scope, measured once per memo.
func (db *DB) memoVariantWords(ctx context.Context, scope string, alternateQueries []string, match func(orQuery string) ([]string, error)) variantWords {
	if len(alternateQueries) == 0 {
		return variantWords{}
	}
	memo, _ := ctx.Value(variantMemoKey{}).(*variantMemo)
	key := scope + "\x00" + strings.Join(alternateQueries, "\x00")
	if memo != nil {
		memo.mu.Lock()
		defer memo.mu.Unlock()
		if v, ok := memo.done[key]; ok {
			return v
		}
	}
	defer observability.Time(ctx, "variant-words")()
	v := variantWordsOf(alternateQueries, match)
	if memo != nil {
		memo.done[key] = v
	}
	return v
}
