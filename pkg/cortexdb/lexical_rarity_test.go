package cortexdb

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"testing"
)

// eachLexicalBackend runs fn on a fresh no-embedder store of each backend
// configured: SQLite always, PostgreSQL when CORTEXDB_TEST_POSTGRES is set.
func eachLexicalBackend(t *testing.T, fn func(t *testing.T, db *DB)) {
	for name, db := range lexicalBackends(t) {
		t.Run(name, func(t *testing.T) { fn(t, db) })
	}
}

func saveGlobalMemories(t *testing.T, db *DB, memories map[string]string) {
	t.Helper()
	for id, content := range memories {
		if _, err := db.SaveMemory(context.Background(), MemorySaveRequest{MemoryID: id, Scope: "global", Content: content}); err != nil {
			t.Fatalf("save %s: %v", id, err)
		}
	}
}

// Most memories in a brain about CortexDB say "CortexDB". A memory that
// shares only that word with a question about CortexDB's WAL is not an
// answer, and the one that holds the question's rare words is.
func TestAMemorySharingOnlyACommonWordRanksBelowOneWithTheRareWords(t *testing.T) {
	eachLexicalBackend(t, func(t *testing.T, db *DB) {
		memories := map[string]string{
			"answer": "The WAL checkpoint was never called, so the file kept growing until the disk filled.",
			// Shares CortexDB and the rarer "checkpoint": more than one word,
			// less than the answer.
			"partial": "CortexDB release notes: checkpoint tooling for snapshots.",
		}
		for i := 0; i < 12; i++ {
			memories[fmt.Sprintf("common-%02d", i)] = fmt.Sprintf("CortexDB release %d shipped to the shared brain on schedule.", i)
		}
		saveGlobalMemories(t, db, memories)

		resp, err := db.SearchMemory(context.Background(), MemorySearchRequest{Query: "CortexDB WAL checkpoint", Scope: "global", TopK: 5})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		got := memoryIDs(resp.Results)
		if len(got) == 0 || got[0] != "answer" {
			t.Fatalf("results %v, want the WAL memory first", got)
		}
		for _, id := range got {
			if strings.HasPrefix(id, "common-") {
				t.Fatalf("results %v include a memory that shares only the common word", got)
			}
		}
		if resp.Results[0].Coverage <= 0 {
			t.Fatalf("a keyword hit carries no coverage: %+v", resp.Results[0])
		}
	})
}

// "CortexDB 是什么，能用来做什么" put a memory sharing only 是什么 first, at
// 0.82. A function word is not part of what a question asks.
func TestAMemorySharingOnlyAFunctionWordIsNotReturned(t *testing.T) {
	eachLexicalBackend(t, func(t *testing.T, db *DB) {
		saveGlobalMemories(t, db, map[string]string{
			"agent-def": "Agent 是什么：用户纠正说 Agent 就是给 LLM 加上工具调用的循环。",
			"about":     "CortexDB 是一个单文件的 AI 记忆库，可以用来做向量检索、知识图谱和 agent 记忆。",
		})
		resp, err := db.SearchMemory(context.Background(), MemorySearchRequest{Query: "CortexDB 是什么，能用来做什么", Scope: "global", TopK: 5})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		if got := memoryIDs(resp.Results); len(got) != 1 || got[0] != "about" {
			t.Fatalf("results %v, want only the memory about CortexDB", got)
		}
	})
}

// No-embedder mode keeps finding a memory by the words it shares, Chinese
// two-character words included — they are below the trigram index's reach.
func TestLexicalModeStillFindsAGoodMatch(t *testing.T) {
	eachLexicalBackend(t, func(t *testing.T, db *DB) {
		saveGlobalMemories(t, db, map[string]string{
			"wechat":  "微信机器人不会主动推送提醒，只能在用户先发消息后回复。",
			"weather": "北京机房的专线本周迁移，迁移期间延迟会升高。",
			"guitar":  "学会用 git bisect 以后，定位回归快了很多。",
		})
		resp, err := db.SearchMemory(context.Background(), MemorySearchRequest{Query: "微信机器人为什么从来不主动给我发提醒", Scope: "global", TopK: 5})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		if got := memoryIDs(resp.Results); len(got) != 1 || got[0] != "wechat" {
			t.Fatalf("results %v, want the WeChat memory alone", got)
		}
	})
}

// A keyword hit the semantic search did not return does not outrank the
// semantic search's own first place, however important and recent it is,
// and a memory both found outranks either.
func TestAKeywordOnlyHitDoesNotOutrankASemanticallyCloserOne(t *testing.T) {
	semantic := []MemorySearchHit{
		{Memory: MemoryRecord{ID: "close"}, Score: 0.41, Similarity: 0.42},
		{Memory: MemoryRecord{ID: "both"}, Score: 0.33, Similarity: 0.34},
	}
	// Scored on another scale, and lifted by importance and recency: none of
	// it may matter against the other list.
	lexical := []MemorySearchHit{
		{Memory: MemoryRecord{ID: "words", Importance: 1}, Score: 0.99, Coverage: 1},
		{Memory: MemoryRecord{ID: "both"}, Score: 0.40, Coverage: 0.5},
	}
	got := memoryIDs(fuseMemoryHits(semantic, lexical, 3))
	if want := "both close words"; strings.Join(got, " ") != want {
		t.Fatalf("fused order %v, want %s", got, want)
	}

	// The same end to end: the keyword hit is important, new and covers the
	// question; the semantic hit shares no word with it.
	memories := map[string]float64{
		"The floating address moves with the Primary when drbd-reactor fails over.": 0.62,
		"vip failover work log, see ticket":                                         0.10,
	}
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "closer.db")),
		WithEmbedder(fixedCosineEmbedder(memories)), WithVectorHealing(VectorHealOptions{Disabled: true}))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	ctx := context.Background()
	if _, err := db.SaveMemory(ctx, MemorySaveRequest{MemoryID: "closer", Scope: "global", Content: "The floating address moves with the Primary when drbd-reactor fails over."}); err != nil {
		t.Fatalf("save: %v", err)
	}
	if _, err := db.SaveMemory(ctx, MemorySaveRequest{MemoryID: "words", Scope: "global", Content: "vip failover work log, see ticket", Importance: 1}); err != nil {
		t.Fatalf("save: %v", err)
	}
	resp, err := db.SearchMemory(ctx, MemorySearchRequest{Query: "how does vip failover work", Scope: "global", TopK: 5})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if got := memoryIDs(resp.Results); len(got) != 2 || got[0] != "closer" {
		t.Fatalf("results %v, want the semantically closer memory first and the keyword one after it", got)
	}
}

// Knowledge below both floors is not returned: an unrelated document no
// longer fills the list, while a keyword match still comes back with no
// embedder at all.
func TestKnowledgeBelowTheFloorIsNotReturned(t *testing.T) {
	eachLexicalBackend(t, func(t *testing.T, db *DB) {
		ctx := context.Background()
		for id, body := range map[string]string{
			"wal":    "The SQLite WAL grew without bound because nothing ever ran wal_checkpoint; v2.111.1 truncates it.",
			"pool":   "A thin pool that fills up fails every backup volume in it at once.",
			"tagit":  "TagIt is a self-hosted team assistant addressed with @agent.",
			"recipe": "Our release checklist: tag, wait for CI, publish, announce.",
		} {
			if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: id, Content: body, Collection: "kb"}); err != nil {
				t.Fatalf("save %s: %v", id, err)
			}
		}
		search := func(query string) []string {
			t.Helper()
			resp, err := db.SearchKnowledge(ctx, KnowledgeSearchRequest{Query: query, Collection: "kb", TopK: 3, RetrievalMode: RetrievalModeLexical})
			if err != nil {
				t.Fatalf("search %q: %v", query, err)
			}
			var ids []string
			for _, h := range resp.Results {
				ids = append(ids, h.KnowledgeID)
			}
			return ids
		}
		if got := search("why did the WAL file keep growing without a checkpoint"); len(got) == 0 || got[0] != "wal" {
			t.Fatalf("results %v, want the WAL document first", got)
		}
		// Shares "release" with the checklist and nothing else.
		if got := search("which science fiction novel should I read before the release party"); len(got) != 0 {
			t.Fatalf("a question about nothing stored returned %v", got)
		}
	})
}

func TestContainsWordRespectsWordBoundaries(t *testing.T) {
	for _, tc := range []struct {
		text, word string
		want       bool
	}{
		{"that is all", "ha", false},
		{"ha failover", "ha", true},
		{"the vm (ha) moved", "ha", true},
		{"we deployed it", "deploy", true}, // a prefix, as the keyword search's deploy* matches
		{"redeploy it", "deploy", false},
		{"cortexdb的使用场景", "cortexdb", true}, // CJK runs into Latin unspaced
		{"v2.111.1 fixed", "111", true},
	} {
		if got := containsWord(tc.text, tc.word); got != tc.want {
			t.Errorf("containsWord(%q, %q) = %v, want %v", tc.text, tc.word, got, tc.want)
		}
	}
}

// A character's weight comes from the rarest-in-use word it belongs to, not
// from a bigram that straddles two words: 用场 in 使用场景 is rare for being
// nonsense. And a character no stored bigram covers weighs what an unseen
// word does, so a question about something the store has never heard of is
// not covered by the one word it shares.
func TestRarityWeightsCJKCharactersByTheWordTheyBelongTo(t *testing.T) {
	q := newRarityQuery("使用场景", nil)
	s := newRarityScorer(q, map[string]int{"使用": 50, "用场": 1, "场景": 10}, 1000)
	if s.charW[1] != rarityCJKCharWeight*rarityIDF(50, 1000) {
		t.Errorf("用 weighs %v, want the weight of 使用", s.charW[1])
	}
	if s.charW[2] != rarityCJKCharWeight*rarityIDF(10, 1000) {
		t.Errorf("场 weighs %v, want the weight of 场景", s.charW[2])
	}

	q = newRarityQuery("北京明天会下雨吗", nil)
	s = newRarityScorer(q, map[string]int{"北京": 3, "明天": 40}, 1000)
	if m := s.coverage("北京机房本周迁移"); s.answers(m, DefaultMemoryKeywordFloor) {
		t.Errorf("one shared place name answers a question about rain: %+v", m)
	}
}

// The floors are options; 0 turns one off.
func TestRelevanceFloorsAreConfigurable(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "opts.db")),
		WithMemoryKeywordFloor(0), WithKnowledgeFloors(0.5, -1))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	if db.memoryKeywordFloor != 0 || db.knowledgeVectorFloor != 0.5 || db.knowledgeKeywordFloor != DefaultKnowledgeKeywordFloor {
		t.Fatalf("floors = %v %v %v", db.memoryKeywordFloor, db.knowledgeVectorFloor, db.knowledgeKeywordFloor)
	}
	t.Setenv("CORTEXDB_MEMORY_KEYWORD_FLOOR", "1.5")
	if _, err := RelevanceFloorOptionsFromEnv(); err == nil {
		t.Fatal("an out-of-range floor was accepted")
	}
}
