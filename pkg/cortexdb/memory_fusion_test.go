package cortexdb

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"strings"
	"testing"
)

// nearShortEmbedder puts every short text, the query included, at one point
// and every long text a cosine of 0.5 away: the way a real model embeds "HA"
// near short profile memories ("the user lives in Chengdu") instead of near
// the memories about HA.
type nearShortEmbedder struct{}

func (nearShortEmbedder) Dim() int { return 2 }
func (nearShortEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	if len([]rune(text)) < 24 {
		return []float32{1, 0}, nil
	}
	return []float32{0.5, float32(math.Sqrt(0.75))}, nil
}
func (e nearShortEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, text := range texts {
		out[i], _ = e.Embed(ctx, text)
	}
	return out, nil
}

func openFusionDB(t *testing.T) *DB {
	t.Helper()
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "fuse.db")),
		WithEmbedder(nearShortEmbedder{}), WithVectorHealing(VectorHealOptions{Disabled: true}))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	ctx := context.Background()
	for i, content := range []string{"用户常住在成都", "用户的姓名是李亮", "User's name is Li Liang.", "用户名字是李亮。", "The user likes tea."} {
		if _, err := db.SaveMemory(ctx, MemorySaveRequest{MemoryID: fmt.Sprintf("profile-%d", i), Scope: "global", Content: content}); err != nil {
			t.Fatalf("save: %v", err)
		}
	}
	for i, content := range []string{
		"The brain runs as its own HA resource with VIP 192.168.123.253, separate from openclaw.",
		"After a plugin upgrade every HA daemon must restart, or failover after a power cut fails.",
		"In DRBD with drbd-reactor an HA service is a chain that follows the Primary between nodes.",
		"Extra services of a self-HA cluster must be listed in the controller config or saving drops them.",
		"The HA page shows which node runs the active controller since the October redesign.",
	} {
		if _, err := db.SaveMemory(ctx, MemorySaveRequest{MemoryID: fmt.Sprintf("ha-%d", i), Scope: "global", Content: content}); err != nil {
			t.Fatalf("save: %v", err)
		}
	}
	return db
}

// Asking "HA" used to return the user's name and city: the semantic list went
// first and filled topK, and the memories that say HA never got a slot.
func TestABareKeywordRecallsTheMemoriesThatContainIt(t *testing.T) {
	db := openFusionDB(t)
	res, err := db.SearchMemory(context.Background(), MemorySearchRequest{Query: "HA", Keywords: []string{"ha"}, Scope: "global", TopK: 5})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(res.Results) != 5 {
		t.Fatalf("got %d results, want 5", len(res.Results))
	}
	for i, hit := range res.Results {
		if !strings.HasPrefix(hit.Memory.ID, "ha-") {
			t.Errorf("rank %d is %s (%q), not a memory about HA", i+1, hit.Memory.ID, hit.Memory.Content)
		}
	}
}

func TestFuseMemoryHitsRanksByReciprocalRank(t *testing.T) {
	hit := func(id string) MemorySearchHit { return MemorySearchHit{Memory: MemoryRecord{ID: id}} }
	semantic := []MemorySearchHit{hit("s1"), hit("both"), hit("s3")}
	lexical := []MemorySearchHit{hit("l1"), hit("l2"), hit("both")}
	var got []string
	for _, h := range fuseMemoryHits(semantic, lexical, 4) {
		got = append(got, h.Memory.ID)
	}
	// "both" is on both lists; a tie between two first places goes to the keyword hit.
	if want := "both l1 s1 l2"; strings.Join(got, " ") != want {
		t.Fatalf("fused order %v, want %s", got, want)
	}
}

func TestIsBareKeyword(t *testing.T) {
	for query, want := range map[string]bool{
		"HA": true, "drbd-reactor": true, "v2.129.3": true, "大脑": true, "发版流程": true,
		"which VIP does HA use": false, "北京明天会下雨吗": false, "HA?": false, "": false,
	} {
		if got := isBareKeyword(query); got != want {
			t.Errorf("isBareKeyword(%q) = %v, want %v", query, got, want)
		}
	}
}
