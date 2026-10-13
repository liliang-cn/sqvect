package agentmem_test

import (
	"context"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/agentmem"
)

type scopedUpdateReflector struct{ target string }

func (r scopedUpdateReflector) Consolidate(_ context.Context, facts, _ []*agentmem.Memory) ([]agentmem.Observation, error) {
	return []agentmem.Observation{{Content: "new observation", EvidenceIDs: []string{facts[0].ID, facts[1].ID}, UpdateObsID: r.target}}, nil
}

func TestReflectRejectsTargetsOutsideItsExistingObservations(t *testing.T) {
	for _, target := range []string{"other-bank", "same-bank-fact", "missing"} {
		t.Run(target, func(t *testing.T) {
			store, cleanup := newStore(t)
			defer cleanup()
			ctx := context.Background()
			scope := agentmem.Scope{Type: agentmem.ScopeUser, ID: "alice"}
			for _, m := range []*agentmem.Memory{
				{ID: "fact-1", Scope: scope, Content: "first fact"},
				{ID: "same-bank-fact", Scope: scope, Content: "second fact"},
				{ID: "other-bank", Scope: agentmem.Scope{Type: agentmem.ScopeUser, ID: "bob"}, Type: agentmem.TypeObservation, Content: "Bob observation"},
			} {
				if err := store.Save(ctx, m); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := store.Reflect(ctx, scope, scopedUpdateReflector{target: target}); err == nil {
				t.Error("invalid supersession target accepted")
			}
			for _, id := range []string{"other-bank", "same-bank-fact"} {
				m, err := store.Get(ctx, id)
				if err != nil || agentmem.IsStale(m) {
					t.Fatalf("unrelated memory %s modified: %+v, %v", id, m, err)
				}
			}
			observations, err := store.GetByType(ctx, agentmem.TypeObservation, 10)
			if err != nil || len(observations) != 1 {
				t.Fatalf("invalid output persisted an observation: count=%d, error=%v", len(observations), err)
			}
		})
	}
}

func TestReflectCanSupersedeAnExistingObservationInItsScope(t *testing.T) {
	store, cleanup := newStore(t)
	defer cleanup()
	ctx := context.Background()
	scope := agentmem.Scope{Type: agentmem.ScopeUser, ID: "alice"}
	for _, m := range []*agentmem.Memory{
		{ID: "fact-1", Scope: scope, Content: "first fact"},
		{ID: "fact-2", Scope: scope, Content: "second fact"},
		{ID: "previous", Scope: scope, Type: agentmem.TypeObservation, Content: "old observation"},
	} {
		if err := store.Save(ctx, m); err != nil {
			t.Fatal(err)
		}
	}
	result, err := store.Reflect(ctx, scope, scopedUpdateReflector{target: "previous"})
	if err != nil || result.Updated != 1 {
		t.Fatalf("valid supersession failed: %+v, %v", result, err)
	}
	previous, err := store.Get(ctx, "previous")
	if err != nil || previous.SupersededBy != result.NewIDs[0] {
		t.Fatalf("previous observation = %+v, error=%v", previous, err)
	}
}
