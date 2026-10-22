package geo

import "testing"

// A GeoPoint's Metadata map must not be shared with the caller in either
// direction.
//
// GetPoint says it returns "a copy to prevent external modification", but
// `copy := *point` is shallow: the returned point carries the indexed point's
// own map, so writing one key through the returned handle rewrites what the
// index will hand to everybody else. Insert had the mirror of the same hole —
// it stored the caller's map — which makes a copy on the way out worth nothing
// anyway, because the map being copied is still the caller's.
//
// Every search also returns points by value, and each of those carries the
// same map.
func TestPointMetadataIsNotSharedWithTheCaller(t *testing.T) {
	const id = "paris"
	paris := GeoPoint{
		ID:         id,
		Coordinate: Coordinate{Lat: 48.8566, Lng: 2.3522},
		Metadata:   map[string]interface{}{"country": "FR"},
	}

	for _, tc := range []struct {
		name string
		// mutate writes through whatever handle the API handed back.
		mutate func(t *testing.T, g *GeoIndex, caller map[string]interface{})
	}{
		{
			name: "the map handed to Insert",
			mutate: func(_ *testing.T, _ *GeoIndex, caller map[string]interface{}) {
				caller["country"] = "mutated by the caller"
			},
		},
		{
			name: "the map returned by GetPoint",
			mutate: func(t *testing.T, g *GeoIndex, _ map[string]interface{}) {
				got, ok := g.GetPoint(id)
				if !ok {
					t.Fatalf("GetPoint(%q) found nothing", id)
				}
				got.Metadata["country"] = "mutated through GetPoint"
			},
		},
		{
			name: "the map returned by SearchRadius",
			mutate: func(t *testing.T, g *GeoIndex, _ map[string]interface{}) {
				results, err := g.SearchRadius(paris.Coordinate, 10, Kilometers)
				if err != nil {
					t.Fatalf("SearchRadius: %v", err)
				}
				if len(results) != 1 {
					t.Fatalf("SearchRadius returned %d points, want 1", len(results))
				}
				results[0].Point.Metadata["country"] = "mutated through SearchRadius"
			},
		},
		{
			name: "the map returned by SearchKNN",
			mutate: func(t *testing.T, g *GeoIndex, _ map[string]interface{}) {
				results, err := g.SearchKNN(paris.Coordinate, 1)
				if err != nil {
					t.Fatalf("SearchKNN: %v", err)
				}
				results[0].Point.Metadata["country"] = "mutated through SearchKNN"
			},
		},
		{
			name: "the map returned by SearchBoundingBox",
			mutate: func(t *testing.T, g *GeoIndex, _ map[string]interface{}) {
				points, err := g.SearchBoundingBox(BoundingBox{MinLat: 48, MaxLat: 49, MinLng: 2, MaxLng: 3})
				if err != nil {
					t.Fatalf("SearchBoundingBox: %v", err)
				}
				if len(points) != 1 {
					t.Fatalf("SearchBoundingBox returned %d points, want 1", len(points))
				}
				points[0].Metadata["country"] = "mutated through SearchBoundingBox"
			},
		},
		{
			name: "the map returned by SearchPolygon",
			mutate: func(t *testing.T, g *GeoIndex, _ map[string]interface{}) {
				points, err := g.SearchPolygon([]Coordinate{
					{Lat: 48, Lng: 2}, {Lat: 49, Lng: 2}, {Lat: 49, Lng: 3}, {Lat: 48, Lng: 3},
				})
				if err != nil {
					t.Fatalf("SearchPolygon: %v", err)
				}
				if len(points) != 1 {
					t.Fatalf("SearchPolygon returned %d points, want 1", len(points))
				}
				points[0].Metadata["country"] = "mutated through SearchPolygon"
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := NewGeoIndex()
			caller := map[string]interface{}{}
			for k, v := range paris.Metadata {
				caller[k] = v
			}
			insert := paris
			insert.Metadata = caller
			if err := g.Insert(insert); err != nil {
				t.Fatalf("Insert: %v", err)
			}

			tc.mutate(t, g, caller)

			got, ok := g.GetPoint(id)
			if !ok {
				t.Fatalf("GetPoint(%q) found nothing", id)
			}
			if got.Metadata["country"] != "FR" {
				t.Errorf("the index now reports country=%v; the caller's write reached the stored point",
					got.Metadata["country"])
			}
		})
	}
}
