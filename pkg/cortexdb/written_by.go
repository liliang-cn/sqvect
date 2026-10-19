package cortexdb

import "context"

// The metadata keys a save records its writer under. A recall shows them, so a
// memory that answers wrongly can be traced to whoever wrote it.
const (
	metadataWrittenBy = "written_by"
	metadataUpdatedBy = "updated_by"
)

type writerContextKey struct{}

// WithWriter names who is writing through ctx: a server puts the identity it
// authenticated there, and SaveMemory, UpdateMemory, SaveKnowledge and
// UpdateKnowledge record it on the row. Whatever a caller wrote under those
// keys itself is replaced, so a client cannot write as someone else. Without a
// writer in ctx (an embedded library, a server without keys) the metadata is
// stored as given.
func WithWriter(ctx context.Context, name string) context.Context {
	if name == "" {
		return ctx
	}
	return context.WithValue(ctx, writerContextKey{}, name)
}

func writerOf(ctx context.Context) string {
	name, _ := ctx.Value(writerContextKey{}).(string)
	return name
}

// stampWriter records the writer of a save: written_by is who wrote the row
// now, and updated_by, which only an update sets, goes.
func stampWriter[V any](ctx context.Context, metadata map[string]V, as func(string) V) {
	name := writerOf(ctx)
	if name == "" {
		return
	}
	metadata[metadataWrittenBy] = as(name)
	delete(metadata, metadataUpdatedBy)
}

// stampUpdater records the writer of an update: the row keeps the written_by
// it was saved with, whatever the update sent, and updated_by names who
// changed it.
func stampUpdater[V any](ctx context.Context, metadata map[string]V, before map[string]any, as func(string) V) {
	name := writerOf(ctx)
	if name == "" {
		return
	}
	if original, ok := before[metadataWrittenBy].(string); ok && original != "" {
		metadata[metadataWrittenBy] = as(original)
	} else {
		delete(metadata, metadataWrittenBy)
	}
	metadata[metadataUpdatedBy] = as(name)
}

func anyString(s string) any { return s }

func plainString(s string) string { return s }
