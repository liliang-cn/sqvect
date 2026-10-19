package main

import (
	"bufio"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// recallSituation is what a prompt is said against: the project the session
// works in and the start of the last reply. "这个修复前后可以验证吗" names
// nothing by itself; asked bare it recalls whatever other project once fixed
// and verified something. The cortexdb-live mod asks the same way when its
// planner does not answer (hooks/register.tsx, situated).
type recallSituation struct {
	Project  string
	Previous string
}

// recallContextRunes is how much of the last reply rides along with the
// prompt: enough to name the subject, not so much that it outweighs the prompt.
const recallContextRunes = 300

// transcriptTailBytes is how far back from the end of the transcript the last
// reply is looked for. A turn's reply is near the end; reading the whole file
// before every prompt would cost a long session seconds.
const transcriptTailBytes = 512 * 1024

func situationFor(in hookInput) recallSituation {
	var s recallSituation
	if cwd := filepath.Clean(strings.TrimSpace(in.Cwd)); in.Cwd != "" {
		// A session started in the home directory works in no project.
		if home, err := os.UserHomeDir(); err != nil || cwd != filepath.Clean(home) {
			s.Project = filepath.Base(cwd)
		}
	}
	if in.TranscriptPath != "" {
		s.Previous = lastAssistantText(in.TranscriptPath)
	}
	return s
}

// query is the prompt with the project and the start of the last reply after
// it. The server embeds the query alone, so the context has to be in it.
func (s recallSituation) query(prompt string) string {
	var lines []string
	if s.Project != "" {
		lines = append(lines, "Project: "+s.Project)
	}
	if s.Previous != "" {
		lines = append(lines, "Context: "+clip(strings.Join(strings.Fields(s.Previous), " "), recallContextRunes))
	}
	if len(lines) == 0 {
		return prompt
	}
	return prompt + "\n\n" + strings.Join(lines, "\n")
}

func (s recallSituation) keywords(prompt string) []string {
	kw := keywordsFromPrompt(prompt)
	if s.Project == "" {
		return kw
	}
	for _, k := range kw {
		if k == s.Project {
			return kw
		}
	}
	return append(kw, s.Project)
}

func (s recallSituation) entities() []string {
	if s.Project == "" {
		return nil
	}
	return []string{s.Project}
}

// lastAssistantText is the text of the last assistant entry in a Claude Code
// transcript, or "" when there is none or the file cannot be read.
func lastAssistantText(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	if info, err := f.Stat(); err == nil && info.Size() > transcriptTailBytes {
		if _, err := f.Seek(info.Size()-transcriptTailBytes, io.SeekStart); err != nil {
			return ""
		}
	}
	last := ""
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 8*1024*1024)
	for scanner.Scan() {
		var entry struct {
			Type    string `json:"type"`
			Message struct {
				Content json.RawMessage `json:"content"`
			} `json:"message"`
		}
		// The first line after a seek is cut mid-entry and fails here.
		if json.Unmarshal(scanner.Bytes(), &entry) != nil || entry.Type != "assistant" {
			continue
		}
		if text := strings.TrimSpace(textFromContent(entry.Message.Content)); text != "" {
			last = text
		}
	}
	return last
}
