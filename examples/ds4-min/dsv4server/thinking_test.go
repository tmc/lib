package dsv4server

import "testing"

func TestSplitThinkingOutput(t *testing.T) {
	tests := []struct {
		name      string
		text      string
		reasoning string
		content   string
		present   bool
		wantErr   bool
	}{
		{name: "plain", text: "answer", content: "answer"},
		{name: "split", text: "<think>work</think>answer", reasoning: "work", content: "answer", present: true},
		{name: "empty", text: "<think></think>answer", content: "answer", present: true},
		{name: "unterminated", text: "<think>work", wantErr: true},
		{name: "closing-only", text: "work</think>", wantErr: true},
		{name: "embedded", text: "answer <think>work</think>", wantErr: true},
		{name: "nested", text: "<think>one<think>two</think>answer", wantErr: true},
		{name: "trailing-marker", text: "<think>one</think>answer</think>", wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := splitThinkingOutput(test.text)
			if test.wantErr {
				if err == nil {
					t.Fatal("splitThinkingOutput succeeded")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got.reasoning != test.reasoning || got.content != test.content || got.present != test.present {
				t.Fatalf("got %+v, want reasoning=%q content=%q present=%v", got, test.reasoning, test.content, test.present)
			}
		})
	}
}

func TestThinkingStream(t *testing.T) {
	tests := []struct {
		name    string
		parts   []string
		reason  string
		content string
		wantErr bool
	}{
		{name: "split markers", parts: []string{"<th", "ink>wo", "rk</", "think>an", "swer"}, reason: "work", content: "answer"},
		{name: "plain", parts: []string{"answer"}, content: "answer"},
		{name: "unterminated", parts: []string{"<think>", "work"}, wantErr: true},
		{name: "late marker", parts: []string{"answer", "<think>work</think>"}, wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var stream thinkingStream
			var reasoning, content string
			for _, part := range test.parts {
				chunks, err := stream.feed(part)
				if err != nil {
					if !test.wantErr {
						t.Fatal(err)
					}
					return
				}
				for _, chunk := range chunks {
					reasoning += chunk.reasoning
					content += chunk.content
				}
			}
			chunks, err := stream.done()
			if test.wantErr {
				if err == nil {
					t.Fatal("stream succeeded")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			for _, chunk := range chunks {
				reasoning += chunk.reasoning
				content += chunk.content
			}
			if reasoning != test.reason || content != test.content {
				t.Fatalf("got reasoning=%q content=%q", reasoning, content)
			}
		})
	}
}
