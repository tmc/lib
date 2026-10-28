package kernelagent_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/tmc/lib/kernelsearch/kernelagent"
)

type exampleOpenAITransport struct{}

func (exampleOpenAITransport) RoundTrip(req *http.Request) (*http.Response, error) {
	defer req.Body.Close()
	// Inert protocol fixture: this example never opens a socket.
	return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"status":"completed","output":[{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"{\"rules\":[]}"}]}]}`))}, nil
}

func ExampleOpenAIConfig() {
	c := kernelagent.OpenAIConfig{APIKey: "explicit-credential", Model: kernelagent.ModelIdentity{Kind: "model", Name: "configured-model", Version: "pinned-revision"}, Timeout: time.Second, MaxResponseBytes: 64 << 10, MaxOutputTokens: 1024, Transport: exampleOpenAITransport{}}
	_, err := kernelagent.NewOpenAI(c)
	fmt.Println(err)
	// Output: <nil>
}

func ExampleNewOpenAI() {
	p, err := kernelagent.NewOpenAI(kernelagent.OpenAIConfig{APIKey: "explicit-credential", Model: kernelagent.ModelIdentity{Kind: "model", Name: "configured-model", Version: "pinned-revision"}, Timeout: time.Second, MaxResponseBytes: 64 << 10, MaxOutputTokens: 1024, Transport: exampleOpenAITransport{}})
	if err != nil {
		fmt.Println(err)
		return
	}
	var proposer kernelagent.Proposer = p
	reply, err := proposer.Propose(context.Background(), kernelagent.Request{Epoch: "1", Operation: "add", Prompt: "select allowed rules"})
	fmt.Println(reply.Model.Name, string(reply.Edit), err)
	// Output: configured-model {"rules":[]} <nil>
}

func ExampleOpenAI_Propose() {
	ExampleNewOpenAI()
	// Output: configured-model {"rules":[]} <nil>
}

func ExampleOpenAI() {
	ExampleNewOpenAI()
	// Output: configured-model {"rules":[]} <nil>
}
