package hfsync

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/tmc/go-iroh/iroh"
)

// Server handles incoming P2P cache sync requests over Iroh QUIC streams.
type Server struct {
	ep    *iroh.Endpoint
	store *Store
}

// NewServer creates an hfsync Server associated with the given endpoint and store.
func NewServer(ep *iroh.Endpoint, store *Store) *Server {
	return &Server{
		ep:    ep,
		store: store,
	}
}

// Serve starts accepting and handling incoming sync connections until ctx is cancelled.
func (s *Server) Serve(ctx context.Context) error {
	for {
		conn, err := s.ep.Accept(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return fmt.Errorf("accept conn: %w", err)
		}

		go s.HandleConnection(ctx, conn)
	}
}

// HandleConnection serves hfsync streams on an already accepted hfsync ALPN connection.
// It is useful with an [iroh.Router] that dispatches multiple protocols.
func (s *Server) HandleConnection(ctx context.Context, conn *iroh.Conn) error {
	if conn.ALPN() != ALPN {
		return fmt.Errorf("unexpected ALPN %q", conn.ALPN())
	}
	for {
		stream, err := conn.AcceptStream(ctx)
		if err != nil {
			return nil
		}

		go s.handleStream(ctx, stream)
	}
}

func (s *Server) handleStream(ctx context.Context, stream *iroh.Stream) {
	defer stream.Close()

	msgType, payload, err := readFrame(stream)
	if err != nil {
		return
	}

	switch msgType {
	case msgTypeListModelsReq:
		var req ListModelsRequest
		if err := json.Unmarshal(payload, &req); err != nil {
			return
		}
		models, err := s.store.ListModels()
		if err != nil {
			models = nil
		}
		if req.Filter != "" {
			filtered := models[:0]
			for _, model := range models {
				if strings.HasPrefix(model.ModelID, req.Filter) {
					filtered = append(filtered, model)
				}
			}
			models = filtered
		}
		_ = writeJSONFrame(stream, msgTypeListModelsResp, ListModelsResponse{Models: models})

	case msgTypeManifestReq:
		var req ManifestRequest
		if err := json.Unmarshal(payload, &req); err != nil {
			return
		}

		manifest, err := s.store.ScanModel(req.ModelID)
		if err != nil {
			_ = writeJSONFrame(stream, msgTypeManifestResp, Manifest{
				ModelID: req.ModelID,
				Files:   nil,
			})
			return
		}

		_ = writeJSONFrame(stream, msgTypeManifestResp, manifest)

	case msgTypeChunkReq:
		var req ChunkRequest
		if err := json.Unmarshal(payload, &req); err != nil {
			return
		}

		f, err := s.store.OpenFileReader(req.ModelID, req.Path)
		if err != nil {
			header := ChunkHeader{Status: fmt.Sprintf("not found: %v", err)}
			_ = writeJSONFrame(stream, msgTypeChunkResp, header)
			return
		}
		defer f.Close()

		if req.Offset < 0 {
			header := ChunkHeader{Status: "invalid negative offset"}
			_ = writeJSONFrame(stream, msgTypeChunkResp, header)
			return
		}
		if req.Length <= 0 {
			req.Length = DefaultChunkSize
		}
		if req.Length > DefaultChunkSize {
			header := ChunkHeader{Status: fmt.Sprintf("chunk length exceeds %d bytes", DefaultChunkSize)}
			_ = writeJSONFrame(stream, msgTypeChunkResp, header)
			return
		}
		if info, err := f.Stat(); err != nil || req.Offset >= info.Size() {
			header := ChunkHeader{Status: "requested range is outside file"}
			_ = writeJSONFrame(stream, msgTypeChunkResp, header)
			return
		}

		buf := make([]byte, req.Length)
		n, err := f.ReadAt(buf, req.Offset)
		if err != nil && err != io.EOF {
			header := ChunkHeader{Status: fmt.Sprintf("read error: %v", err)}
			_ = writeJSONFrame(stream, msgTypeChunkResp, header)
			return
		}

		header := ChunkHeader{
			Status: "ok",
			Offset: req.Offset,
			Length: int64(n),
		}

		headerBytes, _ := json.Marshal(header)
		if err := writeFrame(stream, msgTypeChunkResp, headerBytes); err != nil {
			return
		}

		// Write raw binary chunk
		if n > 0 {
			_, _ = stream.Write(buf[:n])
		}
	}
}
