package api

import (
	"context"
	"fmt"

	"cnnaio/mod/ncnn"
)

// Pool is a fixed set of ncnn.Sessions. Each Session owns its own wazero runtime
// + compiled wasm and is used serially, so the pool size (-j) is the server's
// inference concurrency. Acquire blocks until a session is free.
type Pool struct {
	free chan *ncnn.Session
	all  []*ncnn.Session
}

// NewPool builds n sessions (n >= 1). Each compiles the embedded wasm once.
func NewPool(n int) (*Pool, error) {
	if n < 1 {
		n = 1
	}
	p := &Pool{free: make(chan *ncnn.Session, n)}
	for i := 0; i < n; i++ {
		s, err := ncnn.NewNcnnSession()
		if err != nil {
			p.Close(context.Background())
			return nil, fmt.Errorf("create session %d/%d: %w", i+1, n, err)
		}
		p.all = append(p.all, s)
		p.free <- s
	}
	return p, nil
}

// Size returns the number of sessions in the pool.
func (p *Pool) Size() int { return len(p.all) }

// Acquire blocks until a session is available or ctx is done.
func (p *Pool) Acquire(ctx context.Context) (*ncnn.Session, error) {
	select {
	case s := <-p.free:
		return s, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// Release returns a session to the pool.
func (p *Pool) Release(s *ncnn.Session) {
	if s != nil {
		p.free <- s
	}
}

// Close releases every underlying session's runtime.
func (p *Pool) Close(ctx context.Context) {
	for _, s := range p.all {
		_ = s.Close(ctx)
	}
}
