package api

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"sync"
)

// Job is an async inference task.
type Job struct {
	ID      string `json:"id"`
	Object  string `json:"object"` // "job"
	Status  string `json:"status"` // queued | running | succeeded | failed
	Created int64  `json:"created"`
	Result  any    `json:"result,omitempty"`
	Error   any    `json:"error,omitempty"`
}

// JobStore keeps async jobs in memory (process lifetime).
type JobStore struct {
	mu sync.RWMutex
	m  map[string]*Job
}

// NewJobStore creates an empty store.
func NewJobStore() *JobStore { return &JobStore{m: make(map[string]*Job)} }

// Submit registers a job and runs compute in the background. compute receives a
// fresh context (independent of the originating request).
func (js *JobStore) Submit(compute func(ctx context.Context) (any, error)) *Job {
	id := "job-" + randHex(12)
	job := &Job{ID: id, Object: "job", Status: "queued", Created: now()}
	js.mu.Lock()
	js.m[id] = job
	js.mu.Unlock()

	go func() {
		js.update(id, func(j *Job) { j.Status = "running" })
		res, err := compute(context.Background())
		js.update(id, func(j *Job) {
			if err != nil {
				j.Status = "failed"
				if ae, ok := err.(*apiError); ok {
					j.Error = ae
				} else {
					j.Error = errServer(err.Error())
				}
				return
			}
			j.Status = "succeeded"
			j.Result = res
		})
	}()
	return job
}

// Get returns a copy-safe pointer to a job by id.
func (js *JobStore) Get(id string) (*Job, bool) {
	js.mu.RLock()
	defer js.mu.RUnlock()
	j, ok := js.m[id]
	return j, ok
}

func (js *JobStore) update(id string, fn func(*Job)) {
	js.mu.Lock()
	defer js.mu.Unlock()
	if j, ok := js.m[id]; ok {
		fn(j)
	}
}

func randHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
