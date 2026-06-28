package api

import (
	"net/http"
	"time"
)

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) error {
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": registry})
	return nil
}

func (s *Server) handleModel(w http.ResponseWriter, r *http.Request) error {
	id := r.PathValue("id")
	m, ok := findModel(id)
	if !ok {
		return errNotFound("no such model: "+id, "model_not_found")
	}
	writeJSON(w, http.StatusOK, m)
	return nil
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) error {
	writeJSON(w, http.StatusOK, map[string]any{
		"status":        "ok",
		"version":       Version,
		"models_loaded": len(registry),
		"sessions":      s.pool.Size(),
		"uptime_s":      int64(time.Since(s.start).Seconds()),
	})
	return nil
}

func (s *Server) handleJob(w http.ResponseWriter, r *http.Request) error {
	id := r.PathValue("id")
	job, ok := s.jobs.Get(id)
	if !ok {
		return errNotFound("no such job: "+id, "job_not_found")
	}
	writeJSON(w, http.StatusOK, job)
	return nil
}
