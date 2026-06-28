package api

import (
	"context"
	"net/http"
	"strings"
	"sync"
	"time"

	"cnnaio/mod/ncnn"
)

// Version is reported by /v1/health.
const Version = "0.1.0"

// Server is the HTTP service: a config, a session pool, the token set, and the
// async job store.
type Server struct {
	cfg    Config
	pool   *Pool
	tokens map[string]struct{}
	jobs   *JobStore
	start  time.Time
	limit  *rateLimiter
}

// NewServer builds the service. tokens is the set of accepted API tokens.
func NewServer(cfg Config, pool *Pool, tokens []string) *Server {
	set := make(map[string]struct{}, len(tokens))
	for _, t := range tokens {
		set[t] = struct{}{}
	}
	return &Server{
		cfg:    cfg,
		pool:   pool,
		tokens: set,
		jobs:   NewJobStore(),
		start:  time.Now(),
		limit:  newRateLimiter(cfg.RateLimitRPM),
	}
}

// apiHandler is a handler that returns an error (rendered via writeError).
type apiHandler func(w http.ResponseWriter, r *http.Request) error

func (s *Server) wrap(h apiHandler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if err := h(w, r); err != nil {
			writeError(w, err)
		}
	}
}

// Handler returns the fully-wired http.Handler (routes + middleware).
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("POST /v1/images/classifications", s.wrap(s.handleClassify))
	mux.HandleFunc("POST /v1/images/detections", s.wrap(s.handleDetect))
	mux.HandleFunc("POST /v1/images/segmentations", s.wrap(s.handleSegment))
	mux.HandleFunc("POST /v1/images/poses", s.wrap(s.handlePose))
	mux.HandleFunc("POST /v1/images/oriented", s.wrap(s.handleOriented))

	mux.HandleFunc("POST /v1/faces/detections", s.wrap(s.handleFaceDetect))
	mux.HandleFunc("POST /v1/faces/landmarks", s.wrap(s.handleLandmarks))
	mux.HandleFunc("POST /v1/faces/embeddings", s.wrap(s.handleEmbeddings))
	mux.HandleFunc("POST /v1/faces/comparisons", s.wrap(s.handleComparison))
	mux.HandleFunc("POST /v1/faces/gender", s.wrap(s.handleGender))

	mux.HandleFunc("POST /v1/vision/analyze", s.wrap(s.handleAnalyze))

	mux.HandleFunc("GET /v1/jobs/{id}", s.wrap(s.handleJob))
	mux.HandleFunc("GET /v1/models", s.wrap(s.handleModels))
	mux.HandleFunc("GET /v1/models/{id}", s.wrap(s.handleModel))
	mux.HandleFunc("GET /v1/health", s.wrap(s.handleHealth))

	return s.cors(s.auth(s.ratelimit(mux)))
}

// --- middleware ---

func (s *Server) cors(next http.Handler) http.Handler {
	allow := strings.Join(s.cfg.CORSOrigins, ", ")
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if len(s.cfg.CORSOrigins) > 0 {
			origin := r.Header.Get("Origin")
			if contains(s.cfg.CORSOrigins, "*") {
				w.Header().Set("Access-Control-Allow-Origin", "*")
			} else if origin != "" && contains(s.cfg.CORSOrigins, origin) {
				w.Header().Set("Access-Control-Allow-Origin", origin)
				w.Header().Set("Vary", "Origin")
			} else {
				w.Header().Set("Access-Control-Allow-Origin", allow)
			}
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type")
		}
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) auth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Health is always public; auth can be disabled wholesale via config.
		if s.cfg.NoAuth || r.URL.Path == "/v1/health" {
			next.ServeHTTP(w, r)
			return
		}
		tok := bearerToken(r)
		if tok == "" {
			writeError(w, errAuth("missing API key (Authorization: Bearer <token>)"))
			return
		}
		if _, ok := s.tokens[tok]; !ok {
			writeError(w, errAuth("invalid API key"))
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) ratelimit(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !s.limit.allow() {
			writeError(w, &apiError{http.StatusTooManyRequests, "rate limit exceeded", "rate_limit_error", "", ""})
			return
		}
		next.ServeHTTP(w, r)
	})
}

// withSession acquires a pooled session (respecting the request context /
// configured timeout), runs fn, and returns the session.
func (s *Server) withSession(ctx context.Context, fn func(*ncnn.Session) error) error {
	sess, err := s.pool.Acquire(ctx)
	if err != nil {
		return errServer("no inference session available: " + err.Error())
	}
	defer s.pool.Release(sess)
	return fn(sess)
}

// reqContext returns a context with the configured request timeout.
func (s *Server) reqContext(r *http.Request) (context.Context, context.CancelFunc) {
	if s.cfg.TimeoutSec <= 0 {
		return context.WithCancel(r.Context())
	}
	return context.WithTimeout(r.Context(), time.Duration(s.cfg.TimeoutSec)*time.Second)
}

func bearerToken(r *http.Request) string {
	h := r.Header.Get("Authorization")
	if strings.HasPrefix(h, "Bearer ") {
		return strings.TrimSpace(h[len("Bearer "):])
	}
	return ""
}

func contains(ss []string, v string) bool {
	for _, s := range ss {
		if s == v {
			return true
		}
	}
	return false
}

// rateLimiter is a minimal fixed-window global limiter (RPM); 0 = unlimited.
type rateLimiter struct {
	mu     sync.Mutex
	rpm    int
	window time.Time
	count  int
}

func newRateLimiter(rpm int) *rateLimiter { return &rateLimiter{rpm: rpm, window: time.Now()} }

func (l *rateLimiter) allow() bool {
	if l.rpm <= 0 {
		return true
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if time.Since(l.window) >= time.Minute {
		l.window = time.Now()
		l.count = 0
	}
	if l.count >= l.rpm {
		return false
	}
	l.count++
	return true
}
