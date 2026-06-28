// Package api is the HTTP layer that exposes the cnnaio model packages as an
// OpenAI-style REST service. It is a thin wrapper over the existing mod/* model
// packages running on a pool of shared ncnn.Sessions.
package api

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// DefaultConfigPath is where the server looks for its config by default.
const DefaultConfigPath = "conf/config.json"

// Config holds all server-tunable settings (see conf/config.json).
type Config struct {
	Listen        string   `json:"listen"`          // address to bind, e.g. ":8080"
	NoAuth        bool     `json:"no_auth"`         // disable Bearer-token auth entirely
	MaxImageBytes int64    `json:"max_image_bytes"` // reject larger uploads/payloads
	MaxResults    int      `json:"max_results"`     // cap on detection-like items returned
	TimeoutSec    int      `json:"request_timeout_seconds"`
	RateLimitRPM  int      `json:"rate_limit_per_minute"` // 0 = unlimited
	CORSOrigins   []string `json:"cors_origins"`          // e.g. ["*"] or specific origins

	DefaultModels struct {
		Classification string `json:"classification"`
		Detection      string `json:"detection"`
		FaceDetection  string `json:"face_detection"`
	} `json:"default_models"`
}

// DefaultConfig returns the built-in defaults used when no config file exists.
func DefaultConfig() Config {
	c := Config{
		Listen:        ":8080",
		NoAuth:        false,
		MaxImageBytes: 10 << 20, // 10 MiB
		MaxResults:    100,
		TimeoutSec:    30,
		RateLimitRPM:  0,
		CORSOrigins:   []string{"*"},
	}
	c.DefaultModels.Classification = "mobilenet-v2"
	c.DefaultModels.Detection = "yolo11n"
	c.DefaultModels.FaceDetection = "ultraface-rfb-320"
	return c
}

// LoadConfig reads config from path. If the file does not exist it writes a
// default config there and returns the defaults.
func LoadConfig(path string) (Config, error) {
	if path == "" {
		path = DefaultConfigPath
	}
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		cfg := DefaultConfig()
		if werr := saveConfig(path, cfg); werr != nil {
			return cfg, fmt.Errorf("write default config: %w", werr)
		}
		return cfg, nil
	}
	if err != nil {
		return Config{}, fmt.Errorf("read config %s: %w", path, err)
	}

	cfg := DefaultConfig() // start from defaults so missing keys are filled in
	if err := json.Unmarshal(b, &cfg); err != nil {
		return Config{}, fmt.Errorf("parse config %s: %w", path, err)
	}
	return cfg, nil
}

func saveConfig(path string, cfg Config) error {
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	b, _ := json.MarshalIndent(cfg, "", "  ")
	return os.WriteFile(path, b, 0o644)
}
