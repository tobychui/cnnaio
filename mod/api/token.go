package api

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// TokenDir is the directory where API tokens are stored.
const TokenDir = "token"

const tokensFile = "tokens.json"

// GenerateToken creates a cryptographically-random API token like
// "cxn-<64 hex chars>".
func GenerateToken() (string, error) {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return "cxn-" + hex.EncodeToString(b), nil
}

// CreateAndStoreToken generates a new token, appends it to token/tokens.json
// (creating the dir/file as needed), and returns it.
func CreateAndStoreToken(dir string) (string, error) {
	if dir == "" {
		dir = TokenDir
	}
	tok, err := GenerateToken()
	if err != nil {
		return "", err
	}
	tokens, err := LoadTokens(dir)
	if err != nil {
		return "", err
	}
	tokens = append(tokens, tok)
	if err := saveTokens(dir, tokens); err != nil {
		return "", err
	}
	return tok, nil
}

// LoadTokens reads all stored tokens. A missing file is not an error (returns nil).
func LoadTokens(dir string) ([]string, error) {
	if dir == "" {
		dir = TokenDir
	}
	b, err := os.ReadFile(filepath.Join(dir, tokensFile))
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read tokens: %w", err)
	}
	var tokens []string
	if err := json.Unmarshal(b, &tokens); err != nil {
		return nil, fmt.Errorf("parse tokens: %w", err)
	}
	return tokens, nil
}

func saveTokens(dir string, tokens []string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	b, _ := json.MarshalIndent(tokens, "", "  ")
	return os.WriteFile(filepath.Join(dir, tokensFile), b, 0o600)
}
