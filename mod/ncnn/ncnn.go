// Package ncnn carries the self-contained ncnn inference runtime compiled to a
// WASI WebAssembly module (ncnn + stb_image + a tiny C driver, linked against
// wasi-libc). The .wasm is embedded into the Go binary so there are no external
// runtime files to ship. Rebuild it with build/build.sh.
package ncnn

import (
	"bytes"
	"context"
	_ "embed"
	"fmt"
	"io/fs"
	"time"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/imports/wasi_snapshot_preview1"
)

// Session owns the wazero runtime and the compiled ncnn wasm module. It is the
// shared execution engine: construct one with NewNcnnSession and reuse it across
// any number of classifiers/models via Run — each Run instantiates a fresh,
// isolated module instance, so calls don't share state. Run calls are serial;
// the Session is not intended for concurrent use.
type Session struct {
	runtime  wazero.Runtime
	compiled wazero.CompiledModule
}

// RunRequest describes one invocation of the wasm module.
type RunRequest struct {
	// Args is the argv handed to the module's main() (Args[0] is the program
	// name by convention).
	Args []string
	// Mounts maps a guest path (e.g. "/models") to an fs.FS exposed read-only
	// to the sandbox at that path. Use this to hand the guest its model files,
	// input image, etc. without touching the host disk.
	Mounts map[string]fs.FS
}

// RunResult is the captured outcome of a Run.
type RunResult struct {
	Stdout   string        // everything the guest wrote to stdout
	Stderr   string        // everything the guest wrote to stderr
	Duration time.Duration // wall-clock time of the wasm round-trip
}

//go:embed ncnn_classify.wasm
var ncnnClassifyWasm []byte

// GetNcnnClassifyWasm returns the embedded ncnn_classify.wasm bytes.
func GetNcnnClassifyWasm() []byte {
	return ncnnClassifyWasm
}

func NewNcnnSession() (*Session, error) {
	wasmBytes := GetNcnnClassifyWasm()
	if len(wasmBytes) == 0 {
		return nil, fmt.Errorf("embedded ncnn_classify.wasm is empty (run build/build.sh)")
	}

	// Pure-Go runtime — wazero compiles/interprets the wasm itself (no cgo).
	r := wazero.NewRuntime(context.Background())

	// The module imports wasi_snapshot_preview1 (fd_read, args_get, ...) because
	// it was linked against wasi-libc; register wazero's implementation.
	if _, err := wasi_snapshot_preview1.Instantiate(context.Background(), r); err != nil {
		_ = r.Close(context.Background())
		return nil, fmt.Errorf("instantiate wasi: %w", err)
	}

	compiled, err := r.CompileModule(context.Background(), wasmBytes)
	if err != nil {
		_ = r.Close(context.Background())
		return nil, fmt.Errorf("compile wasm: %w", err)
	}

	return &Session{runtime: r, compiled: compiled}, nil
}

// Run instantiates the compiled module once with the given args and filesystem
// mounts, capturing stdout/stderr. The returned RunResult is populated even when
// err is non-nil (e.g. a non-zero exit from main()), so callers can surface the
// guest's stderr for diagnostics.
func (s *Session) Run(ctx context.Context, req RunRequest) (*RunResult, error) {
	fsCfg := wazero.NewFSConfig()
	for guestPath, fsys := range req.Mounts {
		fsCfg = fsCfg.WithFSMount(fsys, guestPath)
	}

	var stdout, stderr bytes.Buffer
	cfg := wazero.NewModuleConfig().
		WithStdout(&stdout).
		WithStderr(&stderr).
		WithArgs(req.Args...).
		WithName(""). // anonymous so the same compiled module can be re-instantiated
		WithFSConfig(fsCfg)

	start := time.Now()
	mod, err := s.runtime.InstantiateModule(ctx, s.compiled, cfg)
	res := &RunResult{Stdout: stdout.String(), Stderr: stderr.String(), Duration: time.Since(start)}
	if err != nil {
		return res, err
	}
	_ = mod.Close(ctx)
	return res, nil
}

// Close releases the underlying wazero runtime.
func (s *Session) Close(ctx context.Context) error {
	return s.runtime.Close(ctx)
}
