package api

import (
	"embed"
	"io/fs"
	"net/http"
	"os"
)

// webAssets holds the developer web UI (API tester) so it ships inside the
// binary — no external ./web folder needs to be distributed alongside a
// release. Rebuild/edit the source under web/; go:embed picks it up at
// compile time.
//
//go:embed web
var webAssets embed.FS

// DevUIHandler wraps the API handler with a static file server for the developer
// web UI (the API tester). /v1/* goes to the API; everything else is served from
// webDir (so / -> webDir/index.html) if that directory exists on disk — handy
// while editing the UI, since changes are picked up without a rebuild. If webDir
// doesn't exist (e.g. a release binary run without the source tree around it),
// it falls back to the copy embedded in the binary. Only mounted when the server
// runs with -dev.
func DevUIHandler(apiHandler http.Handler, webDir string) http.Handler {
	mux := http.NewServeMux()
	mux.Handle("/v1/", apiHandler)
	mux.Handle("/", http.FileServer(http.FS(webUIFS(webDir))))
	return mux
}

// webUIFS resolves the filesystem backing the developer web UI: a real
// directory at webDir if present, otherwise the assets embedded via go:embed.
func webUIFS(webDir string) fs.FS {
	if info, err := os.Stat(webDir); err == nil && info.IsDir() {
		return os.DirFS(webDir)
	}
	sub, err := fs.Sub(webAssets, "web")
	if err != nil {
		// Unreachable: "web" is embedded at compile time via the directive above.
		return webAssets
	}
	return sub
}
