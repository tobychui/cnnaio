package api

import "net/http"

// DevUIHandler wraps the API handler with a static file server for the developer
// web UI (the API tester). /v1/* goes to the API; everything else is served from
// webDir (so / -> webDir/index.html). Only mounted when the server runs with -dev.
func DevUIHandler(apiHandler http.Handler, webDir string) http.Handler {
	mux := http.NewServeMux()
	mux.Handle("/v1/", apiHandler)
	mux.Handle("/", http.FileServer(http.Dir(webDir)))
	return mux
}
