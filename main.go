// Command server is the cnnaio image-recognition HTTP server. It exposes the
// model packages as an OpenAI-style REST API over a pool of shared ncnn sessions.
//
// Usage:
//
//	server                 # start the server (config from conf/config.json)
//	server -nt             # generate a new API token, store in ./token/, print it, exit
//	server -j 4            # run 4 inference sessions (concurrency)
//	server -addr :9000     # override the listen address
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"

	"cnnaio/mod/api"
)

func main() {
	newToken := flag.Bool("nt", false, "generate a new API token, store it in ./token/, print it to STDOUT, and exit")
	jobs := flag.Int("j", 1, "number of ncnn inference instances (session pool size / concurrency)")
	configPath := flag.String("config", api.DefaultConfigPath, "path to config.json")
	addr := flag.String("addr", "", "listen address override (else taken from config)")
	dev := flag.Bool("dev", false, "serve the developer web UI (API tester) from ./web")
	webDir := flag.String("webdir", "web", "directory served as the web UI in -dev mode")
	flag.Parse()

	// -nt: token generation command. Generate, store, print, exit.
	if *newToken {
		tok, err := api.CreateAndStoreToken(api.TokenDir)
		if err != nil {
			fmt.Fprintln(os.Stderr, "token generation failed:", err)
			os.Exit(1)
		}
		fmt.Println(tok)
		return
	}

	cfg, err := api.LoadConfig(*configPath)
	if err != nil {
		log.Fatalf("config: %v", err)
	}
	if *addr != "" {
		cfg.Listen = *addr
	}

	tokens, err := api.LoadTokens(api.TokenDir)
	if err != nil {
		log.Fatalf("tokens: %v", err)
	}
	if !cfg.NoAuth && len(tokens) == 0 {
		log.Printf("WARNING: auth is enabled but no tokens exist — generate one with `-nt`, or set \"no_auth\": true in %s", *configPath)
	}

	log.Printf("cnnaio: initialising %d inference session(s)…", *jobs)
	pool, err := api.NewPool(*jobs)
	if err != nil {
		log.Fatalf("inference pool: %v", err)
	}
	defer pool.Close(context.Background())

	srv := api.NewServer(cfg, pool, tokens)

	var handler http.Handler = srv.Handler()
	if *dev {
		handler = api.DevUIHandler(srv.Handler(), *webDir)
		log.Printf("dev UI enabled: http://localhost%s/  (serving %q)", cfg.Listen, *webDir)
	}
	httpSrv := &http.Server{Addr: cfg.Listen, Handler: handler}

	log.Printf("cnnaio listening on %s  (auth=%v, sessions=%d)", cfg.Listen, !cfg.NoAuth, pool.Size())
	if err := httpSrv.ListenAndServe(); err != nil {
		log.Fatalf("server: %v", err)
	}
}
