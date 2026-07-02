#!/bin/bash
cat << "EOF"
                         _       
                        (_)      
   ___ _ __  _ __   __ _ _  ___  
  / __| '_ \| '_ \ / _` | |/ _ \ 
 | (__| | | | | | | (_| | | (_) |
  \___|_| |_|_| |_|\__,_|_|\___/ 
                                 
                                 
	----- cnnaio installer -----

EOF

REPO="tobychui/cnnaio"
APPNAME="cnnaio"

echo "cnnaio is a pure-Go, zero-dependency CNN inference server."
echo "No apt packages, no Python, no cgo libraries are required to run it."
echo ""

if [ "$USER" = root ]; then
  echo "You are root"
  sudo=""
else
  sudo="sudo "
fi

# --- figure out where to fetch/where to install ------------------------------
INSTALL_DIR="$(pwd)/${APPNAME}"

if [ -d "$INSTALL_DIR" ]; then
  echo "Warning: ${INSTALL_DIR} already exists."
  read -p "Continue and reuse this folder? (y/n) " reuse
  if [[ $reuse != "y" ]]; then
    echo "Installation aborted."
    exit 1
  fi
fi
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR" || exit 1

# --- detect OS / architecture --------------------------------------------------
case "$(uname -s)" in
  Linux)  goos="linux" ;;
  Darwin) goos="darwin" ;;
  *)      read -p "Could not detect OS. Enter GOOS (linux/darwin): " goos ;;
esac

case "$(uname -m)" in
  x86_64|amd64)  goarch="amd64" ;;
  aarch64|arm64) goarch="arm64" ;;
  *)             read -p "Could not detect architecture. Enter GOARCH (amd64/arm64): " goarch ;;
esac

echo "Detected platform: ${goos}/${goarch}"

# --- download + extract the release tarball -----------------------------------
ASSET="${APPNAME}-${goos}-${goarch}.tar.gz"
DOWNLOAD_URL="https://github.com/${REPO}/releases/latest/download/${ASSET}"

echo "Downloading ${APPNAME} from ${DOWNLOAD_URL} ..."
if command -v curl >/dev/null 2>&1; then
  curl -fL --progress-bar -o "${ASSET}" "${DOWNLOAD_URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${ASSET}" "${DOWNLOAD_URL}"
else
  echo "ERROR: need curl or wget on PATH to download ${APPNAME}." >&2
  exit 1
fi

if [ ! -s "${ASSET}" ]; then
  echo "ERROR: download failed or produced an empty file (${ASSET})." >&2
  echo "Check that a release exists for ${goos}/${goarch} at:" >&2
  echo "  https://github.com/${REPO}/releases/latest" >&2
  exit 1
fi

echo "Extracting ${ASSET} ..."
tar -xzf "${ASSET}"
rm -f "${ASSET}"

# The tarball contains one top-level "cnnaio-<os>-<arch>/" folder; flatten it
# into $INSTALL_DIR so the binary ends up at $INSTALL_DIR/cnnaio.
EXTRACTED_DIR="${APPNAME}-${goos}-${goarch}"
if [ -d "${EXTRACTED_DIR}" ]; then
  cp -r "${EXTRACTED_DIR}"/. .
  rm -rf "${EXTRACTED_DIR}"
fi

if [ ! -f "./${APPNAME}" ]; then
  echo "ERROR: expected binary './${APPNAME}' not found after extraction." >&2
  exit 1
fi
chmod +x "./${APPNAME}"

echo "${APPNAME} binary installed at ${INSTALL_DIR}/${APPNAME}"
echo ""

# --- ask the user about auth --------------------------------------------------
read -p "Enable API token authentication? (recommended for anything reachable over the network) [Y/n] " auth_answer
case "${auth_answer:0:1}" in
  n|N )
    no_auth=true
    echo "Authentication disabled (no_auth: true). Anyone who can reach the server can call the API."
    ;;
  * )
    no_auth=false
    echo "Authentication enabled. A token will be generated after setup."
    ;;
esac

# --- ask the user about concurrency -------------------------------------------
if command -v nproc >/dev/null 2>&1; then
  default_jobs=$(nproc)
elif [[ "$(uname -s)" == "Darwin" ]] && command -v sysctl >/dev/null 2>&1; then
  default_jobs=$(sysctl -n hw.ncpu)
else
  default_jobs=1
fi

read -p "How many concurrent inference sessions (-j)? [default: ${default_jobs}, your CPU count] " jobs_answer
jobs=${jobs_answer:-$default_jobs}
if ! [[ "$jobs" =~ ^[0-9]+$ ]] || [ "$jobs" -lt 1 ]; then
  echo "Invalid value, falling back to ${default_jobs}."
  jobs=$default_jobs
fi

# --- ask for listen port -------------------------------------------------------
read -p "Enter listen port (default: 8080): " port
port=${port:-8080}

# --- write conf/config.json ---------------------------------------------------
mkdir -p "${INSTALL_DIR}/conf"
cat > "${INSTALL_DIR}/conf/config.json" <<EOF
{
  "listen": ":${port}",
  "no_auth": ${no_auth},
  "max_image_bytes": 10485760,
  "max_results": 100,
  "request_timeout_seconds": 30,
  "rate_limit_per_minute": 0,
  "cors_origins": [
    "*"
  ],
  "default_models": {
    "classification": "mobilenet-v2",
    "detection": "yolo11n",
    "face_detection": "ultraface-rfb-320"
  }
}
EOF
echo "Wrote ${INSTALL_DIR}/conf/config.json"

# --- generate a token if auth is enabled ---------------------------------------
if [ "$no_auth" = false ]; then
  echo "Generating an API token ..."
  token="$("./${APPNAME}" -nt)"
  if [ -n "$token" ]; then
    echo ""
    echo "=================================================================="
    echo " Your cnnaio API token (save this now, it will not be shown again):"
    echo ""
    echo "   ${token}"
    echo ""
    echo " It is also stored in ${INSTALL_DIR}/token/tokens.json"
    echo "=================================================================="
    echo ""
  else
    echo "WARNING: token generation did not print anything; check manually with:"
    echo "  cd ${INSTALL_DIR} && ./${APPNAME} -nt"
  fi
fi

# --- create start.sh ------------------------------------------------------------
cat > "${INSTALL_DIR}/start.sh" <<EOF
#!/bin/bash
cd "${INSTALL_DIR}" || exit 1
exec ./${APPNAME} -j ${jobs} -dev
EOF
chmod +x "${INSTALL_DIR}/start.sh"
echo "Created ${INSTALL_DIR}/start.sh (runs with -j ${jobs})"
echo ""

# --- offer to install as a systemd service --------------------------------------
if [[ "$(uname -s)" == "Linux" ]]; then
  read -p "Do you want to install cnnaio as a systemd service? (y/n) " systemd_answer
  if [[ $systemd_answer == "y" || $systemd_answer == "Y" ]]; then
    ${sudo}touch /etc/systemd/system/cnnaio.service
    ${sudo}chmod 666 /etc/systemd/system/cnnaio.service
    cat > /etc/systemd/system/cnnaio.service <<EOF
[Unit]
Description=cnnaio CNN Inference Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${INSTALL_DIR}
ExecStart=/bin/bash ${INSTALL_DIR}/start.sh

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    ${sudo}chmod 644 /etc/systemd/system/cnnaio.service

    ${sudo}systemctl daemon-reload
    ${sudo}systemctl enable cnnaio.service
    ${sudo}systemctl start cnnaio.service

    echo "cnnaio systemd service installed and started."
    echo "Check status with: sudo systemctl status cnnaio"
    echo "Follow logs with:  sudo journalctl -u cnnaio -f"
    ip_address=$(hostname -I 2>/dev/null | awk '{print $1}')
    if [ -n "$ip_address" ]; then
      echo "The server should be reachable at http://${ip_address}:${port}/"
    fi
  else
    echo "Systemd service installation skipped. Start the server manually with:"
    echo "  ${INSTALL_DIR}/start.sh"
  fi
else
  echo "Non-Linux OS detected; skipping systemd setup."
  echo "Start the server manually with: ${INSTALL_DIR}/start.sh"
fi

echo ""
echo "=================================================================="
echo " cnnaio installation complete!"
echo ""
echo " Tip: you can change the server configuration at any time by editing"
echo "   ${INSTALL_DIR}/conf/config.json"
echo " then restarting the service (sudo systemctl restart cnnaio), or"
echo " re-running ${INSTALL_DIR}/start.sh if running it manually."
echo "=================================================================="
