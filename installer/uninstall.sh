#!/bin/bash
cat << "EOF"
                         _       
                        (_)      
   ___ _ __  _ __   __ _ _  ___  
  / __| '_ \| '_ \ / _` | |/ _ \ 
 | (__| | | | | | | (_| | | (_) |
  \___|_| |_|_| |_|\__,_|_|\___/ 
                                 
                                 
	----- cnnaio uninstaller -----

EOF

APPNAME="cnnaio"

if [ "$USER" = root ]; then
  echo "You are root"
  sudo=""
else
  sudo="sudo "
fi

# --- find the install directory -------------------------------------------------
DEFAULT_DIR="$(pwd)/${APPNAME}"
if [ -d "$DEFAULT_DIR" ]; then
  INSTALL_DIR="$DEFAULT_DIR"
else
  read -p "Could not find ${DEFAULT_DIR}. Enter the cnnaio install path to remove: " INSTALL_DIR
fi

echo "This will remove:"
[ -d "$INSTALL_DIR" ] && echo "  - ${INSTALL_DIR}  (binary, conf/config.json, token/tokens.json, all data)"
if [[ $(uname -s) == "Linux" ]] && [ -f "/etc/systemd/system/${APPNAME}.service" ]; then
  echo "  - /etc/systemd/system/${APPNAME}.service"
fi
echo ""
read -p "Are you sure you want to uninstall cnnaio? (y/n) " choice

case "$choice" in
  y|Y )
    ok=true

    # Stop and disable the systemd service if it exists
    if [[ $(uname -s) == "Linux" ]]; then
      if [ -f "/etc/systemd/system/${APPNAME}.service" ]; then
        ${sudo}systemctl stop ${APPNAME} 2>/dev/null
        ${sudo}systemctl disable ${APPNAME} 2>/dev/null
        if ${sudo}rm -f "/etc/systemd/system/${APPNAME}.service"; then
          ${sudo}systemctl daemon-reload
          echo "Removed ${APPNAME} systemd service."
        else
          echo "WARNING: failed to remove /etc/systemd/system/${APPNAME}.service; remove it manually." >&2
          ok=false
        fi
      fi
    fi

    # Remove the install directory (binary, config, tokens, everything)
    if [ -d "$INSTALL_DIR" ]; then
      if ${sudo}rm -rf "$INSTALL_DIR"; then
        echo "Removed ${INSTALL_DIR}"
      else
        echo "WARNING: failed to remove ${INSTALL_DIR}; remove it manually." >&2
        ok=false
      fi
    else
      echo "Nothing found at ${INSTALL_DIR}; skipped."
    fi

    if [ "$ok" = true ]; then
      echo "cnnaio has been uninstalled successfully!"
    else
      echo "cnnaio uninstall finished with warnings above; some files may need manual removal." >&2
    fi
    ;;
  n|N )
    echo "Uninstall cancelled"
    ;;
  * )
    echo "Invalid input, uninstall cancelled"
    ;;
esac
