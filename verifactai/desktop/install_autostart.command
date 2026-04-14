#!/bin/zsh
set -e
PLIST_SRC="/Users/adityaashish/Desktop/ENGINEERING_PROJECT_II/verifactai/desktop/com.verifact.desktop.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.verifact.desktop.plist"
mkdir -p "$HOME/Library/LaunchAgents"
cp "$PLIST_SRC" "$PLIST_DST"
launchctl unload "$PLIST_DST" >/dev/null 2>&1 || true
launchctl load "$PLIST_DST"
echo "VeriFact Desktop autostart installed."
echo "Loaded agent: com.verifact.desktop"
