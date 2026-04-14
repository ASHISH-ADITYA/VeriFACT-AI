#!/bin/zsh
set -euo pipefail

APP_NAME="VeriFact Desktop"
PROJECT_ROOT="/Users/adityaashish/Desktop/ENGINEERING_PROJECT_II/verifactai"
LAUNCH_CMD="$PROJECT_ROOT/desktop/launch_verifact_desktop.command"
APP_PATH="/Applications/${APP_NAME}.app"
DESKTOP_LINK="$HOME/Desktop/${APP_NAME}.app"

if [[ ! -x "$LAUNCH_CMD" ]]; then
  chmod +x "$LAUNCH_CMD"
fi

# Build a native macOS app wrapper via AppleScript
osacompile -o "$APP_PATH" -e "do shell script \"$LAUNCH_CMD >/tmp/verifact_launch.out 2>/tmp/verifact_launch.err &\""

# Put shortcut on Desktop for easy launch
rm -rf "$DESKTOP_LINK"
ln -s "$APP_PATH" "$DESKTOP_LINK"

echo "Installed: $APP_PATH"
echo "Desktop shortcut: $DESKTOP_LINK"
echo "Double-click '${APP_NAME}' to launch."