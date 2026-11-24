#!/usr/bin/env bash
# Serve the static app locally.  Usage: ./serve.sh [port] [bind-address]
#   ./serve.sh              -> http://127.0.0.1:8137/   (this machine only)
#   ./serve.sh 8137 0.0.0.0 -> reachable from other machines on your network (your call)
#   ./serve.sh 8137 tailscale -> bound to this machine's Tailscale address only (tailnet devices, not the LAN)
set -euo pipefail
cd "$(dirname "$0")/web"
PORT="${1:-8137}"; BIND="${2:-127.0.0.1}"
if [ "$BIND" = "tailscale" ]; then
  BIND="$(tailscale ip -4 2>/dev/null | head -n1)"
  [ -n "$BIND" ] || { echo "Tailscale is not up (tailscale ip -4 returned nothing)" >&2; exit 1; }
fi
echo "Lost in Space: http://${BIND}:${PORT}/   (Ctrl-C to stop)"
exec python3 -m http.server "$PORT" --bind "$BIND"
