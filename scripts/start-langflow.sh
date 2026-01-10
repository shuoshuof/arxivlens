#!/usr/bin/env bash
set -euo pipefail

export ALL_PROXY="socks5h://127.0.0.1:7897"
export NO_PROXY="localhost,127.0.0.1,::1"
export LANGFLOW_SKIP_AUTH_AUTO_LOGIN=true

exec langflow run
