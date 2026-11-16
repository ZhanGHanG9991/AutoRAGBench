#!/bin/bash
# Absolute interpreter path: at evaluation PATH is only /usr/local/bin:/usr/bin:/bin.
exec @RAG_BASE_PY@ "$(dirname "$0")/server.py"
