#!/bin/bash
exec @RAG_BASE_PY@ "$(dirname "$0")/server.py"
