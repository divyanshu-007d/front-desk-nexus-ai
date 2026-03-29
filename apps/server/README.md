# FrontDesk Nexus AI Server

FastAPI backend that handles the complete pipeline:

1. Whisper STT (`openai/whisper-tiny`)
2. NVIDIA-hosted chat routing with compliance guardrail (`openai` SDK + NVIDIA endpoint)
3. Kokoro TTS with native word timings for lip-sync
4. WebSocket streaming back to the Next.js client

## Environment Variables

- `NVIDIA_API_KEY` (required)
- `NVIDIA_MODEL` (optional, default: `minimaxai/minimax-m2.5`)
- `NVIDIA_BASE_URL` (optional, default: `https://integrate.api.nvidia.com/v1`)

Create `apps/server/.env` from `.env.example` and fill your key.

PowerShell example:

```powershell
$env:NVIDIA_API_KEY="your_key_here"
$env:NVIDIA_MODEL="minimaxai/minimax-m2.5"
$env:NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"
```

## Run

From this folder:

```powershell
uv sync
uv run python -m uvicorn main:app --reload --env-file .env --host 0.0.0.0 --port 8000
```

## Mobile/Ngrok Notes

Expose port `8000` using ngrok and use the resulting WebSocket URL in the client via `NEXT_PUBLIC_WS_URL`, for example:

`wss://<your-ngrok-domain>/ws/test-client`
