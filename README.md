# FrontDesk Nexus AI

Voice-first hospital front desk assistant built for compliant patient intake, registration support, and department routing.

FrontDesk Nexus AI is designed for healthcare reception workflows where speed, clarity, and compliance matter. The assistant listens to a patient's request, transcribes it in real time, generates a short administrative response, and speaks back through a lip-synced 3D avatar. It is explicitly constrained to front desk tasks and does not provide medical advice.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-15+-black.svg)](https://nextjs.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What It Does

- Handles hospital front desk conversations through voice
- Routes patients to the appropriate department
- Helps with registration-style intake flows
- Refuses diagnosis, prescriptions, dosage guidance, and treatment advice
- Responds through a 3D avatar with synchronized lip movement

## Why This Project Exists

Hospital front desks are overloaded with repetitive but important conversations: where to go, what to do next, how to register, and who to speak to. These interactions are administrative, time-sensitive, and often happen under pressure.

FrontDesk Nexus AI targets that exact layer. It is not a doctor, not a symptom checker, and not a treatment engine. It is a compliant front desk assistant that reduces queue pressure while keeping a strict boundary around medical advice.

## Core Capabilities

- Real-time voice activity detection using an AudioWorklet-based microphone pipeline
- Speech-to-text using `openai/whisper-tiny`
- Administrative response generation using `minimaxai/minimax-m2.5` through NVIDIA's OpenAI-compatible API
- Text-to-speech using Kokoro with native word-level timing
- 3D avatar playback using TalkingHead with synchronized lip-sync
- WebSocket-based low-latency client-server communication
- Support for multimodal context on the backend through optional image input

## Compliance Guardrails

This project is built around healthcare-safe scope control.

- The assistant is instructed to act only as an administrative hospital front desk agent.
- User inputs are checked for medical-advice intent such as diagnosis, prescriptions, medication selection, cure suggestions, or dosage requests.
- Model outputs are checked again for disallowed medical guidance.
- Unsafe outputs are replaced with a compliant redirect response.
- Responses are intentionally kept short to reduce hallucinated overreach.

Safe fallback behavior:

> I cannot provide medical advice. I can register you with a doctor and route you to the correct department.

## Architecture

![System Architecture](./images/architecture.svg)

### End-to-End Flow

1. The user speaks into the mic from the web client.
2. The frontend detects valid speech segments with voice activity detection.
3. Audio is sent over WebSocket to the backend.
4. Whisper transcribes the audio.
5. The backend checks compliance and sends the request to the LLM.
6. Kokoro generates speech plus word timing metadata.
7. The frontend receives audio and timing data.
8. The avatar speaks the response with synchronized lip movement.

## Tech Stack

### Frontend

- Next.js 15
- React 19
- TypeScript
- Tailwind CSS
- TalkingHead for avatar rendering
- Web Audio API + AudioWorklet for VAD
- React Context for WebSocket state management

### Backend

- FastAPI
- Python 3.10
- PyTorch + CUDA 12.4
- Transformers
- OpenAI Python SDK for NVIDIA-hosted model access
- Kokoro TTS
- Pillow for image handling

### Workspace Tooling

- pnpm workspaces
- UV for Python dependency management
- Husky
- Black
- Prettier

## Repository Structure

```text
front-desk-nexus-ai/
├── apps/
│   ├── client/        # Next.js frontend
│   └── server/        # FastAPI backend
├── images/            # Architecture and static assets
├── SUBMISSION.md      # Detailed hackathon submission document
├── package.json       # Root workspace scripts
└── pnpm-workspace.yaml
```

## Quick Start

### Prerequisites

- Node.js 20+
- pnpm
- Python 3.10
- UV
- NVIDIA API key

### Install Dependencies

```bash
pnpm run monorepo-setup
```

### Environment

Backend environment variables:

```powershell
$env:NVIDIA_API_KEY="your_nvidia_api_key_here"
$env:NVIDIA_MODEL="minimaxai/minimax-m2.5"
$env:NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"
```

Optional client environment variable for mobile or remote testing:

```powershell
$env:NEXT_PUBLIC_WS_URL="wss://<your-ngrok-domain>/ws/test-client"
```

### Run the App

```bash
pnpm dev
```

Or run each side independently:

```bash
pnpm dev:client
pnpm dev:server
```

Default local endpoints:

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`

## Using the App

1. Open the client in the browser.
2. Allow microphone access.
3. Hold the mic button to speak, or switch to continuous listening.
4. Ask an administrative hospital question such as department routing or registration help.
5. Listen to the avatar's spoken response.

Example prompts:

- I need to register for a cardiology consultation.
- I have knee pain, which department should I go to?
- I want to book an appointment with a general physician.

## Current Product Shape

The current UI is intentionally minimal:

- Full-viewport 3D avatar
- Bottom-anchored voice controls
- Subtle online/offline status
- Speaking indicator during response playback

The backend still supports multimodal image input, but the main demo experience is centered on voice-first hospital front desk interaction.

## Hackathon Context

This project was built for a healthcare-focused AI agent use case where the key constraint is not only capability, but safe bounded behavior. The main innovation is combining a natural voice interface with explicit compliance guardrails for non-clinical hospital workflows.

For the full submission write-up, see [SUBMISSION.md](./SUBMISSION.md).

## Acknowledgments

- [TalkingHead](https://github.com/met4citizen/TalkingHead) for 3D avatar rendering and lip-sync
- [Kokoro](https://github.com/hexgrad/kokoro) for natural TTS generation
- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition

---

Built as FrontDesk Nexus AI, a compliant hospital front desk assistant.
