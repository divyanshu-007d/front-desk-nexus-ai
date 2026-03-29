import asyncio
import json
import base64
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from openai import OpenAI
import numpy as np
import logging
import sys
import io
from PIL import Image
import time
import os
from datetime import datetime
from pathlib import Path
import re
from typing import Optional, Dict, Any
import traceback
import uvicorn

# FastAPI imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import Kokoro TTS library
from kokoro import KPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class ImageManager:
    """Manages image saving and verification"""

    def __init__(self, save_directory="received_images"):
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(exist_ok=True)
        logger.info(f"Image save directory: {self.save_directory.absolute()}")

    def save_image(self, image_data: bytes, client_id: str, prefix: str = "img") -> str:
        """Save image data and return the filename"""
        try:
            # Create timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
            filename = f"{prefix}_{client_id}_{timestamp}.jpg"
            filepath = self.save_directory / filename

            # Save the image
            with open(filepath, "wb") as f:
                f.write(image_data)

            # Log file info
            file_size = len(image_data)
            logger.info(f"💾 Saved image: {filename} ({file_size:,} bytes)")

            return str(filepath)

        except Exception as e:
            logger.error(f"❌ Error saving image: {e}")
            return None

    def verify_image(self, filepath: str) -> dict:
        """Verify saved image and return info"""
        try:
            if not os.path.exists(filepath):
                return {"error": "File not found"}

            # Get file stats
            stat = os.stat(filepath)
            file_size = stat.st_size

            # Try to open with PIL to verify it's a valid image
            with Image.open(filepath) as img:
                info = {
                    "filepath": filepath,
                    "file_size": file_size,
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "width": img.width,
                    "height": img.height,
                    "valid": True,
                }

            logger.info(f"✅ Image verified: {info}")
            return info

        except Exception as e:
            logger.error(f"❌ Error verifying image {filepath}: {e}")
            return {"error": str(e), "valid": False}


class WhisperProcessor:
    """Handles speech-to-text using Whisper model"""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        logger.info(f"Using device for Whisper: {self.device}")

        # Load Whisper model
        model_id = "openai/whisper-tiny"
        logger.info(f"Loading {model_id}...")

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        # Create pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

        logger.info("Whisper model ready for transcription")
        self.transcription_count = 0

    async def transcribe_audio(self, audio_bytes):
        """Transcribe audio bytes to text"""
        try:
            # Convert audio bytes to numpy array
            audio_array = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )

            # Run transcription in executor to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.pipe(audio_array)
            )

            transcribed_text = result["text"].strip()
            self.transcription_count += 1

            logger.info(
                f"Transcription #{self.transcription_count}: '{transcribed_text}'"
            )

            # Check for noise/empty transcription
            if not transcribed_text or len(transcribed_text) < 3:
                return "NO_SPEECH"

            # Check for common noise indicators
            noise_indicators = ["thank you", "thanks for watching", "you", ".", ""]
            if transcribed_text.lower().strip() in noise_indicators:
                return "NOISE_DETECTED"

            return transcribed_text

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None


class NvidiaProcessor:
    """Handles compliant response generation using NVIDIA's OpenAI-compatible API."""

    _instance = None
    SYSTEM_INSTRUCTION = (
        "You are FrontDesk Nexus AI, an administrative assistant. "
        "You NEVER give medical advice. "
        "If asked for medical advice, decline and offer to register them for a doctor. "
        "Keep responses under 2 sentences."
    )
    MEDICAL_ADVICE_REQUEST_PATTERN = re.compile(
        r"\b(diagnose|diagnosis|prescribe|prescription|what should i take|which medicine|medicine for|treatment for|cure|dose|dosage)\b",
        re.IGNORECASE,
    )
    MEDICAL_ADVICE_CONTENT_PATTERN = re.compile(
        r"\b(diagnosis|prescribe|prescription|dosage|dose|antibiotic|painkiller|medication|medicine)\b",
        re.IGNORECASE,
    )

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError("Missing NVIDIA_API_KEY environment variable")

        self.base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.model_name = os.getenv("NVIDIA_MODEL", "minimaxai/minimax-m2.5")
        self.client = OpenAI(base_url=self.base_url, api_key=api_key)

        self.lock = asyncio.Lock()
        self.message_history = []
        self.max_history_messages = 6
        self.last_image_bytes: Optional[bytes] = None
        self.last_image_timestamp = 0
        self.generation_count = 0
        logger.info(
            f"NVIDIA processor initialized with model: {self.model_name} ({self.base_url})"
        )

    async def set_image(self, image_data: bytes):
        """Cache the most recent image received for multimodal context."""
        async with self.lock:
            try:
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                new_size = (int(image.size[0] * 0.75), int(image.size[1] * 0.75))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

                output = io.BytesIO()
                image.save(output, format="JPEG", quality=85)
                self.last_image_bytes = output.getvalue()
                self.last_image_timestamp = time.time()
                logger.info("Image cached successfully for NVIDIA multimodal context")
                return True
            except Exception as e:
                logger.error(f"Error processing image for NVIDIA: {e}")
                return False

    def _build_history_context(self) -> str:
        if not self.message_history:
            return ""

        history_lines = []
        for message in self.message_history[-self.max_history_messages :]:
            role = message.get("role", "user")
            text = message.get("text", "")
            history_lines.append(f"{role}: {text}")

        return "Recent conversation context:\n" + "\n".join(history_lines)

    def _add_to_history(self, user_text: str, response_text: str):
        self.message_history.append({"role": "user", "text": user_text})
        self.message_history.append({"role": "assistant", "text": response_text})
        if len(self.message_history) > self.max_history_messages:
            self.message_history = self.message_history[-self.max_history_messages :]

    def _extract_response_text(self, response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""

        message = getattr(choices[0], "message", None)
        if not message:
            return ""

        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            extracted_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                    text = item.get("text")
                    if text:
                        extracted_parts.append(text)
            return " ".join(extracted_parts).strip()

        return ""

    def _limit_to_two_sentences(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return ""

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
        if len(sentences) >= 2:
            return " ".join(sentences[:2]).strip()

        if len(sentences) == 1:
            words = sentences[0].split()
            if len(words) <= 32:
                return sentences[0]
            return " ".join(words[:32]).rstrip(" ,;:") + "."

        words = cleaned.split()
        if len(words) <= 32:
            return cleaned
        return " ".join(words[:32]).rstrip(" ,;:") + "."

    def _decline_medical_advice(self) -> str:
        return (
            "I cannot provide medical advice. "
            "I can register you with a doctor and route you to the correct department."
        )

    def _is_medical_advice_request(self, text: str) -> bool:
        return bool(self.MEDICAL_ADVICE_REQUEST_PATTERN.search(text or ""))

    def _contains_medical_advice(self, text: str) -> bool:
        return bool(self.MEDICAL_ADVICE_CONTENT_PATTERN.search(text or ""))

    def _build_prompt_text(self, user_text: str) -> str:
        history_context = self._build_history_context()
        prompt_text = (
            "Front desk task: route the patient to the correct department and help with registration.\n"
            "Never provide diagnosis, treatment, medication, or dosage guidance.\n"
            "If user asks for medical advice, politely decline and offer doctor registration.\n"
            "Reply in plain language and maximum 2 sentences.\n"
        )
        if history_context:
            prompt_text += f"{history_context}\n"
        prompt_text += f"Patient input: {user_text}"
        return prompt_text

    async def _create_completion(self, messages: list[dict[str, Any]]) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2,
                top_p=0.95,
                max_tokens=220,
                stream=False,
            ),
        )

    async def generate_response(self, user_text: str, include_image: bool = False) -> str:
        """Generate a compliant administrative response using NVIDIA model routing."""
        sanitized_user_text = (user_text or "").strip()
        if not sanitized_user_text:
            return self._decline_medical_advice()

        if self._is_medical_advice_request(sanitized_user_text):
            response = self._decline_medical_advice()
            self._add_to_history(sanitized_user_text, response)
            return response

        async with self.lock:
            try:
                prompt_text = self._build_prompt_text(sanitized_user_text)
                user_content: Any = prompt_text

                if include_image and self.last_image_bytes:
                    encoded_image = base64.b64encode(self.last_image_bytes).decode("utf-8")
                    user_content = [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                        },
                    ]

                messages = [
                    {"role": "system", "content": self.SYSTEM_INSTRUCTION},
                    {"role": "user", "content": user_content},
                ]
                response = await self._create_completion(messages)
                response_text = self._extract_response_text(response)

                # Retry with text-only content in case selected model does not support image input.
                if not response_text and include_image and self.last_image_bytes:
                    logger.warning(
                        "NVIDIA multimodal completion returned no text; retrying text-only"
                    )
                    text_only_messages = [
                        {"role": "system", "content": self.SYSTEM_INSTRUCTION},
                        {"role": "user", "content": prompt_text},
                    ]
                    response = await self._create_completion(text_only_messages)
                    response_text = self._extract_response_text(response)

                if not response_text:
                    response_text = (
                        "I can help register you and route your case to the right department. "
                        "Please share your symptoms and I will connect you to a doctor."
                    )

                response_text = self._limit_to_two_sentences(response_text)

                if self._contains_medical_advice(response_text):
                    logger.warning(
                        "NVIDIA output flagged by compliance guardrail; replacing response"
                    )
                    response_text = self._decline_medical_advice()

                self._add_to_history(sanitized_user_text, response_text)
                self.generation_count += 1
                logger.info(
                    f"NVIDIA response #{self.generation_count}: '{response_text}'"
                )
                return response_text

            except Exception as e:
                logger.error(f"NVIDIA generation error: {e}")
                return (
                    "I am unable to process that right now. "
                    "I can still register you to see a doctor."
                )


class KokoroTTSProcessor:
    """Handles text-to-speech conversion using Kokoro model"""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        logger.info("Initializing Kokoro TTS processor...")
        try:
            # Initialize Kokoro TTS pipeline
            self.pipeline = KPipeline(lang_code="a")

            # Set voice
            self.default_voice = "af_sarah"

            logger.info("Kokoro TTS processor initialized successfully")
            # Counter
            self.synthesis_count = 0
        except Exception as e:
            logger.error(f"Error initializing Kokoro TTS: {e}")
            self.pipeline = None

    async def synthesize_initial_speech_with_timing(self, text):
        """Convert initial text to speech using Kokoro TTS data"""
        if not text or not self.pipeline:
            return None, []

        try:
            logger.info(f"Synthesizing initial speech for text: '{text}'")

            # Run TTS in a thread pool to avoid blocking
            audio_segments = []
            all_word_timings = []
            time_offset = 0  # Track cumulative time for multiple segments

            # Use the executor to run the TTS pipeline with minimal splitting
            generator = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipeline(
                    text,
                    voice=self.default_voice,
                    speed=1,
                    split_pattern=None,  # No splitting for initial text to process faster
                ),
            )

            # Process all generated segments and extract NATIVE timing
            for i, result in enumerate(generator):
                # Extract the components as shown in your screenshot
                gs = result.graphemes  # str - the text graphemes
                ps = result.phonemes  # str - the phonemes
                audio = result.audio.cpu().numpy()  # numpy array
                tokens = result.tokens  # List[en.MToken] - THE TIMING GOLD!

                logger.info(
                    f"Segment {i}: {len(tokens)} tokens, audio shape: {audio.shape}"
                )

                # Extract word timing from native tokens with null checks
                for token in tokens:
                    # Check if timing data is available
                    if token.start_ts is not None and token.end_ts is not None:
                        word_timing = {
                            "word": token.text,
                            "start_time": (token.start_ts + time_offset)
                            * 1000,  # Convert to milliseconds
                            "end_time": (token.end_ts + time_offset)
                            * 1000,  # Convert to milliseconds
                        }
                        all_word_timings.append(word_timing)
                        logger.debug(
                            f"Word: '{token.text}' Start: {word_timing['start_time']:.1f}ms End: {word_timing['end_time']:.1f}ms"
                        )
                    else:
                        # Log when timing data is missing
                        logger.debug(
                            f"Word: '{token.text}' - No timing data available (start_ts: {token.start_ts}, end_ts: {token.end_ts})"
                        )

                # Add audio segment
                audio_segments.append(audio)

                # Update time offset for next segment
                if len(audio) > 0:
                    segment_duration = len(audio) / 24000  # seconds
                    time_offset += segment_duration

            # Combine all audio segments
            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                self.synthesis_count += 1
                logger.info(
                    f"✨ Initial speech synthesis complete: {len(combined_audio)} samples, {len(all_word_timings)} word timings"
                )
                return combined_audio, all_word_timings
            return None, []

        except Exception as e:
            logger.error(f"Initial speech synthesis with timing error: {e}")
            return None, []

    async def synthesize_remaining_speech_with_timing(self, text):
        """Convert remaining text to speech using Kokoro TTS data"""
        if not text or not self.pipeline:
            return None, []

        try:
            logger.info(
                f"Synthesizing chunk speech for text: '{text[:50]}...' if len(text) > 50 else text"
            )

            # Run TTS in a thread pool to avoid blocking
            audio_segments = []
            all_word_timings = []
            time_offset = 0  # Track cumulative time for multiple segments

            # Determine appropriate split pattern based on text length
            if len(text) < 100:
                split_pattern = None  # No splitting for very short chunks
            else:
                split_pattern = r"[.!?。！？]+"

            # Use the executor to run the TTS pipeline with optimized splitting
            generator = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.pipeline(
                    text, voice=self.default_voice, speed=1, split_pattern=split_pattern
                ),
            )

            # Process all generated segments and extract NATIVE timing
            for i, result in enumerate(generator):
                # Extract the components with NATIVE timing
                gs = result.graphemes  # str
                ps = result.phonemes  # str
                audio = result.audio.cpu().numpy()  # numpy array
                tokens = result.tokens  # List[en.MToken] - THE TIMING GOLD!

                logger.info(
                    f"Chunk segment {i}: {len(tokens)} tokens, audio shape: {audio.shape}"
                )

                # Extract word timing from native tokens with null checks
                for token in tokens:
                    # Check if timing data is available
                    if token.start_ts is not None and token.end_ts is not None:
                        word_timing = {
                            "word": token.text,
                            "start_time": (token.start_ts + time_offset)
                            * 1000,  # Convert to milliseconds
                            "end_time": (token.end_ts + time_offset)
                            * 1000,  # Convert to milliseconds
                        }
                        all_word_timings.append(word_timing)
                        logger.debug(
                            f"Chunk word: '{token.text}' Start: {word_timing['start_time']:.1f}ms End: {word_timing['end_time']:.1f}ms"
                        )
                    else:
                        # Log when timing data is missing
                        logger.debug(
                            f"Chunk word: '{token.text}' - No timing data available (start_ts: {token.start_ts}, end_ts: {token.end_ts})"
                        )

                # Add audio segment
                audio_segments.append(audio)

                # Update time offset for next segment
                if len(audio) > 0:
                    segment_duration = len(audio) / 24000  # seconds
                    time_offset += segment_duration

            # Combine all audio segments
            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                self.synthesis_count += 1
                logger.info(
                    f"✨ Chunk speech synthesis complete: {len(combined_audio)} samples, {len(all_word_timings)} word timings"
                )
                return combined_audio, all_word_timings
            return None, []

        except Exception as e:
            logger.error(f"Chunk speech synthesis with timing error: {e}")
            return None, []

# Store active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        # Track current processing tasks for each client
        self.current_tasks: Dict[str, Dict[str, asyncio.Task]] = {}
        # Add image manager
        self.image_manager = ImageManager()
        # Track statistics
        self.stats = {
            "audio_segments_received": 0,
            "images_received": 0,
            "audio_with_image_received": 0,
            "last_reset": datetime.now(),
        }

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.current_tasks[client_id] = {"processing": None, "tts": None}
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.current_tasks:
            del self.current_tasks[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def cancel_current_tasks(self, client_id: str):
        """Cancel any ongoing processing tasks for a client"""
        if client_id in self.current_tasks:
            tasks = self.current_tasks[client_id]

            # Cancel processing task
            if tasks["processing"] and not tasks["processing"].done():
                logger.info(f"Cancelling processing task for client {client_id}")
                tasks["processing"].cancel()
                try:
                    await tasks["processing"]
                except asyncio.CancelledError:
                    pass

            # Cancel TTS task
            if tasks["tts"] and not tasks["tts"].done():
                logger.info(f"Cancelling TTS task for client {client_id}")
                tasks["tts"].cancel()
                try:
                    await tasks["tts"]
                except asyncio.CancelledError:
                    pass

            # Reset tasks
            self.current_tasks[client_id] = {"processing": None, "tts": None}

    def set_task(self, client_id: str, task_type: str, task: asyncio.Task):
        """Set a task for a client"""
        if client_id in self.current_tasks:
            self.current_tasks[client_id][task_type] = task

    def update_stats(self, event_type: str):
        """Update statistics"""
        if event_type in self.stats:
            self.stats[event_type] += 1

    def get_stats(self) -> dict:
        """Get current statistics"""
        uptime = datetime.now() - self.stats["last_reset"]
        return {
            **self.stats,
            "uptime_seconds": uptime.total_seconds(),
            "active_connections": len(self.active_connections),
        }


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing FrontDesk Nexus AI processors on startup...")
    try:
        WhisperProcessor.get_instance()
        NvidiaProcessor.get_instance()
        KokoroTTSProcessor.get_instance()
        logger.info("Core processors initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing processors: {e}")
        raise

    yield  # Server is running

    # Shutdown
    logger.info("Shutting down server...")
    # Close any remaining connections
    for client_id in list(manager.active_connections.keys()):
        try:
            await manager.active_connections[client_id].close()
        except Exception as e:
            logger.error(f"Error closing connection for {client_id}: {e}")
        manager.disconnect(client_id)
    logger.info("Server shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title="FrontDesk Nexus AI Backend",
    description="Realtime healthcare front desk assistant backend with STT, NVIDIA routing, and Kokoro TTS",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    return manager.get_stats()


@app.get("/images")
async def list_saved_images():
    """List all saved images"""
    try:
        images_dir = manager.image_manager.save_directory
        if not images_dir.exists():
            return {"images": [], "message": "No images directory found"}

        images = []
        for image_file in images_dir.glob("*.jpg"):
            stat = image_file.stat()
            images.append(
                {
                    "filename": image_file.name,
                    "path": str(image_file),
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                }
            )

        images.sort(key=lambda x: x["created"], reverse=True)  # Most recent first
        return {"images": images, "count": len(images)}

    except Exception as e:
        logger.error(f"Error listing images: {e}")
        return {"error": str(e)}


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time multimodal interaction."""
    await manager.connect(websocket, client_id)

    # Get instances of processors
    whisper_processor = WhisperProcessor.get_instance()
    nvidia_processor = NvidiaProcessor.get_instance()
    tts_processor = KokoroTTSProcessor.get_instance()

    try:
        # Send initial configuration confirmation
        await websocket.send_text(
            json.dumps({"status": "connected", "client_id": client_id})
        )

        async def send_keepalive():
            """Send periodic keepalive pings"""
            while True:
                try:
                    await websocket.send_text(
                        json.dumps({"type": "ping", "timestamp": time.time()})
                    )
                    await asyncio.sleep(10)  # Send ping every 10 seconds
                except Exception:
                    break

        async def process_audio_segment(audio_data, image_data=None):
            """Process a complete audio segment through the pipeline with optional image"""
            try:
                # Log what we received
                if image_data:
                    logger.info(
                        f"🎥 Processing audio+image segment: audio={len(audio_data)} bytes, image={len(image_data)} bytes"
                    )
                    manager.update_stats("audio_with_image_received")

                    # Save the image for verification
                    saved_path = manager.image_manager.save_image(
                        image_data, client_id, "multimodal"
                    )
                    if saved_path:
                        # Verify the saved image
                        verification = manager.image_manager.verify_image(saved_path)
                        if verification.get("valid"):
                            logger.info(
                                f"📸 Image verified successfully: {verification['size']} pixels"
                            )
                        else:
                            logger.warning(
                                f"⚠️ Image verification failed: {verification}"
                            )

                else:
                    logger.info(
                        f"🎤 Processing audio-only segment: {len(audio_data)} bytes"
                    )
                    manager.update_stats("audio_segments_received")

                # Send interrupt immediately since frontend determined this is valid speech
                logger.info("Sending interrupt signal")
                interrupt_message = json.dumps({"interrupt": True})
                await websocket.send_text(interrupt_message)

                # Step 1: Transcribe audio with Whisper
                logger.info("Starting Whisper transcription")
                transcribed_text = await whisper_processor.transcribe_audio(audio_data)
                logger.info(f"Transcription result: '{transcribed_text}'")

                # Check if transcription indicates noise
                if transcribed_text in ["NOISE_DETECTED", "NO_SPEECH", None]:
                    logger.info(
                        f"Noise detected in transcription: '{transcribed_text}'. Skipping further processing."
                    )
                    return

                # Step 2: Set image if provided, then process text
                if image_data:
                    await nvidia_processor.set_image(image_data)
                    logger.info("🖼️ Image set for NVIDIA multimodal context")

                # Step 3: Generate compliant admin response with NVIDIA model routing
                logger.info("Starting NVIDIA generation")
                assistant_text = await nvidia_processor.generate_response(
                    transcribed_text,
                    include_image=bool(image_data),
                )
                logger.info(
                    f"NVIDIA response: '{assistant_text[:80]}...' ({len(assistant_text)} chars)"
                )

                if not assistant_text:
                    logger.info("Empty NVIDIA response. Skipping TTS.")
                    return

                # Step 4: Generate TTS with native timing
                logger.info("Starting TTS for NVIDIA response")
                tts_task = asyncio.create_task(
                    tts_processor.synthesize_initial_speech_with_timing(assistant_text)
                )
                manager.set_task(client_id, "tts", tts_task)

                tts_result = await tts_task
                if isinstance(tts_result, tuple) and len(tts_result) == 2:
                    response_audio, response_timings = tts_result
                else:
                    response_audio = tts_result
                    response_timings = []
                    logger.warning(
                        "TTS returned single value instead of tuple - no timing data available"
                    )

                logger.info(
                    f"TTS complete: {len(response_audio) if response_audio is not None else 0} samples, {len(response_timings)} word timings"
                )

                if response_audio is not None and len(response_audio) > 0:
                    audio_bytes = (response_audio * 32767).astype(np.int16).tobytes()
                    base64_audio = base64.b64encode(audio_bytes).decode("utf-8")

                    audio_message = {
                        "audio": base64_audio,
                        "word_timings": response_timings,
                        "sample_rate": 24000,
                        "method": "native_kokoro_timing",
                        "modality": "multimodal" if image_data else "audio_only",
                    }

                    await websocket.send_text(json.dumps(audio_message))
                    logger.info(
                        f"✨ Response audio sent with {len(response_timings)} native word timings"
                    )

                await websocket.send_text(json.dumps({"audio_complete": True}))
                logger.info("Audio processing complete")

            except asyncio.CancelledError:
                logger.info("Audio processing cancelled")
                raise
            except Exception as e:
                logger.error(f"Error processing audio segment: {e}")
                logger.error(f"Full traceback: {traceback.format_exc()}")

        async def receive_and_process():
            """Receive and process messages from the client"""
            try:
                while True:
                    data = await websocket.receive_text()
                    try:
                        message = json.loads(data)

                        # Handle complete audio segments from frontend
                        if "audio_segment" in message:
                            # Cancel any current processing
                            await manager.cancel_current_tasks(client_id)

                            # Decode audio data
                            audio_data = base64.b64decode(message["audio_segment"])

                            # Check if image is also included
                            image_data = None
                            if "image" in message:
                                image_data = base64.b64decode(message["image"])
                                logger.info(
                                    f"Received audio+image: audio={len(audio_data)} bytes, image={len(image_data)} bytes"
                                )
                            else:
                                logger.info(
                                    f"Received audio-only: {len(audio_data)} bytes"
                                )

                            # Start processing the audio segment with optional image
                            processing_task = asyncio.create_task(
                                process_audio_segment(audio_data, image_data)
                            )
                            manager.set_task(client_id, "processing", processing_task)

                        # Handle standalone images (only if not currently processing)
                        elif "image" in message:
                            if not (
                                client_id in manager.current_tasks
                                and manager.current_tasks[client_id]["processing"]
                                and not manager.current_tasks[client_id][
                                    "processing"
                                ].done()
                            ):
                                image_data = base64.b64decode(message["image"])
                                manager.update_stats("images_received")

                                # Save standalone image
                                saved_path = manager.image_manager.save_image(
                                    image_data, client_id, "standalone"
                                )
                                if saved_path:
                                    verification = manager.image_manager.verify_image(
                                        saved_path
                                    )
                                    logger.info(
                                        f"📸 Standalone image saved and verified: {verification}"
                                    )

                                await nvidia_processor.set_image(image_data)
                                logger.info("Image updated")

                        # Handle realtime input (for backward compatibility)
                        elif "realtime_input" in message:
                            for chunk in message["realtime_input"]["media_chunks"]:
                                if chunk["mime_type"] == "audio/pcm":
                                    # Treat as complete audio segment
                                    await manager.cancel_current_tasks(client_id)

                                    audio_data = base64.b64decode(chunk["data"])
                                    processing_task = asyncio.create_task(
                                        process_audio_segment(audio_data)
                                    )
                                    manager.set_task(
                                        client_id, "processing", processing_task
                                    )

                                elif chunk["mime_type"] == "image/jpeg":
                                    # Only process image if not currently processing audio
                                    if not (
                                        client_id in manager.current_tasks
                                        and manager.current_tasks[client_id][
                                            "processing"
                                        ]
                                        and not manager.current_tasks[client_id][
                                            "processing"
                                        ].done()
                                    ):
                                        image_data = base64.b64decode(chunk["data"])
                                        manager.update_stats("images_received")

                                        # Save image from realtime input
                                        saved_path = manager.image_manager.save_image(
                                            image_data, client_id, "realtime"
                                        )
                                        if saved_path:
                                            verification = (
                                                manager.image_manager.verify_image(
                                                    saved_path
                                                )
                                            )
                                            logger.info(
                                                f"📸 Realtime image saved and verified: {verification}"
                                            )

                                        await nvidia_processor.set_image(image_data)

                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON: {e}")
                        await websocket.send_text(
                            json.dumps({"error": "Invalid JSON format"})
                        )
                    except KeyError as e:
                        logger.error(f"Missing key in message: {e}")
                        await websocket.send_text(
                            json.dumps({"error": f"Missing required field: {e}"})
                        )
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        await websocket.send_text(
                            json.dumps({"error": f"Processing error: {str(e)}"})
                        )

            except WebSocketDisconnect:
                logger.info("WebSocket connection closed during receive loop")

        # Run tasks concurrently
        receive_task = asyncio.create_task(receive_and_process())
        keepalive_task = asyncio.create_task(send_keepalive())

        # Wait for any task to complete (usually due to disconnection or error)
        done, pending = await asyncio.wait(
            [receive_task, keepalive_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Log results of completed tasks
        for task in done:
            try:
                result = task.result()
            except Exception as e:
                logger.error(f"Task finished with error: {e}")

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket session error for client {client_id}: {e}")
    finally:
        # Cleanup
        logger.info(f"Cleaning up resources for client {client_id}")
        await manager.cancel_current_tasks(client_id)
        manager.disconnect(client_id)


def main():
    """Main function to start the FastAPI server"""
    logger.info("Starting FrontDesk Nexus AI backend server...")

    # Configure uvicorn
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        ws_ping_interval=20,
        ws_ping_timeout=60,
        timeout_keep_alive=30,
    )

    server = uvicorn.Server(config)

    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")


if __name__ == "__main__":
    main()
