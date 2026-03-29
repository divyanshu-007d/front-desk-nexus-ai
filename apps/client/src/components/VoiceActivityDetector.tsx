'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Mic } from 'lucide-react';
import { useWebSocketContext } from '@/contexts/WebSocketContext';

interface VADConfig {
  energyThreshold: number;
  conversationBreakDuration: number;
  minSpeechDuration: number;
  maxSpeechDuration: number;
  sampleRate: number;
}

interface VoiceActivityDetectorProps {
  className?: string;
}

const VoiceActivityDetector: React.FC<VoiceActivityDetectorProps> = ({
  className
}) => {
  const [isListening, setIsListening] = useState(false);
  const [isPressToTalkActive, setIsPressToTalkActive] = useState(false);
  const [currentEnergy, setCurrentEnergy] = useState(0);
  const [isSpeechActive, setIsSpeechActive] = useState(false);

  const [config] = useState<VADConfig>({
    energyThreshold: 0.02,
    conversationBreakDuration: 2.5,
    minSpeechDuration: 0.8,
    maxSpeechDuration: 15,
    sampleRate: 16000
  });

  // Audio processing refs
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);

  // Speech detection state
  const audioBufferRef = useRef<Float32Array[]>([]);
  const silenceFramesRef = useRef(0);
  const speechFramesRef = useRef(0);
  const isInSpeechRef = useRef(false);
  const speechStartTimeRef = useRef(0);

  // WebSocket integration
  const { isConnected, connect, sendAudioSegment } = useWebSocketContext();

  const notifyUserGesture = useCallback(() => {
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new Event('frontdesk-user-gesture'));
    }
  }, []);

  // Auto-connect to WebSocket on mount
  useEffect(() => {
    connect();
  }, [connect]);

  // Create and send audio
  const createAndSendAudio = useCallback(
    (audioBuffers: Float32Array[]) => {
      if (!isConnected) return;

      const totalLength = audioBuffers.reduce(
        (sum, buffer) => sum + buffer.length,
        0
      );
      const combinedBuffer = new Float32Array(totalLength);

      let offset = 0;
      for (const buffer of audioBuffers) {
        combinedBuffer.set(buffer, offset);
        offset += buffer.length;
      }

      // Convert Float32 to Int16
      const int16Buffer = new Int16Array(combinedBuffer.length);
      for (let i = 0; i < combinedBuffer.length; i++) {
        int16Buffer[i] = Math.max(
          -32768,
          Math.min(32767, combinedBuffer[i] * 32767)
        );
      }

      const audioData = int16Buffer.buffer;
      sendAudioSegment(audioData);
      console.log('Sent audio to server', { audioSize: audioData.byteLength });
    },
    [isConnected, sendAudioSegment]
  );

  // Initialize audio worklet
  const initializeAudioWorklet = useCallback(async () => {
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext({
          sampleRate: config.sampleRate
        });
      }

      const audioContext = audioContextRef.current;

      const workletCode = `
        class VoiceActivityProcessor extends AudioWorkletProcessor {
          constructor() {
            super();
            this.bufferSize = 1024;
            this.buffer = new Float32Array(this.bufferSize);
            this.bufferIndex = 0;
          }

          process(inputs, outputs, parameters) {
            const input = inputs[0];
            
            if (input.length > 0) {
              const inputChannel = input[0];
              
              for (let i = 0; i < inputChannel.length; i++) {
                this.buffer[this.bufferIndex] = inputChannel[i];
                this.bufferIndex++;
                
                if (this.bufferIndex >= this.bufferSize) {
                  let sum = 0;
                  for (let j = 0; j < this.bufferSize; j++) {
                    sum += this.buffer[j] * this.buffer[j];
                  }
                  const energy = Math.sqrt(sum / this.bufferSize);
                  
                  this.port.postMessage({
                    type: 'audioData',
                    energy: energy,
                    audioData: new Float32Array(this.buffer)
                  });
                  
                  this.bufferIndex = 0;
                }
              }
            }
            
            return true;
          }
        }

        registerProcessor('voice-activity-processor', VoiceActivityProcessor);
      `;

      const blob = new Blob([workletCode], { type: 'application/javascript' });
      const workletUrl = URL.createObjectURL(blob);
      await audioContext.audioWorklet.addModule(workletUrl);
      URL.revokeObjectURL(workletUrl);

      return audioContext;
    } catch (error) {
      console.error('Failed to initialize audio worklet:', error);
      throw error;
    }
  }, [config.sampleRate]);

  // Process audio data for speech detection
  const processAudioData = useCallback(
    (energy: number, audioData: Float32Array) => {
      setCurrentEnergy(energy);

      const {
        energyThreshold,
        conversationBreakDuration,
        minSpeechDuration,
        maxSpeechDuration,
        sampleRate
      } = config;

      const conversationBreakFrames = Math.floor(
        (conversationBreakDuration * sampleRate) / 1024
      );
      const minSpeechFrames = Math.floor(
        (minSpeechDuration * sampleRate) / 1024
      );
      const maxSpeechFrames = Math.floor(
        (maxSpeechDuration * sampleRate) / 1024
      );

      audioBufferRef.current.push(new Float32Array(audioData));

      if (energy > energyThreshold) {
        if (!isInSpeechRef.current) {
          console.log('Speech started');
          isInSpeechRef.current = true;
          speechStartTimeRef.current = Date.now();
          setIsSpeechActive(true);
        }
        speechFramesRef.current++;
        silenceFramesRef.current = 0;
      } else {
        if (isInSpeechRef.current) {
          silenceFramesRef.current++;

          if (
            silenceFramesRef.current >= conversationBreakFrames &&
            speechFramesRef.current >= minSpeechFrames
          ) {
            console.log('Saving speech segment');
            createAndSendAudio(audioBufferRef.current);

            speechFramesRef.current = 0;
            silenceFramesRef.current = 0;
            isInSpeechRef.current = false;
            audioBufferRef.current = [];
            setIsSpeechActive(false);
          }
        }
      }

      // Handle max duration
      if (isInSpeechRef.current && speechFramesRef.current >= maxSpeechFrames) {
        console.log('Max duration reached, saving segment');
        createAndSendAudio(audioBufferRef.current);

        speechStartTimeRef.current = Date.now();
        speechFramesRef.current = 0;
        silenceFramesRef.current = 0;
        audioBufferRef.current = [new Float32Array(audioData)];
      }

      // Prevent buffer overflow during long silences
      if (
        !isInSpeechRef.current &&
        audioBufferRef.current.length > conversationBreakFrames * 2
      ) {
        audioBufferRef.current = audioBufferRef.current.slice(
          -conversationBreakFrames
        );
      }
    },
    [config, createAndSendAudio]
  );

  // Start listening
  const startListening = useCallback(async () => {
    if (isListening) return;

    try {
      notifyUserGesture();

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: config.sampleRate,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      mediaStreamRef.current = stream;
      const audioContext = await initializeAudioWorklet();
      const source = audioContext.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(
        audioContext,
        'voice-activity-processor'
      );
      workletNodeRef.current = workletNode;

      workletNode.port.onmessage = (event) => {
        const { type, energy, audioData } = event.data;
        if (type === 'audioData') {
          processAudioData(energy, audioData);
        }
      };

      source.connect(workletNode);
      setIsListening(true);
      console.log('Started voice detection');
    } catch (error) {
      console.error('Failed to start listening:', error);
    }
  }, [
    isListening,
    notifyUserGesture,
    config.sampleRate,
    initializeAudioWorklet,
    processAudioData
  ]);

  // Stop listening
  const stopListening = useCallback(
    (flushAudio = false) => {
      if (flushAudio && audioBufferRef.current.length > 0 && isConnected) {
        const minSpeechFrames = Math.floor(
          (config.minSpeechDuration * config.sampleRate) / 1024
        );

        if (
          speechFramesRef.current >= minSpeechFrames ||
          isInSpeechRef.current
        ) {
          console.log('Flushing held speech segment');
          createAndSendAudio(audioBufferRef.current);
        }
      }

      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
      }

      if (workletNodeRef.current) {
        workletNodeRef.current.disconnect();
        workletNodeRef.current = null;
      }

      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }

      // Reset state
      audioBufferRef.current = [];
      silenceFramesRef.current = 0;
      speechFramesRef.current = 0;
      isInSpeechRef.current = false;

      setIsListening(false);
      setIsPressToTalkActive(false);
      setCurrentEnergy(0);
      setIsSpeechActive(false);
      console.log('Stopped listening');
    },
    [
      isConnected,
      config.minSpeechDuration,
      config.sampleRate,
      createAndSendAudio
    ]
  );

  // Toggle listening (continuous mode)
  const toggleListening = useCallback(() => {
    if (isPressToTalkActive) return;
    if (isListening) {
      stopListening(false);
    } else {
      startListening();
    }
  }, [isPressToTalkActive, isListening, startListening, stopListening]);

  // Hold-to-speak handlers
  const handleHoldStart = useCallback(
    async (event: React.PointerEvent<HTMLButtonElement>) => {
      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);

      if (!isConnected || (isListening && !isPressToTalkActive)) return;

      if (!isPressToTalkActive) {
        setIsPressToTalkActive(true);
        await startListening();
      }
    },
    [isConnected, isListening, isPressToTalkActive, startListening]
  );

  const handleHoldEnd = useCallback(() => {
    if (!isPressToTalkActive) return;
    stopListening(true);
  }, [isPressToTalkActive, stopListening]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopListening(false);
    };
  }, [stopListening]);

  // Derive visual state
  const isActive = isPressToTalkActive || isListening;
  const pulseScale = 1 + Math.min(currentEnergy / 0.05, 1) * 0.3;

  return (
    <div className={`flex flex-col items-center pt-4 pb-10 ${className ?? ''}`}>
      {/* Gradient fade from transparent to a soft ground */}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-48 bg-gradient-to-t from-[#f0ece6] via-[#f0ece6]/80 to-transparent" />

      {/* Content sits above the gradient */}
      <div className="relative z-10 flex flex-col items-center gap-5">
        {/* Listening indicator text */}
        <p className="text-[13px] font-medium text-[#5a5a7a]/80">
          {!isConnected
            ? 'Connecting to server…'
            : isPressToTalkActive
              ? 'Listening — release to send'
              : isListening
                ? isSpeechActive
                  ? 'Hearing you…'
                  : 'Listening…'
                : 'Hold the mic to speak'}
        </p>

        {/* Mic button */}
        <div className="relative flex items-center justify-center">
          {/* Pulsing ring when active */}
          {isActive && (
            <div
              className="absolute h-20 w-20 rounded-full border-2 border-[#1a1a2e]/20 transition-transform duration-100"
              style={{ transform: `scale(${pulseScale})` }}
            />
          )}

          <button
            type="button"
            className={`relative flex h-16 w-16 touch-none items-center justify-center rounded-full shadow-lg transition-all duration-200 active:scale-95 ${
              isActive
                ? 'bg-[#1a1a2e] text-white shadow-[#1a1a2e]/30'
                : isConnected
                  ? 'bg-white text-[#1a1a2e] shadow-black/10 hover:shadow-lg'
                  : 'cursor-not-allowed bg-gray-200 text-gray-400'
            }`}
            disabled={!isConnected}
            onPointerDown={handleHoldStart}
            onPointerUp={handleHoldEnd}
            onPointerCancel={handleHoldEnd}
            onPointerLeave={handleHoldEnd}
          >
            <Mic size={24} />
          </button>
        </div>

        {/* Tap-to-start option (small text link) */}
        <button
          onClick={toggleListening}
          disabled={!isConnected || isPressToTalkActive}
          className="text-[11px] text-[#5a5a7a]/60 underline decoration-dotted underline-offset-2 transition-colors hover:text-[#5a5a7a] disabled:cursor-not-allowed disabled:no-underline disabled:opacity-40"
        >
          {isListening
            ? 'Stop continuous mode'
            : 'or tap for continuous listening'}
        </button>
      </div>
    </div>
  );
};

export default VoiceActivityDetector;
