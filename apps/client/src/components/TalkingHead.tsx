'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Loader2 } from 'lucide-react';
import { useWebSocketContext } from '@/contexts/WebSocketContext';

interface TalkingHeadProps {
  className?: string;
}

const TalkingHead: React.FC<TalkingHeadProps> = ({
  className = ''
}) => {
  const avatarRef = useRef<HTMLDivElement>(null);
  const headRef = useRef<any>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioQueueRef = useRef<any[]>([]);
  const isPlayingAudioRef = useRef(false);

  const [isLoading, setIsLoading] = useState(true);

  const [selectedAvatar, setSelectedAvatar] = useState('F');
  const [selectedMood, setSelectedMood] = useState('neutral');
  const [scriptsLoaded, setScriptsLoaded] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const buildReadyPlayerAvatarUrl = useCallback((avatarId: string, withMorphTargets: boolean) => {
    const url = new URL(`https://models.readyplayer.me/${avatarId}.glb`);

    if (withMorphTargets) {
      url.searchParams.set(
        'morphTargets',
        [
          'ARKit',
          'Oculus Visemes',
          'mouthOpen',
          'mouthSmile',
          'eyesClosed',
          'eyesLookUp',
          'eyesLookDown'
        ].join(',')
      );
      url.searchParams.set('textureSizeLimit', '1024');
      url.searchParams.set('textureFormat', 'png');
    }

    return url.toString();
  }, []);

  const getAvatarUrlCandidates = useCallback(
    (gender: string) => {
      const envCommon = process.env.NEXT_PUBLIC_AVATAR_URL;
      const envFemale = process.env.NEXT_PUBLIC_AVATAR_URL_F;
      const envMale = process.env.NEXT_PUBLIC_AVATAR_URL_M;

      const defaultFemaleUrl = buildReadyPlayerAvatarUrl(
        '64bfa15f0e72c63d7c3934a6',
        true
      );
      const defaultMaleUrl = buildReadyPlayerAvatarUrl(
        '638df5d0d72bffc6fa179441',
        false
      );

      const candidates = [
        gender === 'F' ? envFemale : envMale,
        envCommon,
        gender === 'F' ? defaultFemaleUrl : defaultMaleUrl,
        '/avatars/brunette.glb'
      ];

      return [...new Set(candidates.filter((value): value is string => Boolean(value)))];
    },
    [buildReadyPlayerAvatarUrl]
  );

  // Get WebSocket context
  const {
    isConnected,
    connect,
    onAudioReceived,
    onInterrupt,
    onError,
    onStatusChange
  } = useWebSocketContext();

  // Initialize audio context
  const initAudioContext = useCallback(async () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext({ sampleRate: 22050 });
    }
    if (audioContextRef.current.state === 'suspended') {
      await audioContextRef.current.resume();
    }
  }, []);

  // Convert base64 to ArrayBuffer
  const base64ToArrayBuffer = useCallback((base64: string) => {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
  }, []);

  // Convert Int16Array to Float32Array
  const int16ArrayToFloat32 = useCallback((int16Array: Int16Array) => {
    const float32Array = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i++) {
      float32Array[i] = int16Array[i] / 32768.0;
    }
    return float32Array;
  }, []);

  // Play next audio in queue
  const playNextAudio = useCallback(async () => {
    if (audioQueueRef.current.length === 0) {
      isPlayingAudioRef.current = false;
      setIsSpeaking(false);
      return;
    }

    isPlayingAudioRef.current = true;
    setIsSpeaking(true);

    const audioItem = audioQueueRef.current.shift();
    console.log('Playing audio item:', audioItem);

    try {
      if (
        headRef.current &&
        audioItem.timingData &&
        audioItem.timingData.words
      ) {
        // Use TalkingHead with native timing
        const speakData = {
          audio: audioItem.buffer,
          words: audioItem.timingData.words,
          wtimes: audioItem.timingData.word_times,
          wdurations: audioItem.timingData.word_durations
        };

        console.log('Using TalkingHead with timing data:', speakData);
        headRef.current.speakAudio(speakData);

        // Set timer for next audio
        setTimeout(() => {
          console.log('TalkingHead audio finished, playing next...');
          playNextAudio();
        }, audioItem.duration * 1000);
      } else if (headRef.current) {
        // Basic TalkingHead audio without timing
        console.log('Using basic TalkingHead audio');
        headRef.current.speakAudio({ audio: audioItem.buffer });

        setTimeout(() => {
          console.log('Basic TalkingHead audio finished, playing next...');
          playNextAudio();
        }, audioItem.duration * 1000);
      } else {
        // Fallback to Web Audio API
        console.log('Using Web Audio API fallback');
        await initAudioContext();
        const source = audioContextRef.current!.createBufferSource();
        source.buffer = audioItem.buffer;
        source.connect(audioContextRef.current!.destination);
        source.onended = () => {
          console.log('Web Audio finished, playing next...');
          playNextAudio();
        };
        source.start();
      }
    } catch (error) {
      console.error('Error playing audio:', error);
      // Continue to next audio on error
      setTimeout(() => playNextAudio(), 100);
    }
  }, [initAudioContext]);

  // Handle audio from WebSocket
  const handleAudioReceived = useCallback(
    async (
      base64Audio: string,
      timingData?: any,
      sampleRate = 24000,
      method = 'unknown'
    ) => {
      console.log('🎵 TALKINGHEAD handleAudioReceived CALLED!', {
        audioLength: base64Audio.length,
        timingData,
        sampleRate,
        method
      });

      try {
        await initAudioContext();

        // Convert base64 to audio buffer
        const arrayBuffer = base64ToArrayBuffer(base64Audio);
        const int16Array = new Int16Array(arrayBuffer);
        const float32Array = int16ArrayToFloat32(int16Array);

        console.log('Audio conversion successful:', {
          arrayBufferLength: arrayBuffer.byteLength,
          int16Length: int16Array.length,
          float32Length: float32Array.length
        });

        // Create AudioBuffer
        const audioBuffer = audioContextRef.current!.createBuffer(
          1,
          float32Array.length,
          sampleRate
        );
        audioBuffer.copyToChannel(float32Array, 0);

        console.log('AudioBuffer created:', {
          duration: audioBuffer.duration,
          sampleRate: audioBuffer.sampleRate,
          length: audioBuffer.length
        });

        // Add to queue
        audioQueueRef.current.push({
          buffer: audioBuffer,
          timingData: timingData,
          duration: audioBuffer.duration,
          method: method
        });

        console.log(
          'Audio added to queue. Queue length:',
          audioQueueRef.current.length
        );

        // Start playing if not already playing
        if (!isPlayingAudioRef.current) {
          console.log('Starting audio playback...');
          playNextAudio();
        } else {
          console.log('Audio already playing, added to queue');
        }

        const timingInfo = timingData
          ? ` with ${timingData.words?.length || 0} word timings`
          : ' (no timing)';
        console.log(
          `✅ Audio queued successfully: ${audioBuffer.duration.toFixed(2)}s${timingInfo} [${method}]`
        );
      } catch (error) {
        console.error(
          '❌ Error processing audio in handleAudioReceived:',
          error
        );
      }
    },
    [initAudioContext, base64ToArrayBuffer, int16ArrayToFloat32, playNextAudio]
  );

  // Handle interrupt from server
  const handleInterrupt = useCallback(() => {
    // Clear audio queue
    audioQueueRef.current = [];
    isPlayingAudioRef.current = false;
    setIsSpeaking(false);

    // Stop TalkingHead if speaking
    // if (headRef.current) {
    //   try {
    //     headRef.current.stop();
    //   } catch (error) {
    //     console.error('Error stopping TalkingHead:', error);
    //   }
    // }

    console.log('Audio interrupted and cleared');
  }, []);

  // Register WebSocket callbacks
  useEffect(() => {
    onAudioReceived(handleAudioReceived);
    onInterrupt(handleInterrupt);
    onError((error) => console.error('WebSocket error:', error));
    onStatusChange((status) => {
      console.log('WebSocket status:', status);
    });
  }, [
    onAudioReceived,
    onInterrupt,
    onError,
    onStatusChange,
    handleAudioReceived,
    handleInterrupt
  ]);

  // Unlock audio playback from explicit user gestures across components
  useEffect(() => {
    const handleUserGesture = () => {
      initAudioContext().catch((error) => {
        console.error('Audio unlock failed:', error);
      });
    };

    window.addEventListener('frontdesk-user-gesture', handleUserGesture);
    return () => {
      window.removeEventListener('frontdesk-user-gesture', handleUserGesture);
    };
  }, [initAudioContext]);

  // Listen for TalkingHead library to load
  useEffect(() => {
    const handleTalkingHeadLoaded = () => {
      setScriptsLoaded(true);
    };

    const handleTalkingHeadError = () => {
      console.error('Failed to load TalkingHead library');
    };

    if ((window as any).TalkingHead) {
      setScriptsLoaded(true);
      return;
    }

    window.addEventListener('talkinghead-loaded', handleTalkingHeadLoaded);
    window.addEventListener('talkinghead-error', handleTalkingHeadError);

    return () => {
      window.removeEventListener('talkinghead-loaded', handleTalkingHeadLoaded);
      window.removeEventListener('talkinghead-error', handleTalkingHeadError);
    };
  }, []);

  // Initialize TalkingHead
  useEffect(() => {
    if (!scriptsLoaded || !avatarRef.current) return;

    const initTalkingHead = async () => {
      try {
        setIsLoading(true);

        const TalkingHead = (window as any).TalkingHead;
        if (!TalkingHead) {
          throw new Error('TalkingHead library not loaded');
        }

        headRef.current = new TalkingHead(avatarRef.current, {
          ttsEndpoint: 'https://texttospeech.googleapis.com/v1/text:synthesize',
          jwtGet: () => Promise.resolve('dummy-jwt-token'),
          lipsyncModules: ['en'],
          lipsyncLang: 'en',
          modelFPS: 30,
          cameraView: 'full',
          avatarMute: false,
          avatarMood: selectedMood
        });

        await loadAvatar(selectedAvatar);
        setIsLoading(false);

        // Auto-connect to WebSocket
        connect();
      } catch (error: any) {
        setIsLoading(false);
        console.error('Failed to initialize TalkingHead:', error);
      }
    };

    initTalkingHead();

    return () => {
      if (headRef.current) {
        try {
          headRef.current.stop();
        } catch (error) {
          console.error('Cleanup error:', error);
        }
      }
    };
  }, [scriptsLoaded, connect]);

  const loadAvatar = async (gender: string = 'F') => {
    const candidates = getAvatarUrlCandidates(gender);
    let lastError: string | null = null;

    for (const url of candidates) {
      try {
        await headRef.current?.showAvatar({
          url,
          body: gender,
          avatarMood: selectedMood,
          lipsyncLang: 'en'
        });

        return;
      } catch (error: any) {
        lastError = error?.message || 'Unknown avatar loading error';
        console.warn(`Avatar load failed for ${url}:`, error);
      }
    }

    console.error(`Failed to load avatar: ${lastError || 'all sources failed'}`);
  };

  const handleAvatarChange = (gender: string) => {
    setSelectedAvatar(gender);
    if (scriptsLoaded && headRef.current) {
      loadAvatar(gender);
    }
  };

  const handleMoodChange = (mood: string) => {
    setSelectedMood(mood);
    if (headRef.current) {
      headRef.current.setMood(mood);
    }
  };

  return (
    <div className={`relative h-full w-full ${className}`}>
      {/* Full-bleed avatar */}
      <div ref={avatarRef} className="h-full w-full" />

      {/* Loading overlay */}
      {(isLoading || !scriptsLoaded) && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#f0ece6]">
          <Loader2 className="mb-3 h-8 w-8 animate-spin text-[#5a5a7a]" />
          <p className="text-sm text-[#5a5a7a]">
            {!scriptsLoaded ? 'Loading avatar engine…' : 'Preparing your assistant…'}
          </p>
        </div>
      )}

      {/* Subtle connection dot — top-left */}
      {scriptsLoaded && !isLoading && (
        <div className="absolute top-16 left-5 flex items-center gap-2">
          <div
            className={`h-2 w-2 rounded-full ${
              isConnected ? 'bg-emerald-500' : 'bg-red-400'
            }`}
          />
          <span className="text-[11px] font-medium text-[#5a5a7a]/70">
            {isConnected ? 'Online' : 'Offline'}
          </span>
        </div>
      )}

      {/* Speaking indicator */}
      {isSpeaking && (
        <div className="absolute top-16 right-5 flex items-center gap-1.5">
          <div className="flex gap-0.5">
            <div className="h-2 w-0.5 animate-pulse rounded-full bg-[#5a5a7a]/60" style={{ animationDelay: '0ms' }} />
            <div className="h-3 w-0.5 animate-pulse rounded-full bg-[#5a5a7a]/60" style={{ animationDelay: '150ms' }} />
            <div className="h-2 w-0.5 animate-pulse rounded-full bg-[#5a5a7a]/60" style={{ animationDelay: '300ms' }} />
          </div>
          <span className="text-[11px] font-medium text-[#5a5a7a]/70">Speaking</span>
        </div>
      )}
    </div>
  );
};

export default TalkingHead;
