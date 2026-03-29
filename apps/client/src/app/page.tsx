'use client';

import VoiceActivityDetector from '@/components/VoiceActivityDetector';
import TalkingHead from '@/components/TalkingHead';

export default function Home() {
  return (
    <main className="relative h-[100dvh] w-full overflow-hidden bg-[#f0ece6]">
      {/* Top bar */}
      <header className="absolute inset-x-0 top-0 z-30 flex items-center justify-between px-5 pt-4 pb-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-[#1a1a2e]">
            FrontDesk<span className="font-light text-[#5a5a7a]"> Nexus</span>
          </h1>
        </div>
        <div className="rounded-full bg-amber-100/80 px-3 py-1 text-[11px] font-medium text-amber-800 backdrop-blur-sm">
          Admin only — no medical advice
        </div>
      </header>

      {/* 3D Avatar — fills the viewport */}
      <div className="absolute inset-0 z-0">
        <TalkingHead />
      </div>

      {/* Voice controls — anchored to bottom */}
      <div className="absolute inset-x-0 bottom-0 z-20">
        <VoiceActivityDetector />
      </div>
    </main>
  );
}
