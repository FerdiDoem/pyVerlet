import { create } from 'zustand';

export interface SimState {
  running: boolean;
  speed: number;
  particleCount: number;
  radius: number;
  colourMode: string;
  fps: number;
  energy: number;
  setRunning: (v: boolean) => void;
  setSpeed: (v: number) => void;
  setParticleCount: (v: number) => void;
  setRadius: (v: number) => void;
  setColourMode: (v: string) => void;
  setFps: (v: number) => void;
  setEnergy: (v: number) => void;
}

export const useSimStore = create<SimState>((set) => ({
  running: false,
  speed: 1 / 60,
  particleCount: 100,
  radius: 50,
  colourMode: 'velocity',
  fps: 0,
  energy: 0,
  setRunning: (v) => set({ running: v }),
  setSpeed: (v) => set({ speed: v }),
  setParticleCount: (v) => set({ particleCount: v }),
  setRadius: (v) => set({ radius: v }),
  setColourMode: (v) => set({ colourMode: v }),
  setFps: (v) => set({ fps: v }),
  setEnergy: (v) => set({ energy: v }),
}));
