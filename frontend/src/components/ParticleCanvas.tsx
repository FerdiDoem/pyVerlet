import { useEffect, useRef } from 'react';
import { Graphics, Application } from 'pixi.js';
import { connectWs, post } from '../api';
import { useSimStore } from '../store/useSimStore';

export default function ParticleCanvas() {
  const containerRef = useRef<HTMLDivElement>(null);
  const appRef = useRef<Application>();
  const wsRef = useRef<WebSocket | null>(null);
  const last = useRef<number>(performance.now());

  const running = useSimStore((s) => s.running);
  const speed = useSimStore((s) => s.speed);
  const particleCount = useSimStore((s) => s.particleCount);
  const radius = useSimStore((s) => s.radius);
  const colourMode = useSimStore((s) => s.colourMode);
  const setFps = useSimStore((s) => s.setFps);
  const setEnergy = useSimStore((s) => s.setEnergy);

  useEffect(() => {
    post('/params', {
      particles: particleCount,
      radius,
      dt: speed,
      colour_mode: colourMode,
    }).catch(() => {});
  }, [particleCount, radius, speed, colourMode]);

  useEffect(() => {
    const app = new Application({
      resizeTo: containerRef.current ?? undefined,
      backgroundAlpha: 0,
    });
    appRef.current = app;
    const target = containerRef.current;
    if (target) {
      target.appendChild(app.view as HTMLCanvasElement);
    }
    return () => {
      wsRef.current?.close();
      app.destroy(true);
    };
  }, []);

  useEffect(() => {
    const app = appRef.current;
    if (!app) return;

    const handle = (ev: MessageEvent) => {
      const arr = new Float32Array(ev.data);
      app.stage.removeChildren();
      let energy = 0;
      for (let i = 0; i < arr.length; i += 5) {
        const x = arr[i];
        const y = arr[i + 1];
        const r = arr[i + 2];
        const vx = arr[i + 3];
        const vy = arr[i + 4];
        energy += 0.5 * (vx * vx + vy * vy);
        const g = new Graphics();
        g.beginFill(0xffffff);
        g.drawCircle(x, y, r);
        g.endFill();
        app.stage.addChild(g);
      }
      const now = performance.now();
      setFps(1000 / (now - last.current));
      last.current = now;
      setEnergy(energy);
    };

    if (running) {
      wsRef.current = connectWs('/ws', handle);
    } else {
      wsRef.current?.close();
      wsRef.current = null;
    }

    return () => {
      wsRef.current?.close();
    };
  }, [running]);

  return <div ref={containerRef} className="h-full w-full" />;
}
