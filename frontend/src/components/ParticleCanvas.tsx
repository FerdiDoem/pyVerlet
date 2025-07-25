import { useEffect, useRef } from 'react';
import { Graphics, Application } from 'pixi.js';

const WS_PATH = '/ws';

export default function ParticleCanvas() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const app = new Application({
      resizeTo: containerRef.current ?? undefined,
      backgroundAlpha: 0,
    });
    const target = containerRef.current;
    if (!target) return;
    target.appendChild(app.view as HTMLCanvasElement);

    const base = (import.meta.env.VITE_API_URL as string) || 'http://localhost:8000';
    const wsUrl = new URL(WS_PATH, base);
    wsUrl.protocol = wsUrl.protocol.replace('http', 'ws');
    const ws = new WebSocket(wsUrl.toString());

    ws.binaryType = 'arraybuffer';

    ws.onmessage = (ev) => {
      const arr = new Float32Array(ev.data);
      app.stage.removeChildren();
      for (let i = 0; i < arr.length; i += 2) {
        const g = new Graphics();
        g.beginFill(0xffffff);
        g.drawCircle(arr[i], arr[i + 1], 3);
        g.endFill();
        app.stage.addChild(g);
      }
    };

    return () => {
      ws.close();
      app.destroy(true);
    };
  }, []);

  return <div ref={containerRef} className="h-full w-full" />;
}
