import { useSimStore } from '../store/useSimStore';

export default function Metrics() {
  const fps = useSimStore((s) => s.fps);
  const energy = useSimStore((s) => s.energy);
  return (
    <div className="text-sm space-y-1">
      <div>FPS: {fps.toFixed(1)}</div>
      <div>Energy: {energy.toFixed(2)}</div>
    </div>
  );
}
