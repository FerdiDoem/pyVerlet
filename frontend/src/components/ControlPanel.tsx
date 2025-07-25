import { post } from '../api';
import { useSimStore } from '../store/useSimStore';

export default function ControlPanel() {
  const {
    running,
    speed,
    particleCount,
    radius,
    colourMode,
    setRunning,
    setSpeed,
    setParticleCount,
    setRadius,
    setColourMode,
  } = useSimStore();

  const toggle = () => setRunning(!running);
  const reset = () => post('/reset', {});
  return (
    <div className="space-y-2 text-sm">
      <div className="flex items-center space-x-2">
        <button
          onClick={toggle}
          className="px-2 py-1 rounded bg-blue-600 text-white"
        >
          {running ? 'Pause' : 'Play'}
        </button>
        <button
          onClick={reset}
          className="px-2 py-1 rounded bg-gray-500 text-white"
        >
          Reset
        </button>
      </div>
      <div>
        <label className="block">Speed {speed.toFixed(3)}</label>
        <input
          type="range"
          min="0.001"
          max="0.1"
          step="0.001"
          value={speed}
          onChange={(e) => setSpeed(parseFloat(e.target.value))}
        />
      </div>
      <div>
        <label className="block">Particles {particleCount}</label>
        <input
          type="range"
          min="10"
          max="2000"
          step="10"
          value={particleCount}
          onChange={(e) => setParticleCount(parseInt(e.target.value))}
        />
      </div>
      <div>
        <label className="block">Radius {radius}</label>
        <input
          type="range"
          min="10"
          max="200"
          step="1"
          value={radius}
          onChange={(e) => setRadius(parseInt(e.target.value))}
        />
      </div>
      <div>
        <label className="block">Colour</label>
        <select
          className="w-full border p-1"
          value={colourMode}
          onChange={(e) => setColourMode(e.target.value)}
        >
          <option value="velocity">Velocity</option>
          <option value="id">Id</option>
        </select>
      </div>
    </div>
  );
}
