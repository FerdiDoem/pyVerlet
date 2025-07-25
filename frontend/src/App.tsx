import ParticleCanvas from './components/ParticleCanvas';
import ControlPanel from './components/ControlPanel';
import Metrics from './components/Metrics';

export default function App() {
  return (
    <div className="h-full w-full flex flex-col md:flex-row">
      <aside className="bg-gray-900 text-white p-2 w-full md:w-64">
        <ControlPanel />
        <div className="mt-4">
          <Metrics />
        </div>
      </aside>
      <main className="flex-1 relative">
        <ParticleCanvas />
      </main>
    </div>
  );
}
