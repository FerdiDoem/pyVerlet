const BASE = (import.meta.env.VITE_API_URL as string) || 'http://localhost:8000';

export async function post(path: string, body: Record<string, unknown>) {
  const res = await fetch(new URL(path, BASE).toString(), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error('Request failed');
  }
  return res.json();
}

export function connectWs(path: string, onMessage: (ev: MessageEvent) => void) {
  let ws: WebSocket;
  function connect() {
    const url = new URL(path, BASE);
    url.protocol = url.protocol.replace('http', 'ws');
    ws = new WebSocket(url.toString());
    ws.binaryType = 'arraybuffer';
    ws.onmessage = onMessage;
    ws.onclose = () => {
      setTimeout(connect, 1000);
    };
  }
  connect();
  return ws;
}
