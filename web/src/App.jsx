import { useEffect, useMemo, useState } from "react";
import {
  StreamVideo,
  StreamCall,
  StreamVideoClient,
  useCall,
  CallControls,
  SpeakerLayout,
} from "@stream-io/video-react-sdk";

import "@stream-io/video-react-sdk/dist/css/styles.css";

const apiKey = "6puba82h4dns"; // your Stream API key
const callType = "default";
const callId = "gym-mvp";

function CallComponent() {
  const call = useCall();

  useEffect(() => {
    if (!call) return;

    const setup = async () => {
      await call.join({ create: true });
      await call.camera.enable();
    };

    setup().catch((e) => console.error("Join/setup error:", e));
  }, [call]);

  return (
    <div style={{ height: "100vh", background: "#111" }}>
      <SpeakerLayout />
      <CallControls />
    </div>
  );
}

export default function App() {
  const [token, setToken] = useState(null);

  useEffect(() => {
    const run = async () => {
      const userId = "lucia";
      const res = await fetch(`http://localhost:8001/token?user_id=${userId}`);
      const data = await res.json();
      console.log("Token response:", data);
      console.log("UserId used by client:", userId);
      console.log("Token response:", data);
      setToken(data.token);
    };

    run().catch((e) => console.error("Token fetch error:", e));
  }, []);

  const client = useMemo(() => {
    if (!token) return null;
    return new StreamVideoClient({
      apiKey,
      user: { id: "lucia" },
      token,
    });
  }, [token]);

  const call = useMemo(() => {
    if (!client) return null;
    return client.call(callType, callId);
  }, [client]);

  if (!client || !call) return <div style={{ padding: 20 }}>Loading…</div>;

  return (
    <StreamVideo client={client}>
      <StreamCall call={call}>
        <CallComponent />
      </StreamCall>
    </StreamVideo>
  );
}
