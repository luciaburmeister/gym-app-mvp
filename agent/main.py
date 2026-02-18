import os
import json
from vision_agents import Agent
from vision_agents.llm import gemini  # or openai realtime later
from dotenv import load_dotenv
load_dotenv()

# Load environment variables
CALL_TYPE = os.getenv("CALL_TYPE")
CALL_ID = os.getenv("CALL_ID")

agent = Agent(
    name="Gym Coach MVP",
    instructions="""
You are a fitness coach AI.
Always respond ONLY in valid JSON using this format:

{
  "exercise": {"label": "unknown", "confidence": 0.0},
  "rep_count": 0,
  "feedback": "",
  "severity": "low",
  "confidence": 0.0
}

Never output anything outside JSON.
""",
    llm=gemini.Realtime(fps=3)  # change later if needed
)

if __name__ == "__main__":
    agent.run(call_type=CALL_TYPE, call_id=CALL_ID)
