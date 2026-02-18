import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from getstream import Stream

load_dotenv()

API_KEY = os.environ["STREAM_API_KEY"]
API_SECRET = os.environ["STREAM_API_SECRET"]

client = Stream(api_key=API_KEY, api_secret=API_SECRET)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/token")
def token(user_id: str):
    # Create a user token (JWT) for the client SDK
    token = client.create_token(user_id)
    return {"user_id": user_id, "token": token}
