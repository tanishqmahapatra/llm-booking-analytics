from fastapi import FastAPI
from pydantic import BaseModel
import json

app = FastAPI()

# Load precomputed insights
with open("insights.json", "r") as file:
    insights = json.load(file)

class Query(BaseModel):
    question: str

@app.post("/analytics")
def get_analytics():
    return {"analytics": insights}

@app.post("/ask")
def ask_question(query: Query):
    question = query.question.lower()

    if "total bookings" in question:
        return {"answer": insights["total_bookings_per_hotel"]}
    elif "canceled bookings" in question:
        return {"answer": insights["canceled_bookings"][:5]}  # Sending first 5 cancellations as example
    else:
        return {"answer": "I don't have an answer for that yet."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
