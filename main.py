import os

import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse


from predict import predict_label

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.post("/predict/")
async def predict_dish(request):
    content_type = request.headers.get('Content-Type')
    if content_type != 'application/octet-stream':
        raise HTTPException(status_code=400, detail="Content-Type 'application/octet-stream'")

    image_bytes = await request.body()
    temp_image_path = "temp_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(image_bytes)

    prediction = predict_label(temp_image_path)
    os.remove(temp_image_path)

    return JSONResponse(content={"prediction": prediction})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8099)
