from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from predict import predict_label
import os
import io
from PIL import Image

app = FastAPI()

# Загрузка изображения и получение предсказания
@app.post("/predict/")
async def predict_dish(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    image_bytes = await file.read()

    # Создание временного файла для сохранения изображения
    temp_image_path = "temp_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(image_bytes)

    try:
        prediction = predict_label(temp_image_path)
    finally:
        # Удаление временного файла
        os.remove(temp_image_path)

    return JSONResponse(content={"prediction": prediction})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
