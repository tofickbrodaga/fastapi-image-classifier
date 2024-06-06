from PIL import Image
from fastapi import FastAPI, File, UploadFile
from predict import predict_label

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/uploadfile/")
async def create_upload_file(file: bytes = File(...)):

    # read image
    imagem = read_image(file)
    # transform and prediction 
    prediction = transformacao(imagem)

    return prediction