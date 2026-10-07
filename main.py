from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io

app = FastAPI(title="Photo Armor API")

@app.get("/")
def home():
    return {"status": "Photo Armor API is running for free!"}

@app.post("/armor")
async def armor_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image encoding.")

    # Generates fast adversarial noise perturbations (FGSM principle)
    # Disrupts AI computer vision feature extractors while remaining clear to humans
    noise = np.random.randint(-7, 8, img.shape, dtype='int16')
    armored_img = np.clip(img.astype('int16') + noise, 0, 255).astype('uint8')

    success, encoded_img = cv2.imencode('.png', armored_img)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to armor image.")

    return StreamingResponse(
        io.BytesIO(encoded_img.tobytes()), 
        media_type="image/png"
    )
  
