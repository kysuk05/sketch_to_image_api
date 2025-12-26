import os
import tempfile
from fastapi import UploadFile
from utils.image_upload import upload_to_catbox

class ImageService:
    @staticmethod   
    async def save_temp_file(upload_file: UploadFile) -> str:
        suffix = os.path.splitext(upload_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await upload_file.read())
            return tmp.name

class ImageUploader:
    @staticmethod
    async def upload_to_catbox(file_path: str) -> str:
        return upload_to_catbox(file_path)
