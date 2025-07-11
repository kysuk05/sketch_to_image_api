import requests

def upload_to_catbox(image_path):
    with open(image_path, 'rb') as f:
        resp = requests.post(
            "https://catbox.moe/user/api.php",
            data={'reqtype': 'fileupload'},
            files={'fileToUpload': f}
        )
    return resp.text.strip()