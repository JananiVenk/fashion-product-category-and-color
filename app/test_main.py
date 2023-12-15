import sys
from fastapi.testclient import TestClient
sys.path.append('/home/ubuntu/janani/app')
from app.main import app
client=TestClient(app)

def test_process_image():
    path=os.path.join(os.path.dirname(__file__,'images/10006.jpg')
    with open(path, 'rb') as image_file:
        files = {"file": ('10006.jpg', image_file, "image/jpg")}
        response = client.post("http://127.0.0.1:8012/predict", files=files)
        assert response.status_code == 200
        data=response.json()
        assert data['category']=='Apparel'
        assert data['color']=='Black'
