from io import BytesIO
from fastapi.testclient import TestClient
from PIL import Image
from wasteops.api import app
from wasteops.inference import service
from wasteops.models import Prediction

class FakeService:
    ready=True;backend='test'
    def predict(self,image):return Prediction('plastic',0.8,[{'label':'plastic','name':'Plastic','score':0.8}], 'test')

def test_health_and_classes():
    with TestClient(app) as client:
        assert client.get('/api/health').status_code==200
        assert len(client.get('/api/classes').json()['classes'])==9

def test_analyze(monkeypatch):
    monkeypatch.setattr('wasteops.api.service',FakeService())
    b=BytesIO();Image.new('RGB',(20,20),'blue').save(b,format='JPEG')
    with TestClient(app) as client:
        r=client.post('/api/analyze',files={'file':('test.jpg',b.getvalue(),'image/jpeg')})
    assert r.status_code==200
    assert r.json()['label']=='plastic'
    assert r.json()['score_is_calibrated_probability'] is False

def test_reject_wrong_mime():
    with TestClient(app) as client:
        r=client.post('/api/analyze',files={'file':('x.txt',b'abc','text/plain')})
    assert r.status_code==415
