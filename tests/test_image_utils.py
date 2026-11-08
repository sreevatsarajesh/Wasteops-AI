from io import BytesIO
import pytest
from PIL import Image
from wasteops.image_utils import decode_image

def test_decode_jpeg():
    b=BytesIO();Image.new('RGB',(20,20),'red').save(b,format='JPEG')
    assert decode_image(b.getvalue()).mode=='RGB'

def test_reject_empty():
    with pytest.raises(ValueError):decode_image(b'')

def test_reject_non_image():
    with pytest.raises(ValueError):decode_image(b'not an image')

def test_reject_svg():
    with pytest.raises(ValueError):decode_image(b'<svg></svg>')
