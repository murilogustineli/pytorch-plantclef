import io
import pytest
import pandas as pd
from PIL import Image


@pytest.fixture
def pandas_df():
    # generate a small dummy image(RGB, 32X32) for testing
    img = Image.new("RGB", (32, 32), color="blue")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()
    data = {"data": [img_bytes, img_bytes]}

    return pd.DataFrame(data)
