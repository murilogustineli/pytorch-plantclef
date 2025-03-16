"""Module for encoding and decoding data structures to and from raw bytes"""

from PIL import Image
import io


def deserialize_image(buffer: bytes | bytearray) -> Image.Image:
    """Decode the image from raw bytes using PIL."""
    buffer = io.BytesIO(bytes(buffer))
    return Image.open(buffer).convert("RGB")


def serialize_image(image: Image.Image) -> bytes:
    """Encode the image as raw bytes using PIL."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()
