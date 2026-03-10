import asyncio
import io
import numpy as np
from PIL import Image

# Import the actual function
from app.routers.predict import _validate_fundus_image

def test():
    print("Testing with a random bright image (like a normal photo)...")
    # create a random image shape 384x384 mostly bright
    img_arr = np.random.randint(100, 255, (384, 384, 3), dtype=np.uint8)
    img = Image.fromarray(img_arr)
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    try:
        _validate_fundus_image(img_bytes)
        print("FAIL: The normal photo was NOT rejected!")
    except Exception as e:
        print(f"PASS: It was correctly rejected. Exception: {e.detail}")

    print("\nTesting with a mock fundus image (dark borders)...")
    # Create an image that has mostly dark borders
    fundus_arr = np.zeros((384, 384, 3), dtype=np.uint8)
    # central circle bright
    for r in range(384):
        for c in range(384):
            if (r-192)**2 + (c-192)**2 < 120**2:
                fundus_arr[r, c] = [150, 100, 100]
    
    fundus_img = Image.fromarray(fundus_arr)
    f_byte_arr = io.BytesIO()
    fundus_img.save(f_byte_arr, format='JPEG')
    f_bytes = f_byte_arr.getvalue()
    
    try:
        _validate_fundus_image(f_bytes)
        print("PASS: The mock fundus image was accepted.")
    except Exception as e:
        print(f"FAIL: The mock fundus image was rejected. Exception: {e}")

if __name__ == "__main__":
    test()
