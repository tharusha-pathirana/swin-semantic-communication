import subprocess
from datetime import datetime

def test_libcamera_still(output_dir="./"):
    try:
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"{output_dir}image_{timestamp}.png"

        # Build and run the libcamera-still command
        command = ['libcamera-still', '--encoding', 'png', '-o', image_path]
        print("Capturing image using libcamera-still...")
        subprocess.run(command, check=True)

        print(f"Image captured and saved at: {image_path}")
        return image_path

    except subprocess.CalledProcessError as e:
        print("Failed to capture image with libcamera-still.")
        print(e)
        return None

if __name__ == "__main__":
    test_libcamera_still()
