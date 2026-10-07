import sys
import os
import glob

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.ocr_service import get_ocr_service

def main():
    target_dir = r"I:\test\chenqian\DICOM\25061007\43140000\overlays"
    print(f"Scanning directory: {target_dir}")
    
    image_files = glob.glob(os.path.join(target_dir, "*.png"))
    print(f"Found {len(image_files)} images.")
    
    print("Initializing OCR Service...")
    ocr_service = get_ocr_service()
    
    for img_path in image_files:
        print(f"\nProcessing: {os.path.basename(img_path)}")
        try:
            result = ocr_service.extract_text_from_image(img_path, {'engine': 'deepseek'})
            print(f"Result: {result}")
            if result.get('text'):
                print(f"Extracted Text:\n{result['text']}")
            else:
                print("No text extracted.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
