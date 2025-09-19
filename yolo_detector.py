import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

class YOLODetector:
    def __init__(self, model_size='m', models_dir="models"):
        """
        Initialize YOLO detector with specified model size
        model_size options: n (nano), s (small), m (medium), l (large), x (xlarge)
        """
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        # Model path
        model_name = f"yolov8{model_size}.pt"
        model_path = os.path.join(models_dir, model_name)
        
        # Check if model exists in the models directory, if not, download to models directory
        if not os.path.exists(model_path):
            print(f"Model {model_name} not found in {models_dir}. Downloading and storing in models folder...")
            # Using YOLO to download the model will automatically store it in the ultralytics cache,
            # so we need to copy it to our models directory
            temp_model = YOLO(model_name)
            # Copy from cache to models directory
            if hasattr(temp_model, 'ckpt_path') and os.path.exists(temp_model.ckpt_path):
                import shutil
                shutil.copy(temp_model.ckpt_path, model_path)
                print(f"Model copied to {model_path}")
            else:
                # If we can't access the cache path, save the model directly
                temp_model.save(model_path)
                print(f"Model saved to {model_path}")
                
        # Load the model from models directory
        self.model = YOLO(model_path)
        print(f"Loaded YOLO model from {model_path}")
        
    def detect(self, image_path, conf_threshold=0.25):
        """
        Detect objects in an image
        """
        # Run inference on the image
        results = self.model(image_path, conf=conf_threshold)
        
        # Get the first result (only one image was passed)
        result = results[0]
        
        # Load the original image to draw on
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process results
        detected_objects = []
        
        if result.boxes is not None:
            boxes = result.boxes.cpu().numpy()
            
            for i, box in enumerate(boxes):
                # Get box coordinates, confidence and class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                
                # Only process detections above threshold
                if conf >= conf_threshold:
                    # Draw bounding box
                    color = (0, 255, 0)  # Green color
                    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with class name and confidence
                    label = f"{cls_name}: {conf:.2f}"
                    cv2.putText(image_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Add to detected objects list
                    detected_objects.append({
                        'class': cls_name,
                        'confidence': conf,
                        'box': (x1, y1, x2, y2)
                    })
        
        # Convert back to BGR for saving with OpenCV
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        return image_bgr, detected_objects
    
    def detect_and_save(self, image_path, output_dir="data/output", conf_threshold=0.25):
        """
        Detect objects in an image and save the result
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Process the image
        result_image, detected_objects = self.detect(image_path, conf_threshold)
        
        # Save the result
        image_name = Path(image_path).stem
        output_path = f"{output_dir}/{image_name}_yolo.jpg"
        cv2.imwrite(output_path, result_image)
        
        # Print detection results
        print(f"YOLO detected {len(detected_objects)} objects in {image_path}:")
        for obj in detected_objects:
            print(f"  - {obj['class']} with confidence {obj['confidence']:.2f}")
        
        print(f"Results saved to {output_path}")
        return output_path, detected_objects
    
    def process_directory(self, input_dir="data/input", output_dir="data/output", conf_threshold=0.25):
        """
        Process all images in a directory
        """
        processed_files = []
        
        for img_file in os.listdir(input_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(input_dir, img_file)
                print(f"Processing {img_path}...")
                output_path, _ = self.detect_and_save(img_path, output_dir, conf_threshold)
                processed_files.append(output_path)
        
        return processed_files

if __name__ == "__main__":
    # Initialize the YOLO detector with the 'n' (nano) model
    # You can change to 's', 'm', 'l', or 'x' for larger models with better accuracy but slower speed
    detector = YOLODetector(model_size='m')
    
    # Process all images in the input directory
    detector.process_directory(conf_threshold=0.3)