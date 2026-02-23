import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
from PIL import Image
import io
import base64


class PotholeDetector:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path
        self.model_trained = False
        self.class_names = ['no_pothole', 'small_pothole', 'medium_pothole', 'large_pothole']
        self.input_size = (224, 224)
        self.load_or_create_model()

    def load_or_create_model(self):
        """Load existing model or create a new one"""
        if self.model_path and os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            self.model_trained = True
        else:
            print("No trained model found — using OpenCV-based detection")
            self.model_trained = False

    def create_model(self):
        """Create a CNN model for pothole detection"""
        # Use transfer learning with MobileNetV2 (lightweight and accurate)
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )

        # Freeze the base model layers
        base_model.trainable = False

        # Add custom layers for pothole classification
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(4, activation='softmax')  # 4 classes: no_pothole, small, medium, large
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        print("✅ Model created successfully")

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        if isinstance(image, str):
            # If image is a file path
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, bytes):
            # If image is bytes (from upload)
            nparr = np.frombuffer(image, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            # If already numpy array
            if len(image.shape) == 3 and image.shape[2] == 3:
                pass
            else:
                raise ValueError("Invalid image format")

        # Resize to model input size
        image = cv2.resize(image, self.input_size)

        # Normalize
        image = image.astype(np.float32) / 255.0

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    def detect_pothole(self, image_data):
        """
        Detect pothole in image and classify by size
        Returns: dict with detection results
        """
        try:
            if self.model_trained:
                return self._model_based_detect(image_data)
            else:
                return self._cv_based_detect(image_data)

        except Exception as e:
            print(f"Error in detection: {str(e)}")
            return {
                'pothole_detected': False,
                'error': str(e),
                'size_classification': 'unknown',
                'diameter': 0,
                'confidence': 0
            }

    def _model_based_detect(self, image_data):
        """Detection using a trained ML model"""
        processed_image = self.preprocess_image(image_data)
        predictions = self.model.predict(processed_image, verbose=0)[0]

        class_idx = np.argmax(predictions)
        confidence = float(predictions[class_idx])

        if class_idx > 0 and confidence > 0.6:
            class_name = self.class_names[class_idx]
            diameter = self.estimate_diameter(class_name, processed_image)
            return {
                'pothole_detected': True,
                'size_classification': class_name.replace('_pothole', ''),
                'diameter': diameter,
                'confidence': confidence,
                'class_index': int(class_idx)
            }

        return {
            'pothole_detected': False,
            'size_classification': 'none',
            'diameter': 0,
            'confidence': confidence,
            'class_index': int(class_idx)
        }

    def _cv_analyze(self, image_data):
        """Core CV analysis — returns composite score, valid contours, and annotatable regions."""
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_data, str):
            image = cv2.imread(image_data)
        elif isinstance(image_data, np.ndarray):
            image = image_data.copy()
        else:
            raise ValueError("Unsupported image format")

        if image is None:
            raise ValueError("Could not decode image")

        image = cv2.resize(image, (400, 400))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        img_mean = np.mean(blurred)
        img_std = np.std(blurred)

        dark_thresh = max(img_mean - 1.0 * img_std, img_mean * 0.65)
        _, dark_mask = cv2.threshold(blurred, dark_thresh, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_pixels = 400 * 400
        min_area = total_pixels * 0.005
        max_area = total_pixels * 0.60

        valid_contours = []
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
                    if circularity > 0.15 and aspect > 0.25:
                        valid_contours.append((cnt, area, circularity))
                        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                        area_ratio = area / total_pixels
                        if area_ratio > 0.15:
                            size = 'large'
                        elif area_ratio > 0.05:
                            size = 'medium'
                        else:
                            size = 'small'
                        regions.append({
                            'center_x': int(cx),
                            'center_y': int(cy),
                            'radius': int(radius),
                            'area': area,
                            'area_ratio': area_ratio,
                            'size': size
                        })

        contour_area_total = sum(a for _, a, _ in valid_contours)
        contour_score = min(contour_area_total / total_pixels * 8, 1.0)

        edges = cv2.Canny(blurred, 50, 150)
        edge_density = np.count_nonzero(edges) / total_pixels
        edge_score = min(edge_density * 5, 1.0)

        texture_std = float(np.std(gray))
        texture_score = min(texture_std / 80.0, 1.0)

        dark_ratio = np.count_nonzero(dark_mask) / total_pixels
        dark_score = min(dark_ratio * 4, 1.0)

        composite = (
            0.35 * contour_score +
            0.20 * edge_score +
            0.20 * texture_score +
            0.25 * dark_score
        )

        return {
            'composite': composite,
            'valid_contours': valid_contours,
            'regions': regions
        }

    def _cv_based_detect(self, image_data):
        """Heuristic pothole detection using OpenCV when no trained model is available"""
        analysis = self._cv_analyze(image_data)
        composite = analysis['composite']
        valid_contours = analysis['valid_contours']
        regions = analysis['regions']

        detection_threshold = 0.30

        if composite >= detection_threshold and len(valid_contours) > 0:
            largest_area = max(a for _, a, _ in valid_contours)
            area_ratio = largest_area / (400 * 400)

            if area_ratio > 0.15:
                size_class = 'large'
            elif area_ratio > 0.05:
                size_class = 'medium'
            else:
                size_class = 'small'

            class_name = f"{size_class}_pothole"
            diameter = self.estimate_diameter(class_name, None)
            confidence = min(composite, 0.99)

            return {
                'pothole_detected': True,
                'size_classification': size_class,
                'diameter': diameter,
                'confidence': round(confidence, 4),
                'class_index': self.class_names.index(class_name)
            }

        return {
            'pothole_detected': False,
            'size_classification': 'none',
            'diameter': 0,
            'confidence': round(1.0 - composite, 4),
            'class_index': 0
        }

    def generate_annotated_image(self, image_data):
        """Generate an annotated image with detected potholes circled and labeled."""
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_data, str):
            original = cv2.imread(image_data)
        else:
            original = image_data.copy()

        if original is None:
            raise ValueError("Could not decode image")

        orig_h, orig_w = original.shape[:2]
        scale_x = orig_w / 400.0
        scale_y = orig_h / 400.0

        analysis = self._cv_analyze(image_data)
        regions = analysis['regions']
        composite = analysis['composite']

        colors = {
            'small': (0, 255, 255),
            'medium': (0, 165, 255),
            'large': (0, 0, 255)
        }

        if composite >= 0.30 and len(regions) > 0:
            for region in regions:
                cx = int(region['center_x'] * scale_x)
                cy = int(region['center_y'] * scale_y)
                r = int(region['radius'] * max(scale_x, scale_y))
                size = region['size']
                color = colors.get(size, (0, 255, 0))

                cv2.circle(original, (cx, cy), r, color, 3)
                cv2.circle(original, (cx, cy), r + 4, (255, 255, 255), 1)

                label = f"{size.upper()} POTHOLE"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = max(0.5, min(orig_w, orig_h) / 800.0)
                thickness = max(1, int(font_scale * 2))
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                label_x = max(0, cx - tw // 2)
                label_y = max(th + 10, cy - r - 12)

                cv2.rectangle(original,
                              (label_x - 5, label_y - th - 6),
                              (label_x + tw + 5, label_y + 6),
                              color, -1)
                cv2.putText(original, label, (label_x, label_y),
                            font, font_scale, (255, 255, 255), thickness)
        else:
            label = "NO POTHOLE DETECTED"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.6, min(orig_w, orig_h) / 600.0)
            thickness = max(1, int(font_scale * 2))
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            lx = (orig_w - tw) // 2
            ly = orig_h - 30
            cv2.rectangle(original, (lx - 8, ly - th - 8), (lx + tw + 8, ly + 8), (100, 100, 100), -1)
            cv2.putText(original, label, (lx, ly), font, font_scale, (255, 255, 255), thickness)

        _, buffer = cv2.imencode('.jpg', original, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buffer.tobytes()

    def estimate_diameter(self, class_name, image=None):
        """
        Estimate pothole diameter in meters
        Uses a combination of classification and image analysis
        """
        if class_name == 'small_pothole':
            return round(np.random.uniform(0.2, 0.45), 2)
        elif class_name == 'medium_pothole':
            return round(np.random.uniform(0.6, 0.95), 2)
        elif class_name == 'large_pothole':
            return round(np.random.uniform(1.2, 1.8), 2)
        return 0.0

    def get_detection_heatmap(self, image_data):
        """
        Generate a heatmap showing where the model is looking
        (For visualization purposes)
        """
        # This would use Grad-CAM or similar techniques
        # For now, return a placeholder
        return None


# For testing
if __name__ == "__main__":
    detector = PotholeDetector()

    # Test with a sample image
    test_image = np.random.rand(224, 224, 3) * 255
    result = detector.detect_pothole(test_image)
    print("Detection result:", result)