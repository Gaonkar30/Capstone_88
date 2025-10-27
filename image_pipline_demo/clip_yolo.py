import os
import torch
import clip
from PIL import Image, ImageDraw
import numpy as np
import json
from typing import List, Dict, Tuple, Union
from ultralytics import YOLO

class YoloCLIPCaptioner:
    def __init__(self, model_name="ViT-B/32", yolo_model_path=None):
        """
        Initialize CLIP model for captioning detected objects and optionally YOLO model
        
        Args:
            model_name: CLIP model name
            yolo_model_path: Path to trained YOLO model weights
        """
        self.device = "cuda"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        self.yolo_model = None
        if yolo_model_path and os.path.exists(yolo_model_path):
            try:
                self.yolo_model = YOLO(yolo_model_path)
                print(f"YOLO model loaded from: {yolo_model_path}")
            except Exception as e:
                print(f"Warning: Could not load YOLO model: {e}")
        
        self.image_categories = [
            "product photo of illegal drugs",
            "weapon or firearm image", 
            "credit card or payment device",
            "identity document or passport",
            "computer equipment or hardware",
            "adult or explicit content",
            "cryptocurrency or bitcoin interface",
            "profile photo of person",
            "logo or website branding",
            "forum avatar or icon",
            "advertisement or promotional image",
            "screenshot or interface capture",
            "QR code or barcode",
            "text document or page",
            "generic object or item",
            "counterfeit money"
        ]
        
        self.encoded_categories = self._encode_categories()
    
    def _encode_categories(self):
        """
        Pre-encode all text categories for faster inference
        """
        print("Encoding image categories...")
        text_inputs = torch.cat([clip.tokenize(f"a photo of {category}") for category in self.image_categories])
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs.to(self.device))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu()
    
    def get_yolo_detections(self, image_path: str, conf_threshold: float = 0.8) -> List[List]:
        """
        Get YOLO detections from image (only high confidence detections)
        
        Args:
            image_path: Path to the image
            conf_threshold: Confidence threshold for detections (default: 0.8 for high confidence only)
            
        Returns:
            List of detections in format [x1, y1, x2, y2, confidence] with confidence >= 0.8
        """
        if self.yolo_model is None:
            print("Error: YOLO model not loaded")
            return []
        
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at '{image_path}'")
            return []
        
        try:
            print(f"Running YOLO inference on '{image_path}' (confidence >= {conf_threshold})...")
            results = self.yolo_model.predict(
                source=image_path,
                conf=conf_threshold,
                device=0
            )
            
            detections = []
            
            for r in results:
                if len(r.boxes) > 0:
                    for box in r.boxes:
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        x_min, y_min, x_max, y_max = coords
                        
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        if confidence >= conf_threshold:
                            detections.append([x_min, y_min, x_max, y_max, confidence])
            
            print(f"Found {len(detections)} high-confidence detections (confidence >= {conf_threshold})")
            return detections
            
        except Exception as e:
            print(f"Error during YOLO inference: {e}")
            return []
    
    def crop_detections(self, image_path: str, detections: List[List]) -> List[Dict]:
        """
        Crop detected regions from the main image
        
        Args:
            image_path: Path to the screenshot
            detections: List of [x1, y1, x2, y2, confidence] detections
            
        Returns:
            List of cropped images with metadata
        """
        try:
            
            main_image = Image.open(image_path).convert('RGB')
            img_width, img_height = main_image.size
            
            cropped_regions = []
            
            for i, detection in enumerate(detections):
                if len(detection) != 5:
                    print(f"Warning: Detection {i} has incorrect format. Expected [x1,y1,x2,y2,confidence], got {detection}")
                    continue
                
                x1, y1, x2, y2, confidence = detection
                
                x1 = max(0, min(int(x1), img_width))
                y1 = max(0, min(int(y1), img_height))
                x2 = max(0, min(int(x2), img_width))
                y2 = max(0, min(int(y2), img_height))
                
                
                if x2 <= x1 or y2 <= y1:
                    print(f"Warning: Invalid bounding box {[x1,y1,x2,y2]}")
                    continue
                
                cropped_image = main_image.crop((x1, y1, x2, y2))
                
                if cropped_image.size[0] < 10 or cropped_image.size[1] < 10:
                    print(f"Warning: Cropped image too small: {cropped_image.size}")
                    continue
                
                cropped_regions.append({
                    'detection_id': i,
                    'image': cropped_image,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'size': cropped_image.size
                })
            
            print(f"Successfully cropped {len(cropped_regions)} regions from {len(detections)} detections")
            return cropped_regions
            
        except Exception as e:
            print(f"Error cropping detections: {e}")
            return []
    
    def caption_cropped_images(self, cropped_regions: List[Dict]) -> List[Dict]:
        """
        Use CLIP to caption each cropped image region
        Always returns the highest confidence caption regardless of threshold
        """
        if not cropped_regions:
            return []
        
        print(f"Captioning {len(cropped_regions)} cropped regions...")
        
        results = []
        
        for region in cropped_regions:
            try:
                image_input = self.preprocess(region['image']).unsqueeze(0)
                
                with torch.no_grad():
                    image_features = self.model.encode_image(image_input.to(self.device))
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                similarities = (image_features.cpu() @ self.encoded_categories.T) * 100
                
                all_indices = similarities[0].argsort(descending=True)
                all_scores = similarities[0][all_indices]
                
                predictions = []
                for idx, score in zip(all_indices, all_scores):
                    predictions.append({
                        'category': self.image_categories[idx],
                        'confidence': float(score),
                        'confidence_normalized': float(score / 100.0)
                    })
                
                best_prediction = predictions[0]
                
                result = {
                    'detection_id': region['detection_id'],
                    'bbox': region['bbox'],
                    'yolo_confidence': region['confidence'],
                    'image_size': region['size'],
                    'best_prediction': best_prediction,
                    'top_3_predictions': predictions[:3],
                    'all_predictions': predictions,
                    'caption': f"{best_prediction['category']} (confidence: {best_prediction['confidence']:.1f}%)"
                }
                
                results.append(result)
                print(f"  Detection {region['detection_id']}: {result['caption']}")
                
            except Exception as e:
                print(f"Error captioning region {region['detection_id']}: {e}")
                
        return results
    
    def process_image_with_detections(self, image_path: str, detections: List[List]) -> Dict:
        """
        Process image with provided YOLO detections and return CLIP captions
        
        Args:
            image_path: Path to screenshot
            detections: List of [x1, y1, x2, y2, confidence] from YOLO
            
        Returns:
            Structured output with captions for each detection
        """
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        if not detections:
            return {"error": "No detections provided"}
        
        print(f"Processing {image_path} with {len(detections)} detections...")
        
        cropped_regions = self.crop_detections(image_path, detections)
        
        if not cropped_regions:
            return {"error": "No valid regions could be cropped"}
        
        captioned_results = self.caption_cropped_images(cropped_regions)
        
        output = {
            "image_path": image_path,
            "total_detections": len(detections),
            "successful_captions": len(captioned_results),
            "results": captioned_results,
            "summary": {
                "detected_categories": [],
                "high_confidence_detections": []
            }
        }
        
        categories = []
        high_conf_detections = []
        
        for result in captioned_results:
            category = result['best_prediction']['category']
            categories.append(category)
            
            high_conf_detections.append({
                'detection_id': result['detection_id'],
                'category': category,
                'yolo_confidence': result['yolo_confidence'],
                'clip_confidence': result['best_prediction']['confidence'],
                'best_caption': result['caption']
            })
        
        output['summary']['detected_categories'] = list(set(categories))
        output['summary']['all_detections'] = high_conf_detections
        
        return output
    
    def process_image_end_to_end(self, image_path: str, conf_threshold: float = 0.8) -> Dict:
        """
        Complete end-to-end processing: YOLO detection + CLIP captioning
        Only processes detections with confidence >= 0.8
        
        Args:
            image_path: Path to image
            conf_threshold: YOLO confidence threshold (default: 0.8 for high confidence only)
            
        Returns:
            Structured output with captions for each high-confidence detection
        """
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        detections = self.get_yolo_detections(image_path, conf_threshold)
        
        if not detections:
            return {
                "error": f"No high-confidence objects detected (confidence >= {conf_threshold})", 
                "image_path": image_path,
                "yolo_threshold_used": conf_threshold
            }
        
        return self.process_image_with_detections(image_path, detections)
    
    def save_visual_results(self, image_path: str, results: Dict, output_path: str = None):
        """
        Save image with bounding boxes and captions for visualization
        """
        if 'error' in results:
            print(f"Cannot create visualization: {results['error']}")
            return
        
        try:
            image = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(image)
            
            for result in results['results']:
                x1, y1, x2, y2 = result['bbox']
                category = result['best_prediction']['category']
                confidence = result['best_prediction']['confidence']
                
                draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                
                label = f"{category[:30]}... ({confidence:.1f}%)"
                label_bbox = draw.textbbox((x1, y1-20), label)
                draw.rectangle(label_bbox, fill='red', outline='red')
                
                draw.text((x1, y1-20), label, fill='white')
            
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = f"{base_name}_yolo_clip_results.png"
            
            image.save(output_path)
            print(f"Visualization saved to: {output_path}")
            
        except Exception as e:
            print(f"Error creating visualization: {e}")

def process_image_complete(image_path: str, yolo_model_path: str, 
                          conf_threshold: float = 0.8,
                          save_visualization: bool = True) -> Dict:
    """
    Complete pipeline: Load YOLO model, detect objects, caption with CLIP
    Only processes high-confidence YOLO detections (>= 0.8)
    
    Args:
        image_path: Path to image
        yolo_model_path: Path to trained YOLO weights
        conf_threshold: YOLO confidence threshold (default: 0.8 for high confidence only)
        save_visualization: Whether to save annotated image
        
    Returns:
        Dictionary with structured results
    """
    captioner = YoloCLIPCaptioner(yolo_model_path=yolo_model_path)
    
    results = captioner.process_image_end_to_end(image_path, conf_threshold)
    
    if save_visualization and 'error' not in results:
        captioner.save_visual_results(image_path, results)
    
    return results