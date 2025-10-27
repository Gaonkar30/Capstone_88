import asyncio
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import os
from image_pipline_demo.full_image import TorWebsiteAnalyzer
from text_classification_bert.classify import ImprovedTextClassifier, predict_with_top_k
from Text_Extraction.image_to_text import OCRProcessor


class CombinedWebsiteClassifier:
    """
    Complete pipeline combining image-based and text-based classification
    """
    
    def __init__(self,
                 yolo_model_path: str,
                 text_model_path: str = './trained_model',
                 label_mapping_path: str = 'label_mapping.json',
                 image_weight: float = 0.6,
                 text_weight: float = 0.4,
                 output_dir: str = "combined_analysis"):
        """
        Initialize the combined classifier
        
        Args:
            yolo_model_path: Path to YOLO model for image analysis
            text_model_path: Path to trained text classification model
            label_mapping_path: Path to label mapping JSON
            image_weight: Weight for image predictions (default: 0.6)
            text_weight: Weight for text predictions (default: 0.4)
            output_dir: Directory for saving results
        """
        self.yolo_model_path = yolo_model_path
        self.text_model_path = text_model_path
        self.image_weight = image_weight
        self.text_weight = text_weight
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load label mapping
        with open(label_mapping_path, 'r') as f:
            self.label_mapping = json.load(f)
        
        self.categories = list(self.label_mapping.keys())
        print(f"Loaded {len(self.categories)} categories: {self.categories}")
        
        # Initialize image analyzer
        print("\nInitializing Image Analyzer (YOLO + CLIP)...")
        self.image_analyzer = TorWebsiteAnalyzer(
            yolo_model_path=yolo_model_path,
            output_base_dir=str(self.output_dir / "image_analysis")
        )
        
        # Initialize text classifier
        print("\nInitializing Text Classifier...")
        self.text_classifier = ImprovedTextClassifier(num_labels=len(self.label_mapping))
        self.text_classifier.load_model(text_model_path)
        
        # Initialize OCR processor
        print("\nInitializing OCR Processor...")
        self.ocr_processor = OCRProcessor()
        
        # Normalize weights
        total = image_weight + text_weight
        self.image_weight = image_weight / total
        self.text_weight = text_weight / total
        
        print(f"\nWeights: Image={self.image_weight:.2f}, Text={self.text_weight:.2f}")
        print("="*70)
    
    async def analyze_website(self, 
                             tor_url: str,
                             max_screenshots: int = 10,
                             yolo_confidence: float = 0.65) -> Dict:
        """
        Complete pipeline: Image analysis + Text analysis + Weighted voting
        
        Args:
            tor_url: Tor website URL to analyze
            max_screenshots: Number of screenshots to capture
            yolo_confidence: YOLO detection confidence threshold
            
        Returns:
            Complete analysis results with final prediction
        """
        print(f"\n{'='*70}")
        print(f"STARTING COMBINED ANALYSIS FOR: {tor_url}")
        print(f"{'='*70}")
        
        analysis_results = {
            'url': tor_url,
            'image_analysis': None,
            'text_analysis': None,
            'combined_prediction': None,
            'success': False
        }
        
        print("\n" + "="*70)
        print("STEP 1: IMAGE-BASED ANALYSIS (YOLO + CLIP)")
        print("="*70)
        
        try:
            image_result = await self.image_analyzer.analyze_website(
                url=tor_url,
                max_screenshots=max_screenshots,
                yolo_confidence=yolo_confidence
            )
            
            if image_result['success']:
                # Extract category from image analysis
                image_category, image_probs = self._extract_image_category(image_result)
                
                analysis_results['image_analysis'] = {
                    'category': image_category,
                    'probabilities': image_probs,
                    'confidence': max(image_probs.values()),
                    'report_path': image_result['report_path']
                }
                
                print(f"\nImage Analysis Result:")
                print(f"  Predicted Category: {image_category}")
                print(f"  Confidence: {max(image_probs.values()):.2%}")
                print(f"  All probabilities: {image_probs}")
            else:
                print(f"\nImage analysis failed: {image_result.get('error', 'Unknown error')}")
                analysis_results['image_analysis'] = {'error': image_result.get('error')}
        
        except Exception as e:
            print(f"\nâŒ Error in image analysis: {e}")
            analysis_results['image_analysis'] = {'error': str(e)}
        
        print("\n" + "="*70)
        print("STEP 2: TEXT-BASED ANALYSIS (OCR + Classification)")
        print("="*70)
        
        try:
            # Extract text from screenshots using OCR
            text_content = self._extract_text_from_screenshots(image_result)
            
            if text_content:
                print(f"\nExtracted {len(text_content)} characters of text")
                print(f"Preview: {text_content[:200]}...")
                
                # Classify text
                text_category, text_probs = self._classify_text(text_content)
                
                analysis_results['text_analysis'] = {
                    'category': text_category,
                    'probabilities': text_probs,
                    'confidence': max(text_probs.values()),
                    'extracted_text_length': len(text_content)
                }
                
                print(f"\nText Analysis Result:")
                print(f"  Predicted Category: {text_category}")
                print(f"  Confidence: {max(text_probs.values()):.2%}")
                print(f"  All probabilities: {text_probs}")
            else:
                print("\n No text could be extracted from screenshots")
                analysis_results['text_analysis'] = {'error': 'No text extracted'}
        
        except Exception as e:
            print(f"\nâŒ Error in text analysis: {e}")
            analysis_results['text_analysis'] = {'error': str(e)}
        
        print("\n" + "="*70)
        print("STEP 3: COMBINING PREDICTIONS (WEIGHTED VOTING)")
        print("="*70)
        
        image_success = analysis_results['image_analysis'] and 'category' in analysis_results['image_analysis']
        text_success = analysis_results['text_analysis'] and 'category' in analysis_results['text_analysis']
        
        if image_success and text_success:
            final_category, combined_probs, voting_details = self._weighted_voting(
                image_category=analysis_results['image_analysis']['category'],
                image_probs=analysis_results['image_analysis']['probabilities'],
                text_category=analysis_results['text_analysis']['category'],
                text_probs=analysis_results['text_analysis']['probabilities']
            )
            
            analysis_results['combined_prediction'] = {
                'final_category': final_category,
                'combined_probabilities': combined_probs,
                'confidence': max(combined_probs.values()),
                'voting_details': voting_details
            }
            analysis_results['success'] = True
            
            print(f"\nCombined Prediction:")
            print(f"  Final Category: {final_category}")
            print(f"  Confidence: {max(combined_probs.values()):.2%}")
            print(f"\n{voting_details}")
            
        elif image_success:
            # Only image analysis successful
            print("\nUsing only image-based prediction (text analysis failed)")
            analysis_results['combined_prediction'] = {
                'final_category': analysis_results['image_analysis']['category'],
                'combined_probabilities': analysis_results['image_analysis']['probabilities'],
                'confidence': analysis_results['image_analysis']['confidence'],
                'method': 'image_only'
            }
            analysis_results['success'] = True
            
        elif text_success:
            # Only text analysis successful
            print("\nâš ï¸ Using only text-based prediction (image analysis failed)")
            analysis_results['combined_prediction'] = {
                'final_category': analysis_results['text_analysis']['category'],
                'combined_probabilities': analysis_results['text_analysis']['probabilities'],
                'confidence': analysis_results['text_analysis']['confidence'],
                'method': 'text_only'
            }
            analysis_results['success'] = True
            
        else:
            print("\nâŒ Both analyses failed - cannot make prediction")
            analysis_results['success'] = False
        
        # Save results
        self._save_results(analysis_results)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        
        return analysis_results
    
    def _extract_image_category(self, image_result: Dict) -> Tuple[str, Dict[str, float]]:
        """
        Extract category prediction from YOLO+CLIP analysis
        
        For now, this uses the most frequent detected category.
        You can modify this to use more sophisticated aggregation.
        """
        report = image_result['report']
        detected_categories = report['summary']['categories_list']
        
        if not detected_categories:
            # No objects detected, return uniform distribution
            uniform_prob = 1.0 / len(self.categories)
            return self.categories[0], {cat: uniform_prob for cat in self.categories}
        
        # Count frequency of each category
        category_counts = {}
        for detection in report['all_detections']:
            cat = detection['detection']['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Convert to probabilities
        total = sum(category_counts.values())
        image_probs = {cat: 0.0 for cat in self.categories}
        
        for cat, count in category_counts.items():
            if cat in image_probs:
                image_probs[cat] = count / total
        
        # Get most frequent category
        image_category = max(category_counts, key=category_counts.get)
        
        return image_category, image_probs
    
    def _extract_text_from_screenshots(self, image_result: Dict) -> str:
        """
        Extract text from all screenshots using OCR
        """
        if not image_result.get('success'):
            return ""
        
        screenshot_paths = [
            shot['path'] 
            for shot in image_result['report']['screenshot_metadata']['screenshots']
        ]
        
        all_text = []
        print(f"\nExtracting text from {len(screenshot_paths)} screenshots...")
        
        for i, screenshot_path in enumerate(screenshot_paths):
            try:
                text = self.ocr_processor.extract_text(screenshot_path)
                if text:
                    all_text.append(text)
                    print(f"  Screenshot {i+1}: Extracted {len(text)} characters")
            except Exception as e:
                print(f"  Screenshot {i+1}: OCR failed - {e}")
        
        combined_text = "\n".join(all_text)
        return combined_text
    
    def _classify_text(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Classify extracted text using the trained text classifier
        """
        pred_label, confidence, all_probs = self.text_classifier.predict(
            text, 
            return_all_probs=True
        )
        
        # Convert to category name and probability dict
        reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        text_category = reverse_mapping[pred_label]
        
        text_probs = {
            reverse_mapping[i]: float(prob) 
            for i, prob in enumerate(all_probs)
        }
        
        return text_category, text_probs
    
    def _weighted_voting(self, 
                        image_category: str,
                        image_probs: Dict[str, float],
                        text_category: str,
                        text_probs: Dict[str, float]) -> Tuple[str, Dict[str, float], str]:
        """
        Combine predictions using weighted voting
        
        Returns:
            (final_category, combined_probabilities, voting_details_text)
        """
        # Combine probabilities
        combined_probs = {}
        for category in self.categories:
            img_prob = image_probs.get(category, 0.0)
            txt_prob = text_probs.get(category, 0.0)
            combined_probs[category] = (self.image_weight * img_prob + 
                                       self.text_weight * txt_prob)
        
        # Get final prediction
        final_category = max(combined_probs, key=combined_probs.get)
        
        # Generate voting details
        voting_details = f"""
Voting Details:
--------------
Image Prediction: {image_category} (weight: {self.image_weight:.2f})
Text Prediction:  {text_category} (weight: {self.text_weight:.2f})

Agreement: {'" YES' if image_category == text_category else 'NO'}

Combined Probabilities:
"""
        for cat in sorted(combined_probs.keys(), key=lambda x: combined_probs[x], reverse=True):
            img_p = image_probs.get(cat, 0.0)
            txt_p = text_probs.get(cat, 0.0)
            comb_p = combined_probs[cat]
            voting_details += f"  {cat:20s}: {comb_p:.2%} (img: {img_p:.2%}, txt: {txt_p:.2%})\n"
        
        return final_category, combined_probs, voting_details
    
    def _save_results(self, results: Dict):
        """Save analysis results to JSON"""
        output_path = self.output_dir / f"combined_analysis_{results['url'].split('/')[-1]}.json"
        
        # Make results JSON serializable
        serializable_results = json.loads(json.dumps(results, default=str))
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")

async def main():
    """
    Example usage of the combined pipeline
    """
    # Configuration
    YOLO_MODEL_PATH = r'D:\capstone\code\yolo\training\runs\detect\train2\weights\best.pt'
    TEXT_MODEL_PATH = './trained_model'
    LABEL_MAPPING_PATH = 'label_mapping.json'
    
    TOR_URL = "http://wbz2lrxhw4dd7h5t2wnoczmcz5snjpym4pr7dzjmah4vi6yywn37bdyd.onion/"
    
    # Verify paths
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"âŒ YOLO model not found at: {YOLO_MODEL_PATH}")
        return
    
    if not os.path.exists(TEXT_MODEL_PATH):
        print(f"âŒ Text model not found at: {TEXT_MODEL_PATH}")
        return
    
    if not os.path.exists(LABEL_MAPPING_PATH):
        print(f"âŒ Label mapping not found at: {LABEL_MAPPING_PATH}")
        return
    
    # Initialize combined classifier
    classifier = CombinedWebsiteClassifier(
        yolo_model_path=YOLO_MODEL_PATH,
        text_model_path=TEXT_MODEL_PATH,
        label_mapping_path=LABEL_MAPPING_PATH,
        image_weight=0.4,
        text_weight=0.6    
    )
    
    # Run analysis
    results = await classifier.analyze_website(
        tor_url=TOR_URL,
        max_screenshots=10,
        yolo_confidence=0.65
    )
    
    # Display final result
    if results['success']:
        print("\n" + "="*70)
        print("FINAL CLASSIFICATION RESULT")
        print("="*70)
        final = results['combined_prediction']
        print(f"Category: {final['final_category']}")
        print(f"Confidence: {final['confidence']:.2%}")
        print("="*70)
    else:
        print("\n❌ Classification failed")


if __name__ == "__main__":
    asyncio.run(main())