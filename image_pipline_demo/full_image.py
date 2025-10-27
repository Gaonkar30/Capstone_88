import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import glob
import asyncio
from image_pipline_demo.clip_yolo import YoloCLIPCaptioner
# from clip_yolo import YoloCLIPCaptioner
from image_pipline_demo.tor_ss_capture import TorScreenshotCapture
# from tor_ss_capture import TorScreenshotCapture

class TorWebsiteAnalyzer:
    """
    Complete pipeline for analyzing Tor websites:
    1. Take screenshots
    2. Run YOLO object detection
    3. Caption objects with CLIP
    4. Generate analysis reports
    """
    
    def __init__(self,
                 yolo_model_path: str,
                 output_base_dir: str = "tor_analysis",
                 proxy_server: str = "socks5://127.0.0.1:9150"):
        """
        Initialize the complete Tor analysis pipeline
        
        Args:
            yolo_model_path: Path to trained YOLO model
            output_base_dir: Base directory for all outputs
            proxy_server: Tor proxy server
        """
        self.yolo_model_path = yolo_model_path
        self.output_base_dir = Path(output_base_dir)
        self.proxy_server = proxy_server
        
        
        self.output_base_dir.mkdir(exist_ok=True)
        self.screenshots_dir = self.output_base_dir / "screenshots"
        self.analysis_dir = self.output_base_dir / "analysis"
        self.reports_dir = self.output_base_dir / "reports"
        
        for dir_path in [self.screenshots_dir, self.analysis_dir, self.reports_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.captioner = YoloCLIPCaptioner(yolo_model_path=yolo_model_path)
        
        self.screenshot_capturer = TorScreenshotCapture(
            proxy_server=proxy_server,
            output_dir=str(self.screenshots_dir)
        )
    
    async def analyze_website(self, 
                            url: str,
                            max_screenshots: int = 15,
                            yolo_confidence: float = 0.5,
                            scroll_strategy: str = "adaptive") -> Dict:
        """
        Complete analysis of a Tor website
        
        Args:
            url: Tor URL to analyze
            max_screenshots: Maximum screenshots to take
            yolo_confidence: YOLO detection confidence threshold
            scroll_strategy: Screenshot scroll strategy
            
        Returns:
            Complete analysis results
        """
        print(f"\n{'='*60}")
        print(f"ANALYZING TOR WEBSITE: {url}")
        print(f"{'='*60}")
        
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        analysis_output_dir = self.analysis_dir / analysis_id
        analysis_output_dir.mkdir(exist_ok=True)
        
        
        print("\nCapturing screenshots...")
        screenshot_result = await self.screenshot_capturer.capture_single_page(
            url=url,
            max_screenshots=max_screenshots,
            custom_scroll_strategy=scroll_strategy
        )
        
        if not screenshot_result['success']:
            return {
                'success': False,
                'error': f"Failed to capture screenshots: {screenshot_result['error']}",
                'analysis_id': analysis_id
            }
        
        screenshot_paths = [shot['path'] for shot in screenshot_result['screenshots']]
        print(f"Captured {len(screenshot_paths)} screenshots")
        
        print(f"\nSTEP 2: Running YOLO+CLIP analysis on {len(screenshot_paths)} images...")
        print(f"   Using YOLO confidence threshold: >= {yolo_confidence}")
        print(f"   CLIP output: Always highest confidence category")
        
        analysis_results = []
        successful_analyses = 0
        total_detections = 0
        all_categories = set()
        all_detections = []
        
        for i, screenshot_path in enumerate(screenshot_paths):
            try:
                print(f"\n  Analyzing image {i+1}/{len(screenshot_paths)}: {Path(screenshot_path).name}")
                
                result = self.captioner.process_image_end_to_end(
                    image_path=screenshot_path,
                    conf_threshold=yolo_confidence
                )
                
                if 'error' not in result:
                    successful_analyses += 1
                    total_detections += result['successful_captions']
                    
                    for category in result['summary']['detected_categories']:
                        all_categories.add(category)
                    
                    for detection in result['summary']['all_detections']:
                        all_detections.append({
                            'screenshot': Path(screenshot_path).name,
                            'detection': detection,
                            'screenshot_path': screenshot_path
                        })
                    
                    self.captioner.save_visual_results(
                        screenshot_path, 
                        result, 
                        str(analysis_output_dir / f"analyzed_{Path(screenshot_path).name}")
                    )
                    
                    print(f"    Found {result['successful_captions']} high-confidence objects: {', '.join(result['summary']['detected_categories'])}")
                else:
                    print(f"    Analysis failed: {result['error']}")
                
                analysis_results.append({
                    'screenshot_path': screenshot_path,
                    'screenshot_name': Path(screenshot_path).name,
                    'analysis_result': result
                })
                
            except Exception as e:
                print(f"    Error analyzing {screenshot_path}: {e}")
                analysis_results.append({
                    'screenshot_path': screenshot_path,
                    'screenshot_name': Path(screenshot_path).name,
                    'error': str(e)
                })
        
        print(f"\nSTEP 3: Generating analysis report...")
        
        report = {
            'analysis_id': analysis_id,
            'timestamp': datetime.now().isoformat(),
            'url': url,
            'configuration': {
                'max_screenshots': max_screenshots,
                'yolo_confidence': yolo_confidence,
                'scroll_strategy': scroll_strategy,
                'yolo_model_path': self.yolo_model_path
            },
            'summary': {
                'total_screenshots': len(screenshot_paths),
                'successful_analyses': successful_analyses,
                'total_objects_detected': total_detections,
                'unique_categories_found': len(all_categories),
                'categories_list': list(all_categories),
                'total_detections_found': len(all_detections)
            },
            'detailed_results': analysis_results,
            'all_detections': all_detections,
            'screenshot_metadata': screenshot_result
        }
        
        report_path = self.reports_dir / f"{analysis_id}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        summary_path = self.reports_dir / f"{analysis_id}_summary.txt"
        self._generate_text_summary(report, summary_path)
        
        print(f"\nANALYSIS COMPLETE!")
        print(f"Results: {successful_analyses}/{len(screenshot_paths)} screenshots analyzed")
        print(f"Objects found: {total_detections} total high-confidence detections")
        print(f"Categories detected: {', '.join(list(all_categories)) if all_categories else 'None'}")
        print(f"All detections: {len(all_detections)} (YOLO >= {yolo_confidence}, CLIP = highest confidence)")
        print(f"Reports saved to: {self.reports_dir}")
        print(f"Analyzed images saved to: {analysis_output_dir}")
        
        return {
            'success': True,
            'analysis_id': analysis_id,
            'report': report,
            'report_path': str(report_path),
            'summary_path': str(summary_path),
            'analyzed_images_dir': str(analysis_output_dir)
        }
    
    def _generate_text_summary(self, report: Dict, output_path: Path):
        """Generate human-readable text summary"""
        with open(output_path, 'w') as f:
            f.write("TOR WEBSITE ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis ID: {report['analysis_id']}\n")
            f.write(f"Timestamp: {report['timestamp']}\n")
            f.write(f"URL Analyzed: {report['url']}\n")
            f.write(f"YOLO Model: {report['configuration']['yolo_model_path']}\n\n")
            
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 20 + "\n")
            summary = report['summary']
            f.write(f"Screenshots captured: {summary['total_screenshots']}\n")
            f.write(f"Successfully analyzed: {summary['successful_analyses']}\n")
            f.write(f"Total objects detected: {summary['total_objects_detected']}\n")
            f.write(f"Unique categories found: {summary['unique_categories_found']}\n")
            f.write(f"Total detections: {summary['total_detections_found']}\n\n")
            
            if summary['categories_list']:
                f.write("DETECTED CATEGORIES:\n")
                f.write("-" * 20 + "\n")
                for category in sorted(summary['categories_list']):
                    f.write(f"• {category}\n")
                f.write("\n")
            
            if report['all_detections']:
                f.write("ALL DETECTIONS (HIGH CONFIDENCE):\n")
                f.write("-" * 35 + "\n")
                for finding in report['all_detections']:
                    f.write(f"Screenshot: {finding['screenshot']}\n")
                    f.write(f"Category: {finding['detection']['category']}\n")
                    f.write(f"YOLO Confidence: {finding['detection']['yolo_confidence']:.3f}\n")
                    f.write(f"CLIP Confidence: {finding['detection']['clip_confidence']:.1f}%\n")
                    f.write(f"Caption: {finding['detection']['best_caption']}\n")
                    f.write("-" * 10 + "\n")


if __name__ == "__main__":
    YOLO_MODEL_PATH = r'D:\capstone\code\yolo\training\runs\detect\train2\weights\best.pt'
    TOR_URL = "http://wbz2lrxhw4dd7h5t2wnoczmcz5snjpym4pr7dzjmah4vi6yywn37bdyd.onion/"
    
    async def main():
        print("Starting Tor Website Analysis...")
        
        if not os.path.exists(YOLO_MODEL_PATH):
            print(f"YOLO model not found at: {YOLO_MODEL_PATH}")
            print("Please update the YOLO_MODEL_PATH variable with the correct path.")
            return
        
        analyzer = TorWebsiteAnalyzer(
            yolo_model_path=YOLO_MODEL_PATH,
            output_base_dir="tor_website_analysis"
        )
        
        result = await analyzer.analyze_website(
            url=TOR_URL,
            max_screenshots=10,
            yolo_confidence=0.65,
            scroll_strategy="adaptive"
        )
        
        if result['success']:
            print(f"\nAnalysis completed successfully!")
            print(f"Report: {result['report_path']}")
            print(f"Summary: {result['summary_path']}")
            print(f"Analyzed images: {result['analyzed_images_dir']}")
        else:
            print(f"Analysis failed: {result['error']}")
    
if __name__ == "__main__":
    asyncio.run(main())
