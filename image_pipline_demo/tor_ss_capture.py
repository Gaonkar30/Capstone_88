import asyncio
import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from playwright.async_api import async_playwright
from urllib.parse import urlparse

class TorScreenshotCapture:
    def __init__(self, 
                 proxy_server: str = "socks5://127.0.0.1:9150",
                 output_dir: str = "tor_screenshots",
                 scroll_height: int = 500,
                 screenshot_delay: int = 1,
                 page_timeout: int = 100000):
        """
        Initialize Tor screenshot capture utility
        
        Args:
            proxy_server: Tor proxy server (default: standard Tor browser proxy)
            output_dir: Directory to save screenshots
            scroll_height: Pixels to scroll between screenshots
            screenshot_delay: Delay between screenshots (seconds)
            page_timeout: Page load timeout (milliseconds)
        """
        self.proxy_server = proxy_server
        self.output_dir = Path(output_dir)
        self.scroll_height = scroll_height
        self.screenshot_delay = screenshot_delay
        self.page_timeout = page_timeout
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Track session info
        self.session_info = {
            "timestamp": datetime.now().isoformat(),
            "screenshots": [],
            "urls_processed": [],
            "errors": []
        }

    def _generate_filename(self, url: str, screenshot_count: int) -> str:
        """Generate screenshot filename based on URL and count"""
        parsed_url = urlparse(url)
        domain_part = parsed_url.netloc.split('.')[0][:15]  # First 15 chars of domain
        return f"{domain_part}_{screenshot_count:04d}.png"

    def _create_url_directory(self, url: str) -> Path:
        """Create subdirectory for specific URL"""
        parsed_url = urlparse(url)
        domain_part = parsed_url.netloc.split('.')[0][:20]
        url_dir = self.output_dir / f"{domain_part}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        url_dir.mkdir(exist_ok=True)
        return url_dir

    async def capture_single_page(self, 
                                url: str, 
                                max_screenshots: int = 50,
                                full_page_first: bool = True,
                                custom_scroll_strategy: str = "uniform") -> Dict:
        """
        Capture screenshots from a single Tor page
        
        Args:
            url: Tor URL to capture
            max_screenshots: Maximum number of screenshots to take
            full_page_first: Take a full-page screenshot first
            custom_scroll_strategy: 'uniform', 'adaptive', or 'manual'
            
        Returns:
            Dictionary with capture results and metadata
        """
        print(f"Starting capture for: {url}")
        
        # Create URL-specific directory
        url_dir = self._create_url_directory(url)
        
        proxy = {"server": self.proxy_server}
        screenshots_taken = []
        
        async with async_playwright() as p:
            try:
                # Launch browser with Tor proxy
                browser = await p.chromium.launch(
                    proxy=proxy, 
                    headless=True,
                    args=[
                        '--disable-web-security',
                        '--disable-features=VizDisplayCompositor',
                        '--no-sandbox'
                    ]
                )
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                )
                page = await context.new_page()
                
                # Navigate to page
                print(f"Loading page: {url}")
                await page.goto(url, timeout=self.page_timeout, wait_until='networkidle')
                
                # Wait for page to stabilize
                await asyncio.sleep(2)
                
                # Take full page screenshot first if requested
                if full_page_first:
                    full_page_path = url_dir / "full_page.png"
                    await page.screenshot(path=str(full_page_path), full_page=True)
                    screenshots_taken.append({
                        'path': str(full_page_path),
                        'type': 'full_page',
                        'scroll_position': 0
                    })
                    print(f"Full page screenshot saved: {full_page_path}")
                
                # Get page dimensions
                page_info = await page.evaluate("""
                    () => ({
                        scrollHeight: document.body.scrollHeight,
                        clientHeight: document.documentElement.clientHeight,
                        scrollWidth: document.body.scrollWidth,
                        clientWidth: document.documentElement.clientWidth
                    })
                """)
                
                print(f"Page dimensions: {page_info}")
                
                # Determine scroll strategy
                total_height = page_info['scrollHeight']
                viewport_height = page_info['clientHeight']
                
                if custom_scroll_strategy == "adaptive":
                    # Adaptive scrolling based on page content
                    scroll_positions = await self._get_adaptive_scroll_positions(page, total_height, viewport_height)
                elif custom_scroll_strategy == "manual":
                    # Manual scroll positions (you can customize this)
                    scroll_positions = [0, total_height // 4, total_height // 2, total_height * 3 // 4, total_height - viewport_height]
                else:
                    # Uniform scrolling (default)
                    scroll_positions = list(range(0, total_height, self.scroll_height))
                
                # Limit screenshots
                if len(scroll_positions) > max_screenshots:
                    step = len(scroll_positions) // max_screenshots
                    scroll_positions = scroll_positions[::step][:max_screenshots]
                
                print(f"Will take {len(scroll_positions)} viewport screenshots")
                
                # Take viewport screenshots at different scroll positions
                for i, scroll_pos in enumerate(scroll_positions):
                    try:
                        # Scroll to position
                        await page.evaluate(f"window.scrollTo(0, {scroll_pos})")
                        await asyncio.sleep(self.screenshot_delay)
                        
                        # Take screenshot
                        filename = self._generate_filename(url, i + 1)
                        screenshot_path = url_dir / filename
                        await page.screenshot(path=str(screenshot_path), full_page=False)
                        
                        screenshots_taken.append({
                            'path': str(screenshot_path),
                            'type': 'viewport',
                            'scroll_position': scroll_pos,
                            'filename': filename
                        })
                        
                        print(f"Screenshot {i+1}/{len(scroll_positions)} saved: {filename}")
                        
                    except Exception as e:
                        print(f"Error taking screenshot at position {scroll_pos}: {e}")
                        self.session_info['errors'].append({
                            'url': url,
                            'error': str(e),
                            'scroll_position': scroll_pos
                        })
                
                # Save page metadata
                metadata = {
                    'url': url,
                    'timestamp': datetime.now().isoformat(),
                    'page_info': page_info,
                    'screenshots_taken': len(screenshots_taken),
                    'scroll_strategy': custom_scroll_strategy,
                    'total_screenshots': len(screenshots_taken)
                }
                
                metadata_path = url_dir / "page_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"Capture complete for {url}: {len(screenshots_taken)} screenshots")
                
                return {
                    'url': url,
                    'success': True,
                    'screenshots': screenshots_taken,
                    'output_directory': str(url_dir),
                    'metadata': metadata
                }
                
            except Exception as e:
                error_msg = f"Error capturing {url}: {e}"
                print(error_msg)
                self.session_info['errors'].append({
                    'url': url,
                    'error': str(e)
                })
                return {
                    'url': url,
                    'success': False,
                    'error': str(e),
                    'screenshots': screenshots_taken
                }
            
            finally:
                if 'browser' in locals():
                    await browser.close()

    async def _get_adaptive_scroll_positions(self, page, total_height: int, viewport_height: int) -> List[int]:
        """Get scroll positions based on page content (e.g., section breaks)"""
        try:
            # Try to find natural breaking points (headers, sections, etc.)
            scroll_positions = await page.evaluate("""
                () => {
                    const positions = [0];
                    const headers = document.querySelectorAll('h1, h2, h3, section, article, .product, .item');
                    
                    headers.forEach(header => {
                        const rect = header.getBoundingClientRect();
                        const scrollY = window.pageYOffset + rect.top;
                        if (scrollY > 100) {  // Skip elements too close to top
                            positions.push(Math.floor(scrollY));
                        }
                    });
                    
                    // Remove duplicates and sort
                    return [...new Set(positions)].sort((a, b) => a - b);
                }
            """)
            
            # If we didn't find many natural breaks, fall back to uniform
            if len(scroll_positions) < 3:
                scroll_positions = list(range(0, total_height, self.scroll_height))
            
            return scroll_positions[:50]  # Limit to 50 positions max
            
        except Exception:
            # Fall back to uniform scrolling
            return list(range(0, total_height, self.scroll_height))

    async def capture_multiple_urls(self, urls: List[str], **kwargs) -> Dict:
        """
        Capture screenshots from multiple URLs
        
        Args:
            urls: List of Tor URLs to process
            **kwargs: Arguments passed to capture_single_page
            
        Returns:
            Summary of all captures
        """
        print(f"Starting batch capture for {len(urls)} URLs")
        
        results = []
        
        for i, url in enumerate(urls):
            print(f"\nProcessing URL {i+1}/{len(urls)}: {url}")
            try:
                result = await self.capture_single_page(url, **kwargs)
                results.append(result)
                self.session_info['urls_processed'].append(url)
                
                # Add delay between URLs to avoid overwhelming the server
                if i < len(urls) - 1:
                    print(f"Waiting before next URL...")
                    await asyncio.sleep(3)
                    
            except Exception as e:
                print(f"Failed to process {url}: {e}")
                results.append({
                    'url': url,
                    'success': False,
                    'error': str(e)
                })
        
        # Save session summary
        summary = {
            'session_info': self.session_info,
            'results': results,
            'total_urls': len(urls),
            'successful_urls': sum(1 for r in results if r.get('success')),
            'total_screenshots': sum(len(r.get('screenshots', [])) for r in results)
        }
        
        summary_path = self.output_dir / f"session_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nBatch capture complete! Summary saved to: {summary_path}")
        print(f"Total screenshots taken: {summary['total_screenshots']}")
        
        return summary

# Convenience functions for easy use
async def capture_tor_website(url: str, 
                            output_dir: str = "tor_screenshots",
                            max_screenshots: int = 20,
                            scroll_strategy: str = "uniform") -> str:
    """
    Simple function to capture a single Tor website
    
    Args:
        url: Tor URL to capture
        output_dir: Output directory
        max_screenshots: Maximum screenshots to take
        scroll_strategy: 'uniform', 'adaptive', or 'manual'
        
    Returns:
        Path to output directory containing screenshots
    """
    capturer = TorScreenshotCapture(output_dir=output_dir)
    result = await capturer.capture_single_page(
        url=url, 
        max_screenshots=max_screenshots,
        custom_scroll_strategy=scroll_strategy
    )
    
    if result['success']:
        print(f"\n Success! Screenshots saved to: {result['output_directory']}")
        print(f" Total screenshots: {len(result['screenshots'])}")
        return result['output_directory']
    else:
        print(f"Failed to capture {url}: {result['error']}")
        return None
