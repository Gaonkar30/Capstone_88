import asyncio
from playwright.async_api import async_playwright


async def take_screenshots_tor():

    tor_url = "http://s53txpmxln2657wz6kbueokpg3kbu4k4whunczeeotfpgb77l2tcwaad.onion/page/10/"
    screenshot_dir = "arms and ammunition"
    screenshot_prefix = "astraguns"
    scroll_height = 500

    # Tor proxy settings
    proxy = {
        "server": "socks5://127.0.0.1:9150",
    }

    async with async_playwright() as p:
        # Launch browser with proxy settings
        browser = await p.chromium.launch(proxy=proxy, headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            # Navigate to the Tor website
            await page.goto(tor_url, timeout=100000)

            # Get the total height of the page
            total_height = await page.evaluate("document.body.scrollHeight")

            # Scroll and take screenshots
            current_scroll = 0
            screenshot_count = 86

            while current_scroll < total_height:
                # Take a screenshot
                screenshot_path = f"{screenshot_dir}/{screenshot_prefix}_{screenshot_count}.png"
                await page.screenshot(path=screenshot_path, full_page=False)
                print(f"Screenshot saved to {screenshot_path}")

                # Scroll down
                await page.evaluate(f"window.scrollBy(0, {scroll_height});")
                current_scroll += scroll_height
                screenshot_count += 1


                await asyncio.sleep(1)

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            await browser.close()



if __name__ == "__main__":
    asyncio.run(take_screenshots_tor())
