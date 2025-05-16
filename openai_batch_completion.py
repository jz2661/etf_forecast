import openai
import time
import asyncio
import aiohttp
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

class OpenAIBatchProcessor:
    def __init__(self, 
                 model: str = "gpt-3.5-turbo", 
                 max_requests_per_minute: int = 3500,
                 max_tokens_per_minute: int = 90000,
                 batch_size: int = 20):
        """
        Initialize the batch processor with rate limits.
        
        Args:
            model: The OpenAI model to use
            max_requests_per_minute: API request rate limit
            max_tokens_per_minute: Token rate limit
            batch_size: Number of concurrent requests
        """
        self.model = model
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.batch_size = batch_size
        self.request_count = 0
        self.token_count = 0
        self.last_reset_time = time.time()
        
    async def _make_request(self, session: aiohttp.ClientSession, 
                           prompt: str, 
                           max_tokens: int = 100) -> Dict[str, Any]:
        """Make a single API request with rate limiting."""
        # Check if we need to reset counters
        current_time = time.time()
        if current_time - self.last_reset_time >= 60:
            self.request_count = 0
            self.token_count = 0
            self.last_reset_time = current_time
            
        # Check if we're at the rate limit
        if (self.request_count >= self.max_requests_per_minute or 
            self.token_count >= self.max_tokens_per_minute):
            wait_time = 60 - (current_time - self.last_reset_time)
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
                self.request_count = 0
                self.token_count = 0
                self.last_reset_time = time.time()
        
        try:
            # For API version 1.0.0 and newer
            response = await openai.AsyncOpenAI().chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            
            # Update rate limiting counters
            self.request_count += 1
            # Approximate token count (prompt + response)
            estimated_prompt_tokens = len(prompt.split()) * 1.3
            self.token_count += int(estimated_prompt_tokens + max_tokens)
            
            return {
                "prompt": prompt,
                "completion": response.choices[0].message.content,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error making request: {str(e)}")
            return {
                "prompt": prompt,
                "completion": None,
                "error": str(e),
                "success": False
            }
    
    async def process_batch(self, prompts: List[str], 
                           max_tokens: int = 100) -> List[Dict[str, Any]]:
        """Process a batch of prompts with controlled concurrency."""
        results = []
        
        # Process in smaller batches to control concurrency
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i+self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}, size: {len(batch)}")
            
            async with aiohttp.ClientSession() as session:
                tasks = [self._make_request(session, prompt, max_tokens) for prompt in batch]
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
            
            # Add a small delay between batches to avoid rate limit issues
            if i + self.batch_size < len(prompts):
                await asyncio.sleep(0.5)
        
        return results


async def main():
    # Example usage with 100 tasks
    prompts = [f"Write a short poem about number {i}" for i in range(1, 101)]
    
    processor = OpenAIBatchProcessor(
        model="gpt-3.5-turbo",
        batch_size=10  # Process 10 requests concurrently
    )
    
    start_time = time.time()
    results = await processor.process_batch(prompts, max_tokens=50)
    end_time = time.time()
    
    # Print summary
    success_count = sum(1 for r in results if r["success"])
    logger.info(f"Completed {success_count}/{len(results)} tasks in {end_time - start_time:.2f} seconds")
    
    # Print first 3 results as examples
    for i, result in enumerate(results[:3]):
        if result["success"]:
            logger.info(f"Example {i+1}:\nPrompt: {result['prompt']}\nCompletion: {result['completion']}\n")

if __name__ == "__main__":
    asyncio.run(main()) 