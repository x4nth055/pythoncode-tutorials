#!/usr/bin/env python3
import requests
import json
import os
import argparse
from typing import Dict, List, Tuple
from openai import OpenAI

class SecurityHeadersAnalyzer:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY')
        self.base_url = base_url or os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
        self.model = model or os.getenv('LLM_MODEL', 'deepseek/deepseek-chat-v3.1:free')
        
        if not self.api_key:
            raise ValueError("API key is required. Set OPENROUTER_API_KEY or provide --api-key")
        
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def fetch_headers(self, url: str, timeout: int = 10) -> Tuple[Dict[str, str], int]:
        """Fetch HTTP headers from URL"""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        try:
            response = requests.get(url, timeout=timeout, allow_redirects=True)
            return dict(response.headers), response.status_code
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return {}, 0

    def analyze_headers(self, url: str, headers: Dict[str, str], status_code: int) -> str:
        """Analyze headers using LLM"""
        prompt = f"""Analyze the HTTP security headers for {url} (Status: {status_code})

Headers:
{json.dumps(headers, indent=2)}

Provide a comprehensive security analysis including:
1. Security score (0-100) and overall assessment
2. Critical security issues that need immediate attention
3. Missing important security headers
4. Analysis of existing security headers and their effectiveness
5. Specific recommendations for improvement
6. Potential security risks based on current configuration

Focus on practical, actionable advice following current web security best practices. Please do not include ** and # 
in the response except for specific references where necessary. use numbers, romans, alphabets instead Format the response well please. """

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Analysis failed: {e}"

    def analyze_url(self, url: str, timeout: int = 10) -> Dict:
        """Analyze a single URL"""
        print(f"\nAnalyzing: {url}")
        print("-" * 50)
        
        headers, status_code = self.fetch_headers(url, timeout)
        if not headers:
            return {"url": url, "error": "Failed to fetch headers"}
        
        print(f"Status Code: {status_code}")
        print(f"\nHTTP Headers ({len(headers)} found):")
        print("-" * 30)
        for key, value in headers.items():
            print(f"{key}: {value}")
        
        print(f"\nAnalyzing with AI...")
        analysis = self.analyze_headers(url, headers, status_code)
        
        print("\nSECURITY ANALYSIS")
        print("=" * 50)
        print(analysis)
        
        return {
            "url": url,
            "status_code": status_code,
            "headers_count": len(headers),
            "analysis": analysis,
            "raw_headers": headers
        }

    def analyze_multiple_urls(self, urls: List[str], timeout: int = 10) -> List[Dict]:
        """Analyze multiple URLs"""
        results = []
        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}]")
            result = self.analyze_url(url, timeout)
            results.append(result)
        return results

    def export_results(self, results: List[Dict], filename: str):
        """Export results to JSON"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults exported to: {filename}")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze HTTP security headers using AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  python security_headers.py https://example.com
  python security_headers.py example.com google.com
  python security_headers.py example.com --export results.json
  
Environment Variables:
  OPENROUTER_API_KEY - API key for OpenRouter
  OPENAI_API_KEY     - API key for OpenAI
  LLM_MODEL         - Model to use (default: deepseek/deepseek-chat-v3.1:free)'''
    )
    
    parser.add_argument('urls', nargs='+', help='URLs to analyze')
    parser.add_argument('--api-key', help='API key for LLM service')
    parser.add_argument('--base-url', help='Base URL for LLM API')
    parser.add_argument('--model', help='LLM model to use')
    parser.add_argument('--timeout', type=int, default=10, help='Request timeout (default: 10s)')
    parser.add_argument('--export', help='Export results to JSON file')
    
    args = parser.parse_args()
    
    try:
        analyzer = SecurityHeadersAnalyzer(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model
        )
        
        results = analyzer.analyze_multiple_urls(args.urls, args.timeout)
        
        if args.export:
            analyzer.export_results(results, args.export)
            
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return 1

if __name__ == '__main__':
    main()