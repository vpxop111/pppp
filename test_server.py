#!/usr/bin/env python3
"""
Simple test script to verify Flask server endpoints
"""
import requests
import json
import time

# Server configuration
SERVER_URL = "http://127.0.0.1:8000"  # Using the PORT you set

def test_endpoint(endpoint, method="GET", data=None):
    """Test a single endpoint"""
    url = f"{SERVER_URL}{endpoint}"
    print(f"\n{'='*50}")
    print(f"Testing {method} {endpoint}")
    print(f"URL: {url}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        # Try to parse JSON response
        try:
            json_response = response.json()
            print(f"Response: {json.dumps(json_response, indent=2)}")
        except:
            print(f"Response (text): {response.text[:200]}...")
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to server. Is it running?")
    except requests.exceptions.Timeout:
        print("❌ ERROR: Request timed out")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

def main():
    print("🚀 Testing Flask Server Endpoints")
    print(f"Server URL: {SERVER_URL}")
    
    # Test 1: Home page
    test_endpoint("/")
    
    # Test 2: Health check
    test_endpoint("/health")
    
    # Test 3: SVG generation (basic test)
    test_data = {
        "prompt": "Create a simple blue circle logo",
        "skip_enhancement": True
    }
    test_endpoint("/api/generate-svg", "POST", test_data)
    
    # Test 4: Parallel SVG generation (basic test)
    test_endpoint("/api/generate-parallel-svg", "POST", test_data)
    
    print(f"\n{'='*50}")
    print("✅ Test completed!")
    print("\nIf you see connection errors, make sure the server is running with:")
    print("PORT=8000 python3 parallel_svg_pipeline.py")

if __name__ == "__main__":
    main() 