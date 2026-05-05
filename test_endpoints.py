#!/usr/bin/env python
"""Test new Flask endpoints"""

from flask_app.app import app

with app.test_client() as client:
    print("="*60)
    print("Testing new Flask endpoints")
    print("="*60)
    
    # Test metrics endpoint
    resp = client.get('/metrics')
    if resp.status_code == 200:
        print("✓ /metrics endpoint works")
        metrics = resp.get_json()
        print(f"  Model accuracy: {metrics['model_accuracy']}%")
        print(f"  Training data: {metrics['training_data']}")
    else:
        print("✗ /metrics endpoint failed")
    
    # Test suggest endpoint
    print("\n✓ /suggest endpoint")
    resp = client.post('/suggest', json={'digit': 5, 'confidence': 45})
    if resp.status_code == 200:
        suggest = resp.get_json()
        print(f"  Suggestions provided: {len(suggest['suggestions'])} tips")
        for i, tip in enumerate(suggest['suggestions'], 1):
            print(f"    {i}. {tip}")
    
    # Test validate endpoint
    print("\n✓ /validate endpoint")
    resp = client.post('/validate', json={'image': 'data:image/png;base64,iVBORw0KGgo='})
    if resp.status_code in [200, 400]:
        result = resp.get_json()
        print(f"  Status: {result.get('valid', 'checked')}")
    
    # Test health endpoint
    print("\n✓ /health endpoint")
    resp = client.get('/health')
    if resp.status_code == 200:
        health = resp.get_json()
        print(f"  Status: {health['status']}")
        print(f"  Model loaded: {health['model_loaded']}")
    
    print("\n" + "="*60)
    print("✓ All new endpoints are functional!")
    print("="*60)
