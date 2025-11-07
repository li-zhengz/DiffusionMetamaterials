#!/usr/bin/env python3
"""
Test script to evaluate equation generation with different total numbers.
Tests: 1000, 2000, 3000, 5000 equations
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), 'dataGeneration'))

from data_generator import main

def test_generation_size(num_eqs, num_processes=20):
    """Test equation generation with a specific number of equations"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING {num_eqs} EQUATIONS")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        valid_eqs, invalid_eqs = main(num_eqs, num_processes)
        
        end_time = time.time()
        total_time = end_time - start_time
        total_generated = len(valid_eqs) + len(invalid_eqs)
        
        # Calculate statistics
        success_rate = (len(valid_eqs) / total_generated * 100) if total_generated > 0 else 0
        eqs_per_second = total_generated / total_time if total_time > 0 else 0
        
        print(f"📊 RESULTS:")
        print(f"   Target equations: {num_eqs}")
        print(f"   Total generated: {total_generated}")
        print(f"   Valid equations: {len(valid_eqs)}")
        print(f"   Invalid equations: {len(invalid_eqs)}")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Generation time: {total_time:.2f} seconds")
        print(f"   Speed: {eqs_per_second:.1f} equations/second")
        
        # Show examples
        print(f"\n📝 EXAMPLES:")
        print("Valid equations:")
        for i, eq in enumerate(valid_eqs[:2]):
            print(f"  ✅ {eq}")
            
        print("Invalid equations:")
        for i, eq in enumerate(invalid_eqs[:2]):
            print(f"  ❌ {eq}")
        
        return {
            'target': num_eqs,
            'total_generated': total_generated,
            'valid_count': len(valid_eqs),
            'invalid_count': len(invalid_eqs),
            'success_rate': success_rate,
            'time': total_time,
            'speed': eqs_per_second,
            'valid_equations': valid_eqs,
            'invalid_equations': invalid_eqs
        }
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main_test():
    """Run tests with different equation counts"""
    print("🚀 EQUATION GENERATION TEST")
    print("Testing different numbers of total generations")
    
    test_sizes = [1000,2000,3000,5000]
    results = []
    
    for size in test_sizes:
        result = test_generation_size(size)
        if result:
            results.append(result)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("📈 SCALING ANALYSIS")
    print(f"{'='*80}")
    
    print(f"{'Target':<8} {'Generated':<10} {'Valid':<8} {'Invalid':<8} {'Success%':<9} {'Time(s)':<8} {'Speed':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['target']:<8} "
              f"{result['total_generated']:<10} "
              f"{result['valid_count']:<8} "
              f"{result['invalid_count']:<8} "
              f"{result['success_rate']:<9.1f} "
              f"{result['time']:<8.2f} "
              f"{result['speed']:<10.1f}")
    
    # Analysis
    if len(results) >= 2:
        print(f"\n🔍 OBSERVATIONS:")
        
        # Success rate consistency
        success_rates = [r['success_rate'] for r in results]
        avg_success = sum(success_rates) / len(success_rates)
        success_std = (sum((x - avg_success)**2 for x in success_rates) / len(success_rates))**0.5
        
        print(f"   Average success rate: {avg_success:.1f}% (±{success_std:.1f}%)")
        
        # Speed analysis
        speeds = [r['speed'] for r in results]
        print(f"   Speed range: {min(speeds):.1f} - {max(speeds):.1f} equations/second")
        
        # Scaling efficiency
        first_result = results[0]
        last_result = results[-1]
        scale_factor = last_result['target'] / first_result['target']
        time_factor = last_result['time'] / first_result['time']
        
if __name__ == '__main__':
    main_test()
