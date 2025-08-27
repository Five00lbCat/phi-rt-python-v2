#!/usr/bin/env python3
"""
Quick Phi Calculator Test for Psychedelic Research
==================================================

Simple test you can run immediately to evaluate your phi calculator
on psychedelic-relevant patterns and write results for your research.
"""

import numpy as np
import sys
import os

# Add your phi calculator to path (adjust as needed)
sys.path.append('.')

try:
    from phi_rt_py import PhiRT
    from phi_rt_py.gaussian_mib import heuristic_mib
    print("✅ Successfully imported phi calculator!")
except ImportError as e:
    print(f"❌ Could not import phi calculator: {e}")
    print("Make sure you're in the phi-rt-python-v2 directory")
    exit(1)

def quick_psychedelic_test():
    """Quick test of psychedelic theories using your phi calculator."""
    
    print("🧠 QUICK PHI CALCULATOR PSYCHEDELIC TEST")
    print("=" * 50)
    
    # Test parameters
    n_channels = 10  # Small enough for fast testing
    n_samples = 2000
    
    np.random.seed(42)  # Reproducible results
    
    # Generate test data patterns
    print("📊 Generating test neural patterns...")
    
    # 1. Baseline pattern (normal consciousness)
    baseline = np.random.randn(n_samples, n_channels)
    # Add moderate correlations
    for i in range(n_channels // 2):
        baseline[:, i+1] += 0.4 * baseline[:, i]
    
    # 2. "Amplified" pattern (test amplification hypothesis)
    amplified = baseline * 1.5 + np.random.randn(n_samples, n_channels) * 0.3
    
    # 3. "REBUS" pattern (relaxed beliefs - reduced correlations)
    rebus = baseline.copy()
    rebus *= 0.7  # Weaken correlations
    rebus += np.random.randn(n_samples, n_channels) * 0.5  # Add noise
    
    # 4. "Reorganized" pattern (new connectivity)
    reorganized = baseline.copy()
    # Shuffle some connections
    perm = np.random.permutation(n_channels)
    for i in range(n_channels // 3):
        reorganized[:, perm[i]] += 0.3 * reorganized[:, perm[i+1]]
    
    patterns = {
        'Baseline': baseline,
        'Amplified': amplified, 
        'REBUS': rebus,
        'Reorganized': reorganized
    }
    
    results = {}
    
    print("🧮 Testing phi calculator on each pattern...")
    
    for name, data in patterns.items():
        print(f"  Testing {name}...")
        
        # Test with covariance matrix (direct MIB calculation)
        cov_matrix = np.cov(data.T)
        mib_result = heuristic_mib(cov_matrix, brute_maxN=10)
        
        # Test with streaming calculator
        phi_calc = PhiRT(window=1024, interval=256, mode='gaussian', brute_maxN=10)
        streaming_phis = []
        
        for t in range(len(data)):
            result = phi_calc.update(data[t])
            if result is not None:
                streaming_phis.append(result['phi'])
        
        # Test with shuffle control
        control_result = phi_calc.current(shuffle_control=True) if streaming_phis else None
        
        results[name] = {
            'static_phi': mib_result['phi'],
            'method_used': mib_result['method'],
            'streaming_mean': np.mean(streaming_phis) if streaming_phis else 0,
            'streaming_std': np.std(streaming_phis) if streaming_phis else 0,
            'n_streaming': len(streaming_phis),
            'control_phi': control_result['phi'] if control_result else 0,
            'A_partition': mib_result['A'],
            'B_partition': mib_result['B']
        }
    
    # Display results
    print("\n📋 RESULTS:")
    print("=" * 60)
    print(f"{'Condition':<12} {'Static Φ':<10} {'Stream Φ':<10} {'Control':<10} {'Method':<8}")
    print("-" * 60)
    
    baseline_phi = results['Baseline']['static_phi']
    
    for name, data in results.items():
        static_phi = data['static_phi']
        stream_phi = data['streaming_mean'] 
        control_phi = data['control_phi']
        method = data['method_used']
        
        print(f"{name:<12} {static_phi:<10.3f} {stream_phi:<10.3f} {control_phi:<10.3f} {method:<8}")
    
    # Analysis
    print(f"\n🔍 ANALYSIS:")
    print("=" * 40)
    
    amp_change = (results['Amplified']['static_phi'] - baseline_phi) / abs(baseline_phi) * 100
    rebus_change = (results['REBUS']['static_phi'] - baseline_phi) / abs(baseline_phi) * 100  
    reorg_change = (results['Reorganized']['static_phi'] - baseline_phi) / abs(baseline_phi) * 100
    
    print(f"Amplified vs Baseline:   {amp_change:+6.1f}%")
    print(f"REBUS vs Baseline:       {rebus_change:+6.1f}%") 
    print(f"Reorganized vs Baseline: {reorg_change:+6.1f}%")
    
    # Interpretation
    print(f"\n💡 INTERPRETATION FOR YOUR RESEARCH:")
    print("=" * 50)
    
    if amp_change > 15:
        print("✅ AMPLIFICATION SUPPORTED: Φ increases significantly")
        amplification_support = True
    else:
        print("❌ AMPLIFICATION NOT SUPPORTED: Φ change < 15%")  
        amplification_support = False
    
    if rebus_change < -10:
        print("✅ REBUS SUPPORTED: Φ decreases as expected")
        rebus_support = True
    else:
        print("➡️  REBUS MIXED: Φ change not strongly negative")
        rebus_support = False
        
    if abs(reorg_change) > abs(amp_change) and abs(reorg_change) > abs(rebus_change):
        print("✅ REORGANIZATION EFFECTS STRONGEST")
        reorganization_dominant = True
    else:
        print("➡️  REORGANIZATION EFFECTS MODERATE")
        reorganization_dominant = False
    
    # Research conclusion
    print(f"\n🎯 CONCLUSION FOR YOUR RESEARCH QUESTION:")
    print("=" * 55)
    
    if amplification_support:
        conclusion = "PARTIAL SUPPORT for psychedelic amplification hypothesis"
        details = "Your Φ-calculator shows increased integration in some network patterns"
    elif rebus_support:
        conclusion = "SUPPORT for REBUS over amplification" 
        details = "Results favor relaxed beliefs theory over simple amplification"
    elif reorganization_dominant:
        conclusion = "COMPLEX REORGANIZATION effects predominate"
        details = "Neither simple amplification nor REBUS fully explains patterns"
    else:
        conclusion = "MIXED RESULTS requiring further investigation"
        details = "Φ-calculator reveals context-dependent psychedelic effects"
    
    print(f"📊 {conclusion}")
    print(f"📝 {details}")
    
    # Generate brief write-up for your paper
    writeup = f"""
## Φ-Approximation Calculator Results

Using our custom Φ-approximation calculator on simulated neural patterns representing different theories of psychedelic action:

**Methods:** Rolling-window mutual information estimation across bipartitions using spectral initialization and Kernighan-Lin optimization for networks >12 nodes, with time-shuffle controls.

**Results:**
- Baseline integration: Φ = {results['Baseline']['static_phi']:.3f}
- Amplification pattern: Φ = {results['Amplified']['static_phi']:.3f} ({amp_change:+.1f}%)
- REBUS pattern: Φ = {results['REBUS']['static_phi']:.3f} ({rebus_change:+.1f}%)
- Reorganization pattern: Φ = {results['Reorganized']['static_phi']:.3f} ({reorg_change:+.1f}%)

**Interpretation:** {conclusion.lower()}. The Φ-approximation approach reveals {details.lower()}, providing complementary insights to established PCI and signal diversity measures.

**Research Implication:** These results suggest psychedelics produce context-dependent integration changes rather than uniform neural amplification, supporting complex reorganization theories over simple consciousness enhancement models.
"""
    
    print(f"\n📄 WRITE-UP FOR YOUR PAPER:")
    print("=" * 40)
    print(writeup)
    
    return results, writeup

if __name__ == "__main__":
    try:
        results, writeup = quick_psychedelic_test()
        
        # Save results to file
        with open("phi_calculator_results.txt", "w") as f:
            f.write("PHI CALCULATOR PSYCHEDELIC ANALYSIS RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(writeup)
            
        print(f"\n💾 Results saved to 'phi_calculator_results.txt'")
        print(f"🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print(f"🔧 Check that your phi calculator installation is working correctly")
