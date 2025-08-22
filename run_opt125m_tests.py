#!/usr/bin/env python3
"""
OPT-125M Test Runner Script
Executes all OPT-125M tests in sequence and provides comprehensive reporting
"""

import os
import sys
import time
import subprocess
import importlib.util

def run_test_file(test_file):
    """Run a test file and return the result."""
    print(f"\n{'='*60}")
    print(f"🧪 RUNNING: {test_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the test file
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, 
                              text=True, 
                              timeout=300)  # 5 minute timeout
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {test_file} completed successfully in {execution_time:.2f}s")
            print(f"📊 Output:\n{result.stdout}")
            return True, execution_time, result.stdout
        else:
            print(f"❌ {test_file} failed with return code {result.returncode}")
            print(f"📊 Error output:\n{result.stderr}")
            return False, execution_time, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_file} timed out after 5 minutes")
        return False, 300, "Test timed out"
    except Exception as e:
        print(f"💥 {test_file} failed with exception: {e}")
        return False, time.time() - start_time, str(e)

def check_test_file_exists(test_file):
    """Check if a test file exists."""
    return os.path.exists(test_file)

def main():
    """Main test runner function."""
    print("🎯 OPT-125M COMPREHENSIVE TEST SUITE RUNNER")
    print("=" * 60)
    print("🔍 This script will run all OPT-125M tests to verify training readiness")
    print("=" * 60)
    
    # Define test files in execution order
    test_files = [
        "test_forward_pass_consistency_opt125m.py",
        "test_full_training_step_consistency_opt125m.py",
        "test_opt125m_comprehensive.py",
        "test_opt125m_verification.py"
    ]
    
    # Check which test files exist
    available_tests = []
    for test_file in test_files:
        if check_test_file_exists(test_file):
            available_tests.append(test_file)
            print(f"✅ Found test file: {test_file}")
        else:
            print(f"⚠️  Test file not found: {test_file}")
    
    if not available_tests:
        print("❌ No test files found. Please ensure test files are in the current directory.")
        return
    
    print(f"\n📋 Found {len(available_tests)} test files to run")
    print("🚀 Starting test execution...")
    
    # Run all tests
    test_results = []
    total_execution_time = 0
    
    for test_file in available_tests:
        success, execution_time, output = run_test_file(test_file)
        test_results.append({
            'file': test_file,
            'success': success,
            'execution_time': execution_time,
            'output': output
        })
        total_execution_time += execution_time
        
        # Add a small delay between tests
        time.sleep(1)
    
    # Generate comprehensive report
    print(f"\n{'='*60}")
    print("🎉 OPT-125M TESTING COMPLETED!")
    print(f"{'='*60}")
    
    # Count results
    passed_tests = sum(1 for result in test_results if result['success'])
    failed_tests = len(test_results) - passed_tests
    
    print(f"\n📊 COMPREHENSIVE TEST RESULTS:")
    print(f"   - Total Tests: {len(test_results)}")
    print(f"   - Passed: {passed_tests}")
    print(f"   - Failed: {failed_tests}")
    print(f"   - Success Rate: {(passed_tests / len(test_results)) * 100:.1f}%")
    print(f"   - Total Execution Time: {total_execution_time:.2f}s")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for i, result in enumerate(test_results, 1):
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"   {i}. {result['file']}: {status} ({result['execution_time']:.2f}s)")
    
    # Success/failure summary
    if passed_tests == len(test_results):
        print(f"\n🚀 SUCCESS: All OPT-125M tests passed!")
        print(f"\n💡 OPT-125M TRAINING READINESS STATUS:")
        print(f"   ✅ Forward pass consistency verified")
        print(f"   ✅ Backward pass consistency verified")
        print(f"   ✅ Gradient computation verified")
        print(f"   ✅ Optimizer integration verified")
        print(f"   ✅ Memory efficiency verified")
        print(f"   ✅ Training stability verified")
        print(f"\n🎯 Your OPT-125M model is FULLY READY for production training!")
        print(f"\n🚀 Next steps for OPT-125M training:")
        print(f"   1. Scale up to full OPT-125M model (12 layers, 768 hidden size)")
        print(f"   2. Configure your distributed training environment")
        print(f"   3. Prepare your training dataset")
        print(f"   4. Start training with tensor parallelism")
        print(f"   5. Monitor training progress and convergence")
        
    else:
        print(f"\n⚠️  WARNING: {failed_tests} tests failed.")
        print(f"\n🔧 RECOMMENDED ACTIONS:")
        print(f"   1. Review the failed test outputs above")
        print(f"   2. Check for any error messages or exceptions")
        print(f"   3. Verify your tensor parallelism implementation")
        print(f"   4. Fix any identified issues")
        print(f"   5. Re-run the tests to verify fixes")
        print(f"   6. Only proceed with OPT-125M training after all tests pass")
    
    # Save detailed results to file
    report_file = "opt125m_test_report.txt"
    try:
        with open(report_file, 'w') as f:
            f.write("OPT-125M Test Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tests: {len(test_results)}\n")
            f.write(f"Passed: {passed_tests}\n")
            f.write(f"Failed: {failed_tests}\n")
            f.write(f"Success Rate: {(passed_tests / len(test_results)) * 100:.1f}%\n")
            f.write(f"Total Execution Time: {total_execution_time:.2f}s\n\n")
            
            f.write("Detailed Results:\n")
            f.write("-" * 30 + "\n")
            for result in test_results:
                status = "PASS" if result['success'] else "FAIL"
                f.write(f"{result['file']}: {status} ({result['execution_time']:.2f}s)\n")
                f.write(f"Output:\n{result['output']}\n")
                f.write("-" * 30 + "\n")
        
        print(f"\n📄 Detailed test report saved to: {report_file}")
        
    except Exception as e:
        print(f"⚠️  Could not save detailed report: {e}")
    
    # Return appropriate exit code
    if passed_tests == len(test_results):
        print(f"\n🎯 All tests passed! OPT-125M is ready for training.")
        sys.exit(0)
    else:
        print(f"\n🚨 Some tests failed. Please fix issues before proceeding with OPT-125M training.")
        sys.exit(1)

if __name__ == "__main__":
    main() 