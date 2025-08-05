"""
Comprehensive validation checklist for RTAI production readiness
"""
import asyncio
import time
import inspect
from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger

from ..indicators.base import OFI, VPIN, KyleLambda, LPI, AdvancedZBandWrapper
from ..state import StateAdapter, create_state_store
from ..utils.environment import validate_environment_security
from ..utils.metrics import PerformanceMetrics


class ValidationSuite:
    """Comprehensive validation for production readiness"""
    
    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.total_checks = 0
        self.passed_checks = 0
        self.failed_checks = 0
        self.warnings = []
        
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        logger.info("🔍 Starting comprehensive RTAI validation...")
        
        validation_categories = [
            ("Environment Security", self._validate_environment),
            ("Indicator Integration", self._validate_indicators),
            ("State Management", self._validate_state_management),
            ("Performance Metrics", self._validate_performance),
            ("Error Handling", self._validate_error_handling),
            ("Memory Management", self._validate_memory_management),
            ("Production Safety", self._validate_production_safety),
            ("API Consistency", self._validate_api_consistency)
        ]
        
        start_time = time.time()
        
        for category_name, validation_func in validation_categories:
            logger.info(f"🧪 Validating: {category_name}")
            try:
                category_result = await validation_func()
                self.results[category_name] = category_result
                self._update_counters(category_result)
            except Exception as e:
                logger.error(f"❌ Validation failed for {category_name}: {e}")
                self.results[category_name] = {
                    'status': 'error',
                    'error': str(e),
                    'checks': []
                }
                self.failed_checks += 1
        
        total_time = time.time() - start_time
        
        final_result = {
            'status': 'pass' if self.failed_checks == 0 else 'fail',
            'summary': {
                'total_checks': self.total_checks,
                'passed': self.passed_checks,
                'failed': self.failed_checks,
                'warnings': len(self.warnings),
                'validation_time': f"{total_time:.2f}s"
            },
            'categories': self.results,
            'warnings': self.warnings,
            'recommendations': self._generate_recommendations()
        }
        
        self._log_final_results(final_result)
        return final_result
    
    def _update_counters(self, category_result: Dict[str, Any]):
        """Update validation counters"""
        if 'checks' in category_result:
            for check in category_result['checks']:
                self.total_checks += 1
                if check.get('status') == 'pass':
                    self.passed_checks += 1
                elif check.get('status') == 'fail':
                    self.failed_checks += 1
                elif check.get('status') == 'warning':
                    self.warnings.append(check.get('message', 'Unknown warning'))
    
    async def _validate_environment(self) -> Dict[str, Any]:
        """Validate environment security and configuration"""
        checks = []
        
        # Environment security
        env_result = validate_environment_security()
        checks.append({
            'name': 'Environment Security',
            'status': 'pass' if env_result['valid'] else 'fail',
            'message': f"Environment validation: {len(env_result['issues'])} issues found",
            'details': env_result
        })
        
        # Required directories
        required_dirs = ['logs', 'output', 'state']
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            checks.append({
                'name': f'Directory: {dir_name}',
                'status': 'pass' if dir_path.exists() else 'warning',
                'message': f"Directory {dir_name}: {'exists' if dir_path.exists() else 'missing (will be created)'}",
                'details': {'path': str(dir_path.absolute())}
            })
        
        return {
            'status': 'pass' if all(c['status'] in ['pass', 'warning'] for c in checks) else 'fail',
            'checks': checks
        }
    
    async def _validate_indicators(self) -> Dict[str, Any]:
        """Validate all indicator classes"""
        checks = []
        
        # Test core indicators
        indicator_classes = [OFI, VPIN, KyleLambda, LPI, AdvancedZBandWrapper]
        
        for indicator_class in indicator_classes:
            try:
                # Check class instantiation
                if indicator_class == AdvancedZBandWrapper:
                    # Special case - needs window_size parameter
                    indicator = indicator_class(window_size=120)
                else:
                    indicator = indicator_class()
                
                # Check required methods
                required_methods = ['update', 'get_value']
                has_methods = all(hasattr(indicator, method) for method in required_methods)
                
                # Check state management
                has_state_methods = hasattr(indicator, 'save_state') and hasattr(indicator, 'load_state')
                
                checks.append({
                    'name': f'Indicator: {indicator_class.__name__}',
                    'status': 'pass' if has_methods else 'fail',
                    'message': f"{indicator_class.__name__}: {'✅ Valid' if has_methods else '❌ Missing methods'}",
                    'details': {
                        'has_required_methods': has_methods,
                        'has_state_methods': has_state_methods,
                        'methods': [m for m in dir(indicator) if not m.startswith('_')]
                    }
                })
                
            except Exception as e:
                checks.append({
                    'name': f'Indicator: {indicator_class.__name__}',
                    'status': 'fail',
                    'message': f"{indicator_class.__name__}: ❌ Failed to instantiate: {e}",
                    'details': {'error': str(e)}
                })
        
        return {
            'status': 'pass' if all(c['status'] == 'pass' for c in checks) else 'fail',
            'checks': checks
        }
    
    async def _validate_state_management(self) -> Dict[str, Any]:
        """Validate state management system"""
        checks = []
        
        # Test state stores
        store_types = ['memory', 'parquet']
        
        for store_type in store_types:
            try:
                store = create_state_store(store_type)
                adapter = StateAdapter(store)
                
                # Test basic operations
                test_state = {'test_key': 'test_value', 'counter': 42}
                save_success = store.save_state('test_component', test_state)
                
                if save_success:
                    loaded_state = store.load_state('test_component')
                    state_matches = loaded_state == test_state
                    
                    checks.append({
                        'name': f'State Store: {store_type}',
                        'status': 'pass' if state_matches else 'fail',
                        'message': f"{store_type} store: {'✅ Working' if state_matches else '❌ State mismatch'}",
                        'details': {
                            'save_success': save_success,
                            'state_matches': state_matches,
                            'original': test_state,
                            'loaded': loaded_state
                        }
                    })
                else:
                    checks.append({
                        'name': f'State Store: {store_type}',
                        'status': 'fail',
                        'message': f"{store_type} store: ❌ Save failed",
                        'details': {'save_success': False}
                    })
                    
                # Cleanup
                store.clear_state('test_component')
                
            except Exception as e:
                checks.append({
                    'name': f'State Store: {store_type}',
                    'status': 'fail',
                    'message': f"{store_type} store: ❌ Error: {e}",
                    'details': {'error': str(e)}
                })
        
        return {
            'status': 'pass' if all(c['status'] == 'pass' for c in checks) else 'fail',
            'checks': checks
        }
    
    async def _validate_performance(self) -> Dict[str, Any]:
        """Validate performance monitoring"""
        checks = []
        
        try:
            metrics = PerformanceMetrics()
            
            # Test metrics collection
            start_time = time.time()
            await asyncio.sleep(0.1)  # Simulate work
            
            metrics.record_latency(time.time() - start_time)
            metrics.record_throughput(100)
            
            # Check metrics functionality
            latency_data = metrics.get_latency_stats()
            throughput_data = metrics.get_throughput_stats()
            
            checks.append({
                'name': 'Performance Metrics',
                'status': 'pass' if latency_data and throughput_data else 'fail',
                'message': f"Metrics: {'✅ Recording' if latency_data else '❌ Not recording'}",
                'details': {
                    'latency_stats': latency_data,
                    'throughput_stats': throughput_data
                }
            })
            
        except Exception as e:
            checks.append({
                'name': 'Performance Metrics',
                'status': 'fail',
                'message': f"Metrics: ❌ Error: {e}",
                'details': {'error': str(e)}
            })
        
        return {
            'status': 'pass' if all(c['status'] == 'pass' for c in checks) else 'fail',
            'checks': checks
        }
    
    async def _validate_error_handling(self) -> Dict[str, Any]:
        """Validate error handling patterns"""
        checks = []
        
        # Test indicator error handling
        try:
            ofi = OFI()
            
            # Test with invalid data
            try:
                ofi.update(None, None)  # Invalid inputs
                error_handled = True
            except Exception:
                error_handled = False  # Should handle gracefully
            
            checks.append({
                'name': 'Indicator Error Handling',
                'status': 'pass' if error_handled else 'warning',
                'message': f"Error handling: {'✅ Graceful' if error_handled else '⚠️ Exceptions not caught'}",
                'details': {'graceful_handling': error_handled}
            })
            
        except Exception as e:
            checks.append({
                'name': 'Indicator Error Handling',
                'status': 'fail',
                'message': f"Error handling test failed: {e}",
                'details': {'error': str(e)}
            })
        
        return {
            'status': 'pass' if all(c['status'] in ['pass', 'warning'] for c in checks) else 'fail',
            'checks': checks
        }
    
    async def _validate_memory_management(self) -> Dict[str, Any]:
        """Validate memory management"""
        checks = []
        
        try:
            import psutil
            import gc
            
            # Get initial memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create and destroy indicators
            indicators = []
            for _ in range(10):
                ofi = OFI()
                for i in range(100):
                    ofi.update(100, 50000)  # trade_qty, current_price
                indicators.append(ofi)
            
            # Clear references and force GC
            indicators.clear()
            gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = final_memory - initial_memory
            
            checks.append({
                'name': 'Memory Management',
                'status': 'pass' if memory_growth < 50 else 'warning',  # Allow 50MB growth
                'message': f"Memory: {memory_growth:.1f}MB growth ({'✅ Acceptable' if memory_growth < 50 else '⚠️ High'})",
                'details': {
                    'initial_memory_mb': initial_memory,
                    'final_memory_mb': final_memory,
                    'growth_mb': memory_growth
                }
            })
            
        except ImportError:
            checks.append({
                'name': 'Memory Management',
                'status': 'warning',
                'message': "Memory validation skipped: psutil not available",
                'details': {'psutil_available': False}
            })
        except Exception as e:
            checks.append({
                'name': 'Memory Management',
                'status': 'fail',
                'message': f"Memory validation failed: {e}",
                'details': {'error': str(e)}
            })
        
        return {
            'status': 'pass' if all(c['status'] in ['pass', 'warning'] for c in checks) else 'fail',
            'checks': checks
        }
    
    async def _validate_production_safety(self) -> Dict[str, Any]:
        """Validate production safety measures"""
        checks = []
        
        # Check for debug code
        debug_patterns = ['print(', 'pdb.', 'breakpoint()', 'debug=True']
        python_files = list(Path('rtai').glob('**/*.py'))
        
        debug_found = []
        for file_path in python_files:
            try:
                content = file_path.read_text(encoding='utf-8')
                for pattern in debug_patterns:
                    if pattern in content:
                        debug_found.append(f"{file_path}:{pattern}")
            except Exception:
                continue
        
        checks.append({
            'name': 'Debug Code Detection',
            'status': 'pass' if not debug_found else 'warning',
            'message': f"Debug code: {'✅ None found' if not debug_found else f'⚠️ {len(debug_found)} instances'}",
            'details': {'debug_instances': debug_found}
        })
        
        # Check for TODO/FIXME comments
        todo_patterns = ['TODO', 'FIXME', 'HACK', 'XXX']
        todo_found = []
        for file_path in python_files:
            try:
                content = file_path.read_text(encoding='utf-8')
                for line_num, line in enumerate(content.split('\n'), 1):
                    for pattern in todo_patterns:
                        if pattern in line.upper():
                            todo_found.append(f"{file_path}:{line_num}")
            except Exception:
                continue
        
        checks.append({
            'name': 'Code Quality Markers',
            'status': 'warning' if todo_found else 'pass',
            'message': f"TODO/FIXME: {'✅ None found' if not todo_found else f'⚠️ {len(todo_found)} items'}",
            'details': {'todo_items': todo_found[:10]}  # Limit output
        })
        
        return {
            'status': 'pass' if all(c['status'] in ['pass', 'warning'] for c in checks) else 'fail',
            'checks': checks
        }
    
    async def _validate_api_consistency(self) -> Dict[str, Any]:
        """Validate API consistency across components"""
        checks = []
        
        # Check method signatures consistency
        indicator_classes = [OFI, VPIN, KyleLambda, LPI]
        update_signatures = {}
        
        for indicator_class in indicator_classes:
            try:
                update_method = getattr(indicator_class, 'update')
                signature = inspect.signature(update_method)
                update_signatures[indicator_class.__name__] = str(signature)
            except Exception as e:
                checks.append({
                    'name': f'API Signature: {indicator_class.__name__}',
                    'status': 'fail',
                    'message': f"Could not get signature for {indicator_class.__name__}: {e}",
                    'details': {'error': str(e)}
                })
        
        # Check for consistent parameter names
        if update_signatures:
            unique_signatures = set(update_signatures.values())
            is_consistent = len(unique_signatures) <= 2  # Allow some variation
            
            checks.append({
                'name': 'API Consistency',
                'status': 'pass' if is_consistent else 'warning',
                'message': f"Method signatures: {'✅ Consistent' if is_consistent else '⚠️ Variations found'}",
                'details': {
                    'signatures': update_signatures,
                    'unique_count': len(unique_signatures)
                }
            })
        
        return {
            'status': 'pass' if all(c['status'] in ['pass', 'warning'] for c in checks) else 'fail',
            'checks': checks
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if self.failed_checks > 0:
            recommendations.append("🔴 Fix failed validation checks before production deployment")
        
        if len(self.warnings) > 5:
            recommendations.append("🟡 Review and address validation warnings")
        
        # Specific recommendations based on results
        for category, result in self.results.items():
            if result.get('status') == 'fail':
                recommendations.append(f"🔧 Address issues in {category}")
        
        if not recommendations:
            recommendations.append("✅ System passes all validation checks - ready for production")
        
        return recommendations
    
    def _log_final_results(self, result: Dict[str, Any]):
        """Log final validation results"""
        summary = result['summary']
        
        logger.info("=" * 50)
        logger.info("🏁 RTAI VALIDATION COMPLETE")
        logger.info("=" * 50)
        
        if result['status'] == 'pass':
            logger.info(f"✅ VALIDATION PASSED: {summary['passed']}/{summary['total_checks']} checks")
        else:
            logger.error(f"❌ VALIDATION FAILED: {summary['failed']} failures, {summary['warnings']} warnings")
        
        logger.info(f"⏱️  Validation completed in {summary['validation_time']}")
        
        for recommendation in result['recommendations']:
            logger.info(recommendation)
        
        logger.info("=" * 50)


async def run_validation_suite() -> Dict[str, Any]:
    """Run the complete validation suite"""
    suite = ValidationSuite()
    return await suite.run_full_validation()


def quick_health_check() -> bool:
    """Quick health check for basic functionality"""
    try:
        # Test basic indicator creation
        ofi = OFI()
        ofi.update(100, 50000)  # trade_qty, current_price
        
        # Test state management
        store = create_state_store("memory")
        store.save_state("test", {"healthy": True})
        
        return True
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return False
