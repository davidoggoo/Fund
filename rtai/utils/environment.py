"""
Environment security and validation for RTAI
"""
import os
import re
from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger


class EnvironmentValidator:
    """Validate environment security and configuration"""
    
    SENSITIVE_PATTERNS = [
        r'(?i)(api[_-]?key|secret|token|password|pwd)',
        r'(?i)(auth|bearer|credential)',
        r'(?i)(private[_-]?key|priv[_-]?key)',
        r'(?i)(telegram[_-]?bot[_-]?token)',
        r'(?i)(database[_-]?url|db[_-]?url)',
    ]
    
    REQUIRED_ENV_VARS = {
        'RTAI_ENV': ['DEV', 'STAGING', 'PROD'],
        'RTAI_LOG_LEVEL': ['DEBUG', 'INFO', 'WARNING', 'ERROR']
    }
    
    OPTIONAL_ENV_VARS = {
        'TELEGRAM_BOT_TOKEN': str,
        'TELEGRAM_CHAT_ID': str,
        'QUESTDB_HOST': str,
        'QUESTDB_PORT': int,
        'DATA_DIR': str,
        'LOG_DIR': str
    }
    
    def __init__(self):
        self.issues: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
        
    def validate_environment(self) -> Dict[str, Any]:
        """Comprehensive environment validation"""
        logger.info("🔒 Starting environment security validation...")
        
        self.issues.clear()
        self.warnings.clear()
        
        # Check required environment variables
        self._check_required_env_vars()
        
        # Check for exposed secrets
        self._check_exposed_secrets()
        
        # Validate file permissions
        self._check_file_permissions()
        
        # Check production safety
        self._check_production_safety()
        
        # Validate optional configuration
        self._check_optional_env_vars()
        
        result = {
            'valid': len(self.issues) == 0,
            'issues': self.issues,
            'warnings': self.warnings,
            'environment': os.getenv('RTAI_ENV', 'DEV'),
            'log_level': os.getenv('RTAI_LOG_LEVEL', 'INFO')
        }
        
        if result['valid']:
            logger.info("✅ Environment validation passed")
        else:
            logger.error(f"❌ Environment validation failed: {len(self.issues)} issues")
            
        return result
    
    def _check_required_env_vars(self):
        """Check required environment variables"""
        for var_name, allowed_values in self.REQUIRED_ENV_VARS.items():
            value = os.getenv(var_name)
            
            if not value:
                self.issues.append({
                    'type': 'missing_required_env',
                    'variable': var_name,
                    'message': f"Required environment variable {var_name} not set",
                    'suggestion': f"Set {var_name} to one of: {allowed_values}"
                })
            elif value not in allowed_values:
                self.issues.append({
                    'type': 'invalid_env_value',
                    'variable': var_name,
                    'value': value,
                    'message': f"{var_name}={value} is not valid",
                    'suggestion': f"Use one of: {allowed_values}"
                })
    
    def _check_optional_env_vars(self):
        """Check optional environment variables for type safety"""
        for var_name, expected_type in self.OPTIONAL_ENV_VARS.items():
            value = os.getenv(var_name)
            
            if value is None:
                continue  # Optional, skip if not set
                
            # Type validation
            if expected_type == int:
                try:
                    int(value)
                except ValueError:
                    self.warnings.append(f"{var_name}={value} should be integer")
            elif expected_type == str and not value.strip():
                self.warnings.append(f"{var_name} is empty")
    
    def _check_exposed_secrets(self):
        """Check for exposed secrets in various locations"""
        env = os.getenv('RTAI_ENV', 'DEV').upper()
        
        # In development mode, be less strict about secrets
        if env == 'DEV':
            # Only warn about obvious issues, don't fail
            for env_var, value in os.environ.items():
                if self._is_sensitive_name(env_var):
                    if self._is_exposed_secret(value):
                        self.warnings.append(f"Potentially exposed secret in {env_var} (dev mode)")
            
            # Check for .env file existence but don't fail
            if Path('.env').exists():
                self.warnings.append("Potential secret in .env (dev mode)")
        else:
            # In production, be strict
            for env_var, value in os.environ.items():
                if self._is_sensitive_name(env_var):
                    if self._is_exposed_secret(value):
                        self.issues.append({
                            'type': 'exposed_secret',
                            'location': f"environment:{env_var}",
                            'message': f"Potentially exposed secret in {env_var}",
                            'suggestion': "Use secure secret management"
                        })
            
            # Check common files for secrets
            secret_files = ['.env', '.env.local', 'config.ini', 'settings.py']
            for filename in secret_files:
                filepath = Path(filename)
                if filepath.exists():
                    self._scan_file_for_secrets(filepath)
    
    def _check_file_permissions(self):
        """Check critical file permissions"""
        sensitive_patterns = ['*.key', '*.pem', '.env*', 'config.*']
        
        for pattern in sensitive_patterns:
            for filepath in Path('.').glob(pattern):
                if filepath.is_file():
                    # On Windows, checking basic existence is sufficient
                    # On Unix systems, we could check permissions more thoroughly
                    stat = filepath.stat()
                    if stat.st_size == 0:
                        self.warnings.append(f"Empty sensitive file: {filepath}")
    
    def _check_production_safety(self):
        """Check production environment safety"""
        env = os.getenv('RTAI_ENV', 'DEV').upper()
        
        if env == 'PROD':
            # Production-specific checks
            if os.getenv('RTAI_LOG_LEVEL', 'INFO').upper() == 'DEBUG':
                self.warnings.append("DEBUG logging enabled in production")
                
            # Check for development files
            dev_files = ['debug.py', 'test_*.py', 'development.ini']
            for pattern in dev_files:
                for filepath in Path('.').glob(pattern):
                    if filepath.is_file():
                        self.warnings.append(f"Development file in production: {filepath}")
        
        elif env == 'DEV':
            # Development-specific checks
            if not os.getenv('TELEGRAM_BOT_TOKEN'):
                self.warnings.append("TELEGRAM_BOT_TOKEN not set (notifications disabled)")
    
    def _is_sensitive_name(self, name: str) -> bool:
        """Check if environment variable name suggests sensitive data"""
        name_lower = name.lower()
        for pattern in self.SENSITIVE_PATTERNS:
            if re.search(pattern, name_lower):
                return True
        return False
    
    def _is_exposed_secret(self, value: str) -> bool:
        """Check if value looks like an exposed secret"""
        if not value or len(value) < 8:
            return False
            
        # Common patterns for exposed secrets
        exposed_patterns = [
            r'^(test|demo|example|sample)',  # Test values
            r'(123|abc|xxx|placeholder)',   # Placeholder values
            r'^[a-f0-9]{32,}$',            # Long hex strings (API keys)
            r'^[A-Za-z0-9+/]{20,}={0,2}$'  # Base64 encoded
        ]
        
        for pattern in exposed_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
                
        return False
    
    def _scan_file_for_secrets(self, filepath: Path):
        """Scan file for potential secrets"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    # Look for key=value patterns
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('\'"')
                        
                        if self._is_sensitive_name(key) and self._is_exposed_secret(value):
                            self.issues.append({
                                'type': 'secret_in_file',
                                'location': f"{filepath}:{line_num}",
                                'message': f"Potential secret in {filepath}",
                                'suggestion': "Move to secure environment variables"
                            })
                            
        except Exception as e:
            logger.warning(f"Could not scan {filepath}: {e}")


def validate_environment_security() -> Dict[str, Any]:
    """Quick environment security validation"""
    validator = EnvironmentValidator()
    return validator.validate_environment()


def ensure_environment_defaults():
    """Set secure environment defaults if not configured"""
    defaults = {
        'RTAI_ENV': 'DEV',
        'RTAI_LOG_LEVEL': 'INFO'
    }
    
    for key, default_value in defaults.items():
        if not os.getenv(key):
            os.environ[key] = default_value
            logger.info(f"🔧 Set environment default: {key}={default_value}")


def is_production() -> bool:
    """Check if running in production environment"""
    return os.getenv('RTAI_ENV', 'DEV').upper() == 'PROD'


def is_development() -> bool:
    """Check if running in development environment"""
    return os.getenv('RTAI_ENV', 'DEV').upper() == 'DEV'


def get_safe_env_summary() -> Dict[str, str]:
    """Get environment summary with sensitive values masked"""
    env_vars = {}
    
    for key, value in os.environ.items():
        if key.startswith('RTAI_'):
            if 'token' in key.lower() or 'secret' in key.lower() or 'key' in key.lower():
                env_vars[key] = '*' * min(len(value), 8) if value else 'unset'
            else:
                env_vars[key] = value
                
    return env_vars
