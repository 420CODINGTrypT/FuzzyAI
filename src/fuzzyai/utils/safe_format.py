"""Safe string formatting utility to prevent template injection attacks."""

import re


def safe_format(template: str, **replacements: str) -> str:
    """
    Safely replace named placeholders in a template string.
    Uses simple string replacement instead of .format() to prevent
    template injection via malicious user input containing {} specifiers.
    
    Args:
        template: The template string with {placeholder} style placeholders
        **replacements: Key-value pairs to replace
        
    Returns:
        The formatted string with all placeholders replaced
        
    Example:
        >>> safe_format("Hello {name}!", name="World")
        'Hello World!'
    """
    result = template
    for key, value in replacements.items():
        placeholder = "{" + key + "}"
        result = result.replace(placeholder, str(value))
    return result


def sanitize_attack_id(attack_id: str) -> str:
    """
    Sanitize an attack ID to prevent path traversal.
    Only allows alphanumeric characters, hyphens, and underscores.
    
    Args:
        attack_id: The attack ID to sanitize
        
    Returns:
        Sanitized attack ID safe for use in file paths
    """
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', attack_id)
    if not sanitized:
        raise ValueError("Attack ID contains no valid characters after sanitization")
    return sanitized
