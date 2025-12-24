"""
Text cleaning utilities.
"""

import re
import string


def clean_text(text):
    """
    Clean and normalize text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters
    text = remove_special_characters(text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def remove_special_characters(text):
    """
    Remove special characters from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text without special characters
    """
    # Keep only alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    return text
