"""Utility Functions for FreeRangeZoo.

This module provides utility functions used throughout the FreeRangeZoo
package for caching, dataset management, logging conversion, and module discovery.

Key Components:
    Utilities:
        - caching: Hashing and memoization functions
        - dataset: Configuration dataset splitting
        - sql_logging: SQLAlchemy models for database logging
        - convert_sql_logs: SQL to CSV conversion utility
        - all_modules: Module discovery and registration
        - mem_restrict: Memory usage limiting

Example Usage:
    >>> from free_range_zoo.utils.caching import hash_tensor
    >>> from free_range_zoo.utils.dataset import train_test_split
"""

__all__ = []