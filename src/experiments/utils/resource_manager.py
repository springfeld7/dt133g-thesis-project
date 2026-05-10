"""resource_manager.py

Utility module for hardware resource detection and management.

This module provides cross-platform helpers to determine available computing 
power, specifically designed to bridge the gap between local Windows development 
environments and Linux-based High-Performance Computing (HPC) clusters.
"""

import os


class ResourceManager:
    """
    A helper class to manage and detect hardware resources.

    Provides static methods to safely query CPU availability without
    triggering platform-specific errors.
    """

    @staticmethod
    def get_cpu_limit() -> int:
        """
        Determines the number of CPU cores available to the current process.

        On Linux clusters, this respects the CPU affinity mask (e.g., Slurm
        allocations). On Windows and macOS, it falls back to the system-wide
        CPU count.

        Returns:
            int: The number of usable CPU cores. Defaults to 1 if detection fails.
        """
        # Check for affinity (Linux/Cluster)
        if hasattr(os, "sched_getaffinity"):
            try:
                # 0 refers to the process ID of the current process
                return len(os.sched_getaffinity(0))
            except (AttributeError, OSError, NotImplementedError):
                # Fallback if affinity call fails despite attribute existing
                pass

        # Fallback (Windows/macOS/Legacy Linux)
        # os.cpu_count() can return None, so we ensure a minimum of 1
        return os.cpu_count() or 1


if __name__ == "__main__":
    # Quick test to verify local detection
    resources = ResourceManager()
    print(f"Detected usable CPUs: {resources.get_cpu_limit()}")
