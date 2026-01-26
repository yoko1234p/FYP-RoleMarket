"""
Setup script for FYP-RoleMarket project
Allows the project to be installed as a package for proper imports in Streamlit Cloud
"""

from setuptools import setup, find_packages

setup(
    name="fyp-rolemarket",
    version="1.0.0",
    description="AI-Driven Character IP Design & Demand Forecasting System",
    author="ToyzeroPlus FYP Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        # Core dependencies are listed in requirements.txt
    ],
    include_package_data=True,
)
