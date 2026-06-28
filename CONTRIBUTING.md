# Contributing to Adversarial Fraud Detector

Thank you for your interest in contributing to this project! This document outlines the process for setting up your environment, making changes, and submitting pull requests.

## Setting Up Your Development Environment

1. **Fork and Clone**: Fork the repository to your own GitHub account and clone it locally.
   ```bash
   git clone https://github.com/your-username/adversarial-fraud-detector.git
   cd adversarial-fraud-detector
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Making Changes

- **Branch Naming**: 
  - `feature/your-feature-name`
  - `fix/your-fix-name`
  - `experiment/your-experiment-name`

- **Code Style**: 
  - Use `black` for formatting and `flake8` for linting.
  - Max line length is 100 characters.
  ```bash
  black src/ api/ tests/
  flake8 src/ api/ tests/ --max-line-length=100
  ```

- **Testing**:
  - Write unit tests for new features.
  - Ensure all tests pass before submitting a PR.
  ```bash
  pytest tests/unit/ -v
  ```

## Pull Request Process

When you are ready to submit your changes, open a Pull Request (PR) against the `main` branch. 

Please use the provided Pull Request template to describe your changes, outline any metrics that were impacted (e.g., AUC, Recall, F1), and confirm that your code passes all tests and styling checks.
