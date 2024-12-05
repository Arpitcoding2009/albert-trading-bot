# Contributing to Albert Trading Bot

First off, thank you for considering contributing to Albert Trading Bot! It's people like you that make Albert such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include screenshots if possible

### Suggesting Enhancements

If you have a suggestion for the project, we'd love to hear it. Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* Use a clear and descriptive title
* Provide a step-by-step description of the suggested enhancement
* Provide specific examples to demonstrate the steps
* Describe the current behavior and explain which behavior you expected to see instead
* Explain why this enhancement would be useful

### Pull Requests

* Fill in the required template
* Do not include issue numbers in the PR title
* Include screenshots and animated GIFs in your pull request whenever possible
* Follow the Python style guide
* Include thoughtfully-worded, well-structured tests
* Document new code
* End all files with a newline

## Development Process

1. Fork the repo
2. Create a new branch from `main`
3. Make your changes
4. Run the tests
5. Push your changes to your fork
6. Submit a Pull Request

### Setting Up Development Environment

```bash
# Clone your fork
git clone https://github.com/your-username/albert-trading-bot.git

# Add upstream remote
git remote add upstream https://github.com/original/albert-trading-bot.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### Running Tests

```bash
pytest
```

### Code Style

We use:
* Black for code formatting
* isort for import sorting
* flake8 for style guide enforcement
* mypy for type checking

Run all style checks:
```bash
# Format code
black .
isort .

# Check style
flake8
mypy .
```

## Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

## Documentation

* Use docstrings for all public modules, functions, classes, and methods
* Follow the Google Python Style Guide for docstrings
* Keep README.md up to date

## Additional Notes

### Issue and Pull Request Labels

* `bug`: Something isn't working
* `enhancement`: New feature or request
* `documentation`: Improvements or additions to documentation
* `good first issue`: Good for newcomers
* `help wanted`: Extra attention is needed
* `question`: Further information is requested

## Recognition

Contributors who make significant improvements will be recognized in our README.md and on our website.

Thank you for contributing to Albert Trading Bot! 🚀
