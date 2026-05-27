# Contributing to FlexLLM

Thank you for your interest in contributing to FlexLLM!
FlexLLM is an open-source project that provides hybrid CPU-GPU KV Cache management and efficient inference scheduling for large language models.
We welcome contributions of all kinds, including bug fixes, feature improvements, documentation, testing, and code refactoring.

This document provides clear guidelines to help you participate in the project.

### 1、Code of Conduct

- All contributors should respect each other, communicate openly, and maintain a positive and inclusive environment.
- Discrimination, harassment, or rude behavior will not be tolerated.
- By participating in this project, you agree to abide by this Code of Conduct.

### 2、Ways to Contribute

You can contribute to FlexLLM in the following ways:

- Report bugs and issues
- Propose new features or optimizations
- Improve documentation and comments
- Submit code for bug fixes or new features
- Help review pull requests
- Run experiments and provide feedback

### 3、Reporting Issues

When you create an issue, please include:
- A clear and concise title
- Detailed steps to reproduce the problem
- Expected behavior and actual behavior
- Environment information:
  - Python version
  - CUDA version
  - Operating system
  - GPU model
  - Project commit hash (if applicable)
- Logs, error messages, or screenshots

The more complete the information, the faster we can locate and solve the problem.

### 4、Submitting Pull Requests

Follow these steps to submit a pull request:

1. Fork the main repository to your GitHub account.
2. Clone your forked repository locally.
3. Create a new branch for your work.
   Use a meaningful name, such as:
   - feature/xxx
   - fix/xxx
   - refactor/xxx
4. Make your changes and ensure they follow the project's coding standards.
5. Write clear and standardized commit messages.
6. Push your branch to your fork.
7. Open a Pull Request to the main branch of the main repository.
8. In the PR description:
   - Explain what you changed
   - Link related issues
   - Describe how you tested your changes

### 5、Development Workflow

- Keep your fork synchronized with the upstream main branch.
- Create a new branch for each feature or bug fix.
- Do not commit directly to the main branch.
- Test your changes before submitting.
- Ensure all existing tests pass.
- Keep pull requests small and focused for easier review.

### 6、Coding Standards

To maintain code consistency and readability:
- Use 4 spaces for indentation (do not use tabs).
- Use English for all code, comments, variables, and functions.
- Class names use CamelCase.
- Functions, variables, and methods use snake_case.
- Add docstrings for all public classes and functions.
- Keep functions short and focused on a single function.
- Remove unused imports, variables, and code.
- Format code using black before submission.

### 7、Commit Message Rules

We use Conventional Commits to keep the Git history clean.
Common commit types:

- feat: new feature
- fix: bug fix
- docs: documentation update
- style: code format, no logic change
- refactor: code restructuring
- test: add or update tests
- perf: performance optimization
- ci: CI or workflow changes
- chore: other minor changes

Examples:

- feat: add dynamic backup policy in scheduler
- fix: correct block memory calculation in swapper
- docs: update user guide and examples
- refactor: simplify predictor module interface

### 8、Documentation

If you modify features, interfaces, or workflows:
- Update corresponding docstrings
- Update README.md if necessary
- Keep descriptions clear and easy to understand
- Provide examples where appropriate

Good documentation helps other users and developers use FlexLLM correctly.

### 9、License

By contributing to FlexLLM, you agree that your contributions will be licensed under the MIT License.
This allows others to freely use, modify, and distribute your contributions while preserving copyright notices.

Thank you for your support!
We appreciate every contribution that helps make FlexLLM better.

If you have questions, please open an issue or contact the maintainers.