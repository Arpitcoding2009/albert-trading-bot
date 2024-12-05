import os
import subprocess
import json

class DeploymentSetup:
    def __init__(self, project_path, github_token):
        self.project_path = project_path
        self.github_token = github_token

    def init_git_repo(self):
        """Initialize git repository"""
        os.chdir(self.project_path)
        subprocess.run(["git", "init"], check=True)
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit for Albert Quantum Trading Bot"], check=True)

    def create_gitignore(self):
        """Create .gitignore file"""
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class

# Environment files
.env
*.env

# Sensitive configurations
*.key
*.pem

# Deployment files
.oci/

# Logs
*.log

# Virtual environments
venv/
env/
.venv/

# IDE settings
.vscode/
.idea/

# Secrets
secrets.yml
"""
        with open(os.path.join(self.project_path, ".gitignore"), "w") as f:
            f.write(gitignore_content)

    def create_readme(self):
        """Create README.md"""
        readme_content = """# Albert Quantum Trading Bot

## Overview
AI-powered cryptocurrency trading platform deployed on Oracle Cloud.

## Deployment
Automated deployment using GitHub Actions and Oracle Cloud Infrastructure.

## Setup
1. Configure Oracle Cloud credentials
2. Set up GitHub Secrets
3. Push to main branch to trigger deployment

## Technologies
- Python 3.11
- FastAPI
- Docker
- Oracle Cloud
- GitHub Actions
"""
        with open(os.path.join(self.project_path, "README.md"), "w") as f:
            f.write(readme_content)

    def setup_github_remote(self, repo_name):
        """Setup GitHub remote repository"""
        remote_url = f"https://github.com/arpit/{repo_name}.git"
        subprocess.run(["git", "remote", "add", "origin", remote_url], check=True)

    def push_to_github(self):
        """Push code to GitHub"""
        subprocess.run(["git", "branch", "-M", "main"], check=True)
        subprocess.run(["git", "push", "-u", "origin", "main"], check=True)

    def generate_deployment_config(self):
        """Generate deployment configuration"""
        config = {
            "project_name": "Albert Quantum Trading Bot",
            "python_version": "3.11",
            "deployment_platform": "Oracle Cloud",
            "ci_cd_tool": "GitHub Actions"
        }
        
        with open(os.path.join(self.project_path, "deployment_config.json"), "w") as f:
            json.dump(config, f, indent=4)

def main():
    project_path = r"c:/Users/arpit/CascadeProjects/crypto_trading_bot"
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise ValueError("Please set the GITHUB_TOKEN environment variable")
    
    setup = DeploymentSetup(project_path, github_token)
    
    try:
        setup.init_git_repo()
        setup.create_gitignore()
        setup.create_readme()
        setup.setup_github_remote("albert-quantum-trading-bot")
        setup.generate_deployment_config()
        setup.push_to_github()
        
        print("Deployment setup completed successfully!")
    except Exception as e:
        print(f"Error during deployment setup: {e}")

if __name__ == "__main__":
    main()
