# PowerShell script to help set up GitHub secrets for OCI deployment

Write-Host "OCI Deployment Secrets Setup Helper" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

# OCI Core Credentials
Write-Host "`nStep 1: OCI Core Credentials" -ForegroundColor Cyan
Write-Host "You can find these in your OCI Console under Profile > User Settings > API Keys"
$OCI_USER_OCID = Read-Host "Enter your OCI User OCID"
$OCI_TENANCY_OCID = Read-Host "Enter your OCI Tenancy OCID"
$OCI_FINGERPRINT = Read-Host "Enter your API Key Fingerprint"
$OCI_REGION = Read-Host "Enter your OCI Region (e.g., us-ashburn-1)"

Write-Host "`nStep 2: Private Key" -ForegroundColor Cyan
Write-Host "Enter the path to your OCI API private key file (usually oci_api_key.pem)"
$private_key_path = Read-Host "Private Key Path"
$OCI_PRIVATE_KEY = Get-Content $private_key_path -Raw

# Docker Registry Details
Write-Host "`nStep 3: Container Registry Details" -ForegroundColor Cyan
Write-Host "You can find these in your OCI Console under Developer Services > Container Registry"
$OCI_REGISTRY = Read-Host "Enter your OCI Registry URL (e.g., [region].ocir.io/[tenancy-namespace]/[repo-name])"
$OCI_USER_NAME = Read-Host "Enter your OCI Username (usually [tenancy-namespace]/[username])"
$OCI_AUTH_TOKEN = Read-Host "Enter your Auth Token" -AsSecureString

# Kubernetes Cluster Details
Write-Host "`nStep 4: Kubernetes Cluster Details" -ForegroundColor Cyan
Write-Host "You can find the Cluster ID in your OCI Console under Developer Services > Kubernetes Clusters (OKE)"
$OCI_CLUSTER_ID = Read-Host "Enter your OKE Cluster OCID"

# Output instructions
Write-Host "`nSetup Complete!" -ForegroundColor Green
Write-Host "Please add the following secrets to your GitHub repository:" -ForegroundColor Yellow
Write-Host "Go to your GitHub repository > Settings > Secrets and variables > Actions > New repository secret"
Write-Host "`nAdd these secrets:"
Write-Host "1. OCI_USER_OCID: $OCI_USER_OCID"
Write-Host "2. OCI_TENANCY_OCID: $OCI_TENANCY_OCID"
Write-Host "3. OCI_FINGERPRINT: $OCI_FINGERPRINT"
Write-Host "4. OCI_PRIVATE_KEY: [Your private key content]"
Write-Host "5. OCI_REGION: $OCI_REGION"
Write-Host "6. OCI_REGISTRY: $OCI_REGISTRY"
Write-Host "7. OCI_USER_NAME: $OCI_USER_NAME"
Write-Host "8. OCI_AUTH_TOKEN: [Your auth token]"
Write-Host "9. OCI_CLUSTER_ID: $OCI_CLUSTER_ID"
