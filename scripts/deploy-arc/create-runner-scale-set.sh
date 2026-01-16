#!/bin/bash
set -e

RUNNER_NAMESPACE="arc-runners"
GITHUB_CONFIG_URL="https://github.com/YOUR_ORG/red-hat-ai-examples"
RUNNER_NAME="rhoai-gpu-runner"

# Prompt for GitHub PAT if not set
if [ -z "$GITHUB_PAT" ]; then
  echo "Enter your GitHub Personal Access Token (with repo and admin:org scopes):"
  read -s GITHUB_PAT
fi

echo "=== Creating GPU Runner Scale Set ==="

# Create runner scale set with GPU support
helm install ${RUNNER_NAME} \
  --namespace ${RUNNER_NAMESPACE} \
  --create-namespace \
  --set githubConfigUrl="${GITHUB_CONFIG_URL}" \
  --set githubConfigSecret.github_token="${GITHUB_PAT}" \
  --set minRunners=1 \
  --set maxRunners=3 \
  --set runnerGroup="default" \
  --set template.spec.nodeSelector."nvidia\.com/gpu\.product"="NVIDIA-L40S" \
  --set template.spec.tolerations[0].key="nvidia.com/gpu" \
  --set template.spec.tolerations[0].operator="Exists" \
  --set template.spec.tolerations[0].effect="NoSchedule" \
  --set template.spec.containers[0].resources.limits."nvidia\.com/gpu"=1 \
  --set template.spec.containers[0].resources.limits.memory="64Gi" \
  --set template.spec.containers[0].resources.limits.cpu="16" \
  --set template.spec.containers[0].resources.requests."nvidia\.com/gpu"=1 \
  --set template.spec.containers[0].resources.requests.memory="32Gi" \
  --set template.spec.containers[0].resources.requests.cpu="8" \
  -f runner-values.yaml \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set

echo "=== Runner Scale Set created ==="
echo ""
echo "Check runner status:"
echo "  oc get pods -n ${RUNNER_NAMESPACE}"
echo ""
echo "View in GitHub:"
echo "  ${GITHUB_CONFIG_URL}/settings/actions/runners"
