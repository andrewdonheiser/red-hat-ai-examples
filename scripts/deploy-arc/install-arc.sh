#!/bin/bash
set -e

NAMESPACE="arc-systems"
RUNNER_NAMESPACE="arc-runners"
GITHUB_CONFIG_URL="https://github.com/YOUR_ORG/red-hat-ai-examples"

echo "=== Installing Actions Runner Controller ==="

# Create namespaces
oc new-project ${NAMESPACE} || oc project ${NAMESPACE}
oc new-project ${RUNNER_NAMESPACE} || true

# Install ARC controller
helm install arc \
  --namespace ${NAMESPACE} \
  --create-namespace \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller

echo "Waiting for ARC controller to be ready..."
oc wait --for=condition=available deployment/arc-gha-runner-scale-set-controller \
  -n ${NAMESPACE} --timeout=300s

echo "=== ARC Controller installed successfully ==="
echo ""
echo "Next: Run ./create-runner-scale-set.sh to create GPU runners"
