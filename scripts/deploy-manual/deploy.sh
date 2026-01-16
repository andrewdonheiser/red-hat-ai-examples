#!/bin/bash
set -e

NAMESPACE="github-runners"

echo "=== Deploying GitHub Actions Runner ==="

# Create namespace if it doesn't exist
oc new-project ${NAMESPACE} 2>/dev/null || oc project ${NAMESPACE}

# Check for required secrets
if ! oc get secret github-runner-token -n ${NAMESPACE} &>/dev/null; then
  echo "ERROR: Secret 'github-runner-token' not found!"
  echo ""
  echo "Create it with:"
  echo "  oc create secret generic github-runner-token \\"
  echo "    --from-literal=token=YOUR_GITHUB_TOKEN \\"
  echo "    -n ${NAMESPACE}"
  exit 1
fi

# Apply deployment
oc apply -f github-runner-deployment.yaml

echo ""
echo "Waiting for runner pod to be ready..."
oc wait --for=condition=available deployment/github-actions-runner \
  -n ${NAMESPACE} --timeout=300s

echo ""
echo "=== Deployment successful ==="
echo ""
echo "Check pod status:"
echo "  oc get pods -n ${NAMESPACE}"
echo ""
echo "View logs:"
echo "  oc logs -f deployment/github-actions-runner -n ${NAMESPACE}"
echo ""
echo "Verify in GitHub:"
echo "  Go to repo Settings → Actions → Runners"
echo "  You should see a runner named like 'github-actions-runner-xxxxx'"
