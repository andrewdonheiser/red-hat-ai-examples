#!/bin/bash

NAMESPACE="${1:-github-runners}"

echo "=== GitHub Runner Verification ==="
echo ""

# Check pods
echo "1. Pod Status:"
oc get pods -n ${NAMESPACE} -l app=github-runner
echo ""

# Check GPU allocation
echo "2. GPU Allocation:"
POD=$(oc get pods -n ${NAMESPACE} -l app=github-runner -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ -n "$POD" ]; then
  oc exec -n ${NAMESPACE} ${POD} -- nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
  echo "  No runner pod found"
fi
echo ""

# Check runner logs
echo "3. Recent Logs:"
if [ -n "$POD" ]; then
  oc logs -n ${NAMESPACE} ${POD} --tail=20
fi
echo ""

# Check events
echo "4. Recent Events:"
oc get events -n ${NAMESPACE} --sort-by='.lastTimestamp' | tail -10
echo ""

echo "=== Verification Complete ==="
