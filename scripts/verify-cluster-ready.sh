#!/bin/bash
# Verify RHOAI ROSA cluster is ready for GitHub Actions runners
# Run this BEFORE deploying runners

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

CHECKS_PASSED=0
CHECKS_FAILED=0

check() {
    local name="$1"
    local cmd="$2"
    local expected="$3"

    echo -n "Checking $name... "
    if eval "$cmd" &>/dev/null; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((CHECKS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        echo -e "  ${YELLOW}Expected: $expected${NC}"
        ((CHECKS_FAILED++))
    fi
}

echo "============================================"
echo "RHOAI ROSA Cluster Readiness Check"
echo "============================================"
echo ""

echo "--- OpenShift Cluster ---"
check "Cluster accessible" \
    "oc whoami" \
    "Valid oc login session"

check "Cluster version" \
    "oc get clusterversion version -o jsonpath='{.status.desired.version}'" \
    "OpenShift 4.x installed"

echo ""
echo "--- RHOAI Operator ---"
check "RHOAI operator namespace" \
    "oc get namespace redhat-ods-operator" \
    "redhat-ods-operator namespace exists"

check "RHOAI operator running" \
    "oc get pods -n redhat-ods-operator -l name=rhods-operator --field-selector=status.phase=Running -o name | grep -q pod" \
    "RHOAI operator pod running"

check "RHOAI applications namespace" \
    "oc get namespace redhat-ods-applications" \
    "redhat-ods-applications namespace exists"

check "RHOAI dashboard accessible" \
    "oc get route -n redhat-ods-applications rhods-dashboard -o jsonpath='{.spec.host}'" \
    "RHOAI dashboard route exists"

echo ""
echo "--- NVIDIA GPU Operator ---"
check "GPU operator namespace" \
    "oc get namespace nvidia-gpu-operator" \
    "nvidia-gpu-operator namespace exists"

check "GPU operator pods running" \
    "oc get pods -n nvidia-gpu-operator --field-selector=status.phase=Running -o name | wc -l | grep -q '[1-9]'" \
    "At least 1 GPU operator pod running"

check "ClusterPolicy exists" \
    "oc get clusterpolicy gpu-cluster-policy" \
    "GPU ClusterPolicy configured"

echo ""
echo "--- GPU Nodes ---"
check "GPU nodes labeled" \
    "oc get nodes -l nvidia.com/gpu.product -o name | grep -q node" \
    "At least 1 node with nvidia.com/gpu.product label"

GPU_NODE=$(oc get nodes -l nvidia.com/gpu.product -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
if [ -n "$GPU_NODE" ]; then
    check "GPU allocatable on node $GPU_NODE" \
        "oc get node $GPU_NODE -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' | grep -q '[1-9]'" \
        "nvidia.com/gpu allocatable > 0"

    GPU_PRODUCT=$(oc get node $GPU_NODE -o jsonpath='{.metadata.labels.nvidia\.com/gpu\.product}' 2>/dev/null || echo "unknown")
    echo "  GPU Type: $GPU_PRODUCT"
fi

echo ""
echo "--- Node Feature Discovery ---"
check "NFD operator" \
    "oc get pods -n openshift-nfd --field-selector=status.phase=Running -o name 2>/dev/null | grep -q pod || oc get pods -n nvidia-gpu-operator -l app=nfd --field-selector=status.phase=Running -o name | grep -q pod" \
    "NFD pods running"

echo ""
echo "--- Permissions Check ---"
check "Can create namespaces" \
    "oc auth can-i create namespace" \
    "Cluster admin or namespace creation permission"

check "Can create deployments" \
    "oc auth can-i create deployment -n default" \
    "Can create deployments"

echo ""
echo "============================================"
echo "SUMMARY"
echo "============================================"
echo -e "Passed: ${GREEN}$CHECKS_PASSED${NC}"
echo -e "Failed: ${RED}$CHECKS_FAILED${NC}"
echo ""

if [ $CHECKS_FAILED -gt 0 ]; then
    echo -e "${RED}Cluster is NOT ready for runner deployment.${NC}"
    echo "Please resolve the failed checks before proceeding."
    exit 1
else
    echo -e "${GREEN}Cluster is ready for runner deployment!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Create github-runners namespace: oc new-project github-runners"
    echo "  2. Create GitHub token secret (see Prerequisites section)"
    echo "  3. Deploy runners using scripts/deploy-arc/ or scripts/deploy-manual/"
    exit 0
fi
