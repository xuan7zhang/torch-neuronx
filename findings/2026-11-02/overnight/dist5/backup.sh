#!/bin/bash
# Copy report + all overnight results to the findings worktree, redact host info, commit, push to fork.
set -e
SRC=~/torch-neuronx/agent_space; D=~/tnx-findings/findings/2026-11-02
rsync -a --exclude '__pycache__' --exclude '*.pyc' --exclude '*.distcp' --exclude '.metadata' --exclude '*.pt' --exclude 'rec_oom_gradual.log' $SRC/overnight/ $D/overnight/
cp $SRC/overnight_report.md $D/README.md; cp $SRC/overnight_task.md $SRC/overnight_dist_task.md $D/
grep -rIl -E "<INSTANCE_ID>|<HOSTNAME>|172\.31\.22\.78" $D | xargs -r sed -i -e 's/<INSTANCE_ID>/<INSTANCE_ID>/g' -e 's/<HOSTNAME>\.sa-east-1\.compute\.internal/<HOSTNAME>/g' -e 's/<HOSTNAME>/<HOSTNAME>/g' -e 's/172\.31\.22\.78/<HOST_IP>/g'
if grep -rIl -E "<INSTANCE_ID>|<HOSTNAME>|172\.31\.22\.78|AQ\.Ab8|AKIA[0-9A-Z]{16}" $D; then echo "REDACTION FAILED"; exit 1; fi
cd ~/tnx-findings
git add -A findings/2026-11-02
git -c user.name=xuan7zhang -c user.email=zxq5706346@gmail.com commit -q --no-verify -m "${1:-Update distributed findings and logs}" || echo "nothing to commit"
git push -q fork overnight-findings-2026-11-02
git log --oneline -1
