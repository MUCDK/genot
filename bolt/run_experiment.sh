. bolt/setup.sh

# log git branch and revision.
git config --global --add safe.directory /mnt/task_runtime
branch=$(git rev-parse --abbrev-ref HEAD)
rev=$(git rev-parse HEAD)
echo $branch $rev > ${BOLT_ARTIFACT_DIR}/git_branch_and_rev

# get credential to copy checkpoint(s)
if  [ ! -z "${BOLT_TASK_ID_CREDENTIAL}" ]; then
  echo "Ask permission to read artifacts from ${BOLT_TASK_ID_CREDENTIAL}"
  eval `bolt task get-credentials ${BOLT_TASK_ID_CREDENTIAL}`

fi

# restore credential, might not be necessary.
eval `bolt task get-credentials $TASK_ID`

python bolt/main.py \
  --workdir=${BOLT_ARTIFACT_DIR} \
  --config="/mnt/task_runtime/config_for_bolt_machine.py" \
  --datadir=${TMPDIR}/datasets/ \
  $MAIN_ARGS