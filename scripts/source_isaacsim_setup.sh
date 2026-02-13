# Detect script directory (works in both bash and zsh)
if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
elif [ -n "${ZSH_VERSION}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${(%):-%x}" )" &> /dev/null && pwd )
fi

CONDA_ENV_NAME="hssim"
echo "conda environment name is set to: $CONDA_ENV_NAME"

source ${SCRIPT_DIR}/source_common.sh
source ${CONDA_ROOT}/bin/activate $CONDA_ENV_NAME
export OMNI_KIT_ACCEPT_EULA=1

# Make IsaacLab importable
# IsaacLab is installed by `scripts/setup_isaacsim.sh` into $WORKSPACE_DIR/IsaacLab.
ISAACLAB_PATH=${ISAACLAB_PATH:-"${WORKSPACE_DIR}/IsaacLab"}
export ISAACLAB_PATH

if [ -d "${ISAACLAB_PATH}/source/isaaclab" ]; then
    export PYTHONPATH="${ISAACLAB_PATH}/source/isaaclab:${PYTHONPATH}"
else
    echo "WARNING: IsaacLab not found at: ${ISAACLAB_PATH}/source/isaaclab"
    echo "         Run: scripts/setup_isaacsim.sh"
fi
