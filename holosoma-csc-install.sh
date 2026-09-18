MUJOCO_WARP_COMMIT="09ec1da"
SCRATCH_DIR="/scratch/project_2020755"
PROJAPPL_DIR="/projappl/project_2020755"

cd $SCRATCH_DIR

export PIP_CACHE_DIR="$SCRATCH_DIR/pip-cache"
mkdir -p $PIP_CACHE_DIR

# Install Holosoma
module load python-data/3.10-17.04

python -m venv --system-site-packages holosoma-venv
source holosoma-venv/bin/activate
pip install -e "$PROJAPPL_DIR/holosoma/src/holosoma[unitree]"

# Install MjWarp
git clone https://github.com/google-deepmind/mujoco_warp.git $SCRATCH_DIR/mujoco_warp && \
    git -C $SCRATCH_DIR/mujoco_warp checkout ${MUJOCO_WARP_COMMIT}

pip install -e $SCRATCH_DIR/mujoco_warp[dev,cuda]
