## conda build from source
VENV=python310venv;
conda config --set anaconda_upload no;
# output_folder=/home/jiaqigu/pkgs/miniforge3/envs/python310venv/conda-bld/torchonn_0.0.8;
output_folder=${CONDA-"/home/jiaqigu/pkgs/miniforge3"}/envs/${VENV}/conda-bld/torchonn_0.0.8;
echo Output folder: ${output_folder};
echo "rm -rf ${output_folder}";
rm -rf "${output_folder}";
echo "mkdir -p ${output_folder}";
mkdir -p "${output_folder}";
conda mambabuild . --no-anaconda-upload --no-test --output-folder "${output_folder}" -c pytorch -c nvidia;
echo "Finished conda mambabuild";


## conda local installation
local_channel="${output_folder}";
# mamba install -y -c "file://${local_channel}" pytorch==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia;
mamba install -y -n "${VENV}" -c local torchonn
echo "Finished mamba install";

