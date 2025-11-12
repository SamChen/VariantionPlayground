# Project Title

## Description

## Installation

**Prerequisites:**

*   Git for cloning the repository.
*   For Method 1: [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) must be installed.
*   For Method 2: [Docker](https://docs.docker.com/get-docker/) and [envd](https://envd.tensorchord.ai/guide/getting-started.html#setup-your-first-envd-environment-in-3-minutes) must be installed.

### Method 1: Using Conda Virtual Environment

This method sets up a local Python environment on your machine.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/SamChen/VariantionPlayground.git
    cd VariantionPlayground
    ```

2.  **Create and activate the Conda environment:**

    ```bash
    conda create -n VariantionPlayground python=3.11
    conda activate VariantionPlayground
    ```

3.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

### Method 2: Using Docker with envd

This method uses `envd` to build a containerized development environment, ensuring consistency across different machines. This is the recommended approach for reproducibility.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/SamChen/VariantionPlayground.git
    cd VariantionPlayground
    ```

2.  **Build and run the environment:**

    From the root of the project directory, run the following command. `envd` will read the `build.envd` file, build the Docker image with all dependencies, and start the container.

    ```bash
    envd up
    ```

## Usage

**Example Usage:**

```bash
cd notebooks/pTau217/
bash run_addtional_simulation.sh 
```

## Code
Source for data simulation are stored under `src/simulation.py`
