
# Table of Contents

1. [Installation](#org0b1e512)
2. [Data production](#orga49b706)
    1. [Skimming: install `yaml-cpp` dependency](#org18f77dd)
    2. [Skimming: run](#org5746685)
    3. [Data sources](#orgfc86ff0)
3. [Reconstruction Chain](#org0bc224d)
    1. [Cluster Size Studies](#orgc33e2a6)
4. [Event Visualization](#org44a4071)
    1. [Setup](#orgc4a7ba6)
    2. [Setup in local browser](#org288a700)
        1. [1)](#orgb4acfa6)
        2. [2)](#orgbafd8a5)
    3. [Visualization in local browser](#org3f41a7d)
    4. [Visualization with OpenShift OKD4](#org4164d71)
        1. [Additional information](#orgab38b90)
5. [Cluster Radii Studies](#orga2b91d9)
6. [Merging `plotly` and `bokeh` with `flask`](#org988e7e8)
    1. [Introduction](#org05cd898)
    2. [Flask embedding](#org9c3d76f)
        1. [Note](#org1d744c2)
7. [Producing `tikz` standalone pictures](#org0cc155d)

![img](https://img.shields.io/github/license/bfonta/bye_splits.svg "license")

This repository reproduces the CMS HGCAL L1 Stage2 reconstruction chain in Python for quick testing. It can generate an event visualization app. It was originally used for understanding and fixing the observed cluster splitting.

<a id="org0b1e512"></a>

# Installation

    # setup conda environment
    create -n <EnvName> python=3 pandas uproot pytables h5py
    conda activate <EnvName>
    
    # setup a ssh key if not yet done and clone the repository
    git clone git@github.com:bfonta/bye_splits.git
    # enforce git hooks locally (required for development)
    git config core.hooksPath .githooks

<a id="orga49b706"></a>

# Data production

<a id="org18f77dd"></a>

## Skimming: install `yaml-cpp` dependency

To make the size of the files more manageable, a skimming step was implemented in `C++`. It depends on the `yaml-cpp` package ([source](https://github.com/jbeder/yaml-cpp),  [release used](https://github.com/jbeder/yaml-cpp/releases/tag/yaml-cpp-0.7.0): version `0.7.0`). The instructions on the `README` page of the project are a bit cryptic. Below follows a step-by-step guide:

    # 1) Download the 0.7.0 '.zip' release
    # 2) Unzip it
    unzip yaml-cpp-yaml-cpp-0.7.0.zip
    # 3) The package uses CMake for compilation. To avoid cluttering the same folder with many CMake-related files, create a new folder and build the project there
    cd yaml-cpp-yaml-cpp-0.7.0
    mkdir build
    cd build
    # 3.5) This command might be necessary if your cmake executable is set to the default for the LLR machines, which is an earlier version than required.
    module load /opt/exp_soft/vo.llr.in2p3.fr/modulefiles_el7/cmake/3.22.3
    # 4) Compile a static library (the flag removes a -Werror issue)
    cmake -DYAML_CPP_BUILD_TESTS=OFF ..
    # 5) Build the package
    cmake --build .
    # 6) Verify the library 'libyaml-cpp.a' was created
    ls -l
    # 7) Check the package was correctly installed by compiling a test example (this assumes you have g++ installed):
    g++ bye_splits/tests/test_yaml_cpp.cc -I <installation_path_yaml_cpp>/yaml-cpp-yaml-cpp-0.7.0/include/ -L <installation_path_yaml_cpp>/yaml-cpp-yaml-cpp-0.7.0/build/ -std=c++11 -lyaml-cpp -o test_yaml_cpp.o
    # 8) Run the executable
    ./test_yaml_cpp.o

The above should print the contents stored in `bye_splits/tests/params.yaml`.
Occasionally the following error message is printed: ``relocation R_X86_64_32 against symbol `_ZTVN4YAML9ExceptionE' can not be used when making a PIE object; recompile with -fPIE``. This is currently not understood but removing the `yaml-cpp-yaml-cpp-0.7.0` folder (`rm -rf`) and running the above from scratch solves the issue.

<a id="org5746685"></a>

## Skimming: run

To run the skimming step, you will need to compile the `C++` files stored under `bye_splits/production`. Run `make` from the top repository directory.

To skim a ROOT file, you can run either:

    ./produce.exe --inpath </full/path/to/file.root>

or simply ```./produce.exe``` in which case it will skim the file listed under ```skim/infilePath``` in ```config.yaml```. In both cases the script infers the particle type from the path name, so the files and/or directories should be named accordingly. The base name for the output file can be chosen in the configuration file, as well as the base output directory—the final output file will go to ```/<dir>/<particles>/new_skims/```.

The above skims the input files, applying:

- &Delta; R matching between the generated particle and the clusters (currently works only for one generated particle, and throws an assertion error otherwise)
- &Delta; R matching between the generated particle and the trigger cells, set in the ```skim``` section of the config file. Default is set to be twice the &Delta; R between gen and cl3ds.
- a trigger cell energy threshold cut
- unconverted photon cut
- positive endcap cut (reduces the amount of data processed by a factor of 2)
- type conversion for `uproot` usage at later steps

### HTCondor Jobs

You can skim several files at once by sending them as jobs to [HTCondor](https://htcondor.readthedocs.io/en/latest/index.html). The parameters for job submission should be set in ```config.yaml``` under ```job```. In addition to general parameters (user, proxy, queue, etc), you should also set ```script``` as the full path to the script you wish to submit, and the boolean ```read_dir```, explained shortly. The following should be set for each particle type:

- ```submit_dir```: the base directory where your submission scripts will be written to. If ```read_dir``` is ```True```, it assumes this contains a sub directory ```ntuples/``` where your ```ROOT``` files are located.
- ```files```: a full path to a ```.txt``` file containing the paths to your ```ROOT``` files (line-delimited); will be read if ```read_dir``` is ```False```.
- ```files_per_batch```: Multiple "batch files" will be written to ```<submit_dir>/batches```, containing this number of file paths to your ```ROOT``` files (again, line-delimted). These batch files are queued by HTCondor such that one condor job contains this number of input files.

This requires that your script is structured in the following way:

    batch_file=$1
    while read -r line; do
    <something> $line
    done <$batch_file

where ```<something>``` is a command that accepts a ```/full/path/to/file.root``` as an argument. A full (though short) example can be viewed in [skim_pu_multicluster.sh](bye_splits/production/submit_scripts/skim_pu_multicluster.sh).

Finally, you run ```python bye_splits/production/submit_scripts/job_submit.py``` to launch the jobs.

A few notes:

- All submission related scripts will be created using the basename of your passed script, such that changing the script you run will not overwrite previous files and they will be easier to find.
- In addition, version numbers will be appended to each script every time you run, so you are free to test different configurations without worry of overwriting existing scripts.

<a id="orgfc86ff0"></a>

## Data sources

This framework relies on photon-, electron- and pion-gun samples produced via CRAB. The most up to date versions are currently stored under:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">

<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">Photons (PU0)</td>
<td class="org-left"><code>/dpm/in2p3.fr/home/cms/trivcat/store/user/lportale/DoublePhoton_FlatPt-1To100/GammaGun_Pt1_100_PU0_HLTSummer20ReRECOMiniAOD_2210_BCSTC-FE-studies_v3-29-1_realbcstc4/221025_153226/0000/</code></td>
</tr>

<tr>
<td class="org-left">Electrons (PU0)</td>
<td class="org-left"><code>/dpm/in2p3.fr/home/cms/trivcat/store/user/lportale/DoubleElectron_FlatPt-1To100/ElectronGun_Pt1_100_PU200_HLTSummer20ReRECOMiniAOD_2210_BCSTC-FE-studies_v3-29-1_realbcstc4/221102_102633/0000/</code></td>
</tr>

<tr>
<td class="org-left">Pions (PU0)</td>
<td class="org-left"><code>/dpm/in2p3.fr/home/cms/trivcat/store/user/lportale/SinglePion_PT0to200/SinglePion_Pt0_200_PU0_HLTSummer20ReRECOMiniAOD_2210_BCSTC-FE-studies_v3-29-1_realbcstc4/221102_103211/0000</code></td>
</tr>

<tr>
<td class="org-left">Photons (PU200)</td>
<td class="org-left"><code>/data_CMS/cms/ehle/L1HGCAL/PU200/photons/ntuples</code></td>
</tr>

<tr>
<td class="org-left">Electrons (PU200)</td>
<td class="org-left"><code>/data_CMS/cms/ehle/L1HGCAL/PU200/electrons/ntuples</code></td>
</tr>
</tbody>
</table>

The `PU0` files above were merged and are stored under `/data_CMS/cms/alves/L1HGCAL/`, accessible to LLR users and under `/eos/user/b/bfontana/FPGAs/new_algos/`, accessible to all lxplus and LLR users. The latter is used since it is well interfaced with CERN services. The `PU200` files were merged and stored under `/eos/user/i/iehle/data/PU200/<particle>/`.

<a id="org0bc224d"></a>

# Reconstruction Chain

The reconstruction chain is implemented in Python. To run it:

    python bye_splits/run_chain.py

where one can use the `-h` flag to visualize available options. To use the steps separately in your own script use the functions defined under `bye_splits/tasks/`, just as done in the `iterative_optimization.py` script.

For plotting results as a function of the optimization trigger cell parameter:

    python plot/meta_algorithm.py

The above will create `html` files with interactive outputs.

<a id="orgc33e2a6"></a>

## Cluster Size Studies

The script `bye_splits/scripts/cluster_size.py` reads a configuration file `bye_splits/scripts/cl_size_params.yaml` and runs the Reconstruction Chain on the `.root` inside corresponding to the chosen particle, where the clustering step is repeated for a range of cluster radii that is specified in the parameter file under `cl_size: Coeffs`.

The most convenient way of running the study is to do:

    bash run_cluster_size.sh <username>

where `<username>` is your lxplus username, creating `.hdf5` files containing Pandas DFs containing cluster properties (notably energy, eta, phi) and associated gen-level particle information for each radius. The bash script acts as a wrapper for the python script, setting a few options that are convenient for the cluster size studies that are not the default options for the general reconstruction chain. As of now, the output `.hdf5` files will be written to your local directory using the structure:

    ├── /<base_dir>
    │            ├── out
    │            ├── data
    │            │   ├──new_algos

with the files ending up in `new_algos/`. Currently working on implementing an option to send the files directly to your `eos/` directory, assuming the structure:

    ├── /eos/user/<first_letter>/<username>
    │                                   ├── out
    │                                   ├── data
    │                                   │   ├──PU0
    │                                   │   │   ├──electrons
    │                                   │   │   ├──photons
    │                                   │   │   ├──pions
    │                                   │   ├──PU200
    │                                   │   │   ├──electrons
    │                                   │   │   ├──photons
    │                                   │   │   ├──pions

<a id="org44a4071"></a>

# Event Visualization

The repository creates two web apps that can be visualized in a browser. The code is stored under `bye_splits/plot`.

<a id="orgc4a7ba6"></a>

## Setup

Please install the following from within the `conda` environment you should have already created:

    conda install -c conda-forge pyarrow
    #if the above fails: python -m pip install pyarrow
    python3 -m pip install --upgrade pip setuptools #to avoid annoying "Setuptools is replacing distutils." warning

<a id="org288a700"></a>

## Setup in local browser

Since browser usage directly in the server will necessarily be slow, we can:

<a id="orgb4acfa6"></a>

### 1)

Use LLR's intranet at `llruicms01.in2p3.fr:<port>/display`

<a id="orgbafd8a5"></a>

### 2)

Forward it to our local machines via `ssh`. To establish a connection between the local machine and the remote `llruicms01` server, passing by the gate, use:

    ssh -L <port>:llruicms01.in2p3.fr:<port> -N <llr_username>@llrgate01.in2p3.fr
    # for instance: ssh -L 8080:lruicms01.in2p3.fr:8080 -N alves@llrgate01.in2p3.fr

The two ports do not have to be the same, but it avoids possible confusion. Leave the terminal open and running (it will not produce any output).

<a id="org3f41a7d"></a>

## Visualization in local browser

In a new terminal window go to the `llruicms01` mahcines and launch one of the apps, for instance:

    bokeh serve bye_splits/plot/display/ --address llruicms01.in2p3.fr --port <port>  --allow-websocket-origin=localhost:<port>
    # if visualizing directly at LLR: --allow-websocket-origin=llruicms01.in2p3.fr:<port>

This uses the server-creation capabilities of `bokeh`, a `python` package for interactive visualization ([docs](https://docs.bokeh.org/en/latest/index.html)). Note the port number must match. For further customisation of `bokeh serve` see [the serve documentation](https://docs.bokeh.org/en/latest/docs/reference/command/subcommands/serve.html).
The above command should give access to the visualization under `http://localhost:8080/display`. For debugging, just run `python bye_splits/plot/display/main.py`  and see that no errors are raised.

<a id="org4164d71"></a>

## Visualization with OpenShift OKD4

We use the [S2I](https://docs.openshift.com/container-platform/3.11/creating_images/s2i.html) (Source to Image) service via CERN's [PaaS](https://paas.docs.cern.ch/) (Platform-as-a-Service) using OpenShift to deploy and host web apps in the CERN computing environment [here](https://paas.cern.ch/). There are three ways to deploys such an app: S2I represents the easiest (but less flexible) of the three; instructions [here](https://paas.docs.cern.ch/2._Deploy_Applications/Deploy_From_Git_Repository/2-deploy-s2i-app/). It effectively abstracts away the need for Dockerfiles.

We will use S2I's simplest configuration possible under `app.sh`. The image is created alongside the packages specified in `requirements.txt`. The two latter definitions are documented [here](https://github.com/kubesphere/s2i-python-container/blob/master/2.7/README.md#source-repository-layout).

We are currently running a pod at <https://viz2-hgcal-event-display.app.cern.ch/>. The port being served by `bokeh` in `app.sh` must match the one the pod is listening to, specified at configuration time before deployment in the [OpenShift management console](https://paas.cern.ch/) at CERN. The [network visibility](https://paas.docs.cern.ch/5._Exposing_The_Application/2-network-visibility/) was also updated to allow access from outside the CERN network.

<a id="orgab38b90"></a>

### Additional information

- [What is a pod](https://cloud.google.com/kubernetes-engine/docs/concepts/pod)?
- [How to mount `/eos` at CERN so that it is accessible by a pod?](https://paas.docs.cern.ch/3._Storage/eos/)

<a id="orga2b91d9"></a>

# Cluster Radii Studies

A DashApp has been built to interactively explore the effect of cluster size on various cluster properties, which is currently hosted at <https://bye-splits-app-hgcal-cl-size-studies.app.cern.ch/>.
To run the app locally, you can do:

    bash run_cluster_app.sh <username>

where `<username>` is your lxplus username. The app reads the configuration file `bye_splits/plot/display_clusters/config.yaml` and assumes that you have a directory structure equivalent to the directories described in the cluster size step (depending on your choice of ```local```).

It performs the necessary analysis on the files in the specified directory to generate the data for each page, which are themselves written to files in this directory. In order to minimize duplication and greatly speed up the user experience, if one of these files does not exist in your own directory, it looks for it under the appropriate directories (listed in our Data Sources), where a large number of the possible files already exist. The same procedure is used for reading the generated cluster size files, so you can use the app without having had to run the study yourself.

<a id="org988e7e8"></a>

# Merging `plotly` and `bokeh` with `flask`

<a id="org05cd898"></a>

## Introduction

Flask is a python micro web framework to simplify web development. It is considered "micro" because it’s lightweight and only provides essential components.
Given that `plotly`'s dashboard framework, `dash`, runs on top of `flask`, and that `bokeh` can produce html components programatically (which can be embedded in a `flask` app), it should be possible to develop a `flask`-powered web app mixing these two plotting packages. Having a common web framework also simplifies future integration.

<a id="org9c3d76f"></a>

## Flask embedding

The embedding of bokeh and plotly plots within flask is currently demonstrated in `plot/join/app.py`. Two servers run: one from `flask` and the other from `bokeh`, so special care is required to ensure the browser where the app is being served listens to both ports. Listening to `flask`'s port only will cause the html `plot/join/templates/embed.html` to be rendered without bokeh plots.

<a id="org1d744c2"></a>

### Note

Running a server is required when more advanced callbacks are needed. Currently only `bokeh` has a server of its own; `plotly` simply creates an html block with all the required information. If not-so-simple callbacks are required for `plotly` plots, another port will have to be listened to.

<a id="org0cc155d"></a>

# Producing `tikz` standalone pictures

For the purpose of illustration, `tikz` standalone script have been included under `docs/tikz/`. To run them (taking `docs/tikz/flowchart.tex` as an example):

    cd docs/tikz/
    pdflatex -shell-escape flowchart.tex

The above should produce the `flowchart.svg` file. The code depends on `latex` and `pdf2svg`.
