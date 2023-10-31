
# Table of Contents

1.  [Installation](#org0b1e512)
2.  [Data production](#dataprod)
    1.  [Skimming](#skim)
    2.  [Data sources](#sources)
    3.  [Job Submission](#job-submission)
3.  [Reconstruction Chain](#org0bc224d)
    1.  [Cluster Size Studies](#orgc33e2a6)
4.  [Event Visualization](#org44a4071)
    1.  [Setup](#orgc4a7ba6)
    2.  [Setup in local browser](#org288a700)
    3.  [Visualization in local browser](#org3f41a7d)
        1.  [2D display app](#orgb4acfa6)
        2.  [3D display app](#orgbafd8a5)
    4.  [Visualization with OpenShift OKD4](#org4164d71)
        1.  [Additional information](#orgab38b90)
5.  [Cluster Radii Studies](#orga2b91d9)
6.  [Merging `plotly` and `bokeh` with `flask`](#org988e7e8)
    1.  [Introduction](#org05cd898)
    2.  [Flask embedding](#org9c3d76f)
        1.  [Note](#org1d744c2)
7.  [Producing `tikz` standalone pictures](#org0cc155d)

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

The user could also use [Mamba](https://mamba.readthedocs.io/en/latest/index.html), a fast and robust package manager. It is fully compatible with conda packages and supports most of conda’s commands.


<a id="dataprod"></a>
# Data production

<a id="skim"></a>
## Skimming

To make the size of the files more manageable, a skimming step was implemented that relies on ```ROOT```'s ```RDataFrame```. Several cuts are applied, and additionally many type conversions are run for `uproot` usage at later steps. To run it:

```
python bye_splits/production/produce.py --nevents -1 --particles photons
```

where "-1" represents all events, and the input file is defined in ```config.yaml```. 

<a id="sources"></a>
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
<td class="org-left"><code>/eos/user/i/iehle/data/PU200/photons/ntuples</code></td>
</tr>

<tr>
<td class="org-left">Electrons (PU200)</td>
<td class="org-left"><code>/eos/user/i/iehle/data/PU200/electrons/ntuples</code></td>
</tr>
</tbody>
</table>

The `PU0` files above were merged and are stored under `/data_CMS/cms/alves/L1HGCAL/`, accessible to LLR users and under `/eos/user/b/bfontana/FPGAs/new_algos/`, accessible to all lxplus and LLR users. The latter is used since it is well interfaced with CERN services. The `PU200` files were merged and stored under `/eos/user/i/iehle/data/PU200/<particle>/`.

<a id="job-submission"></a>
## Job Submission

Job submission to HT Condor is handled through `bye_splits/production/submit_scripts/job_submit.py` using the section of `config.yaml` for its configuration. The configuration should include usual condor variables, i.e `user`, `proxy`, `queue`, and `local`, as well as a path to the `script` you would like to run on condor. The `arguments` sub-section should contain `key/value` pairs matching the expected arguments that `script` accepts. You can also pass arguments directly in the command line, in which case these values will superseed the defaults set in the configuration file. The new `Arguments` class in `bye_splits/utils/job_helpers.py` verifies that the passed arguments are accepted by `script` and that all required arguments have assigned values. For now, this requires that `script` uses `Arguments` to import its arguments, using a dictionary called `arg_dict`; an example can be found in `tests/submission/dummy_submit.py`. The variable that you would like to iterate over should be set in `iterOver` and its value should correspond to a `key` in the `arguments` sub-section whose value is a list containing the values the script should iterate over. It then contains a section for each particle type which should contain a `submit_dir`, i.e. the directory in which to read and write submission related files, and `args_per_batch` which can be any number between 1 and `len(arguments[<iterOver>])`. An example of the `job` configuration settings is as such:

```yaml
job:
    user: iehle
    proxy: ~/.t3/proxy.cert
    queue: short
    local: False
    script: /grid_mnt/vol_home/llr/cms/ehle/NewRepos/bye_splits/tests/submission/dummy_submit.py
    iterOver: gen_arg
    arguments:
        float_arg: 0.11
        str_arg: a_string
        gen_arg: [gen, 3.14, work, broke, 9, False, 12.9, hello]
    test:
        submit_dir: /home/llr/cms/ehle/NewRepos/bye_splits/tests/submission/
        args_per_batch: 2
```

After setting the configuration variables, the jobs are created and launched via

    python bye_splits/production/submit_scripts/job_submit.py

while will produce the executable `.sh` file in `<submit_dir>/subs/` that looks like:

    #!/usr/bin/env bash
    export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
    export SITECONFIG_PATH=$VO_CMS_SW_DIR/SITECONF/T2_FR_GRIF_LLR/GRIF-LLR/
    source $VO_CMS_SW_DIR/cmsset_default.sh
    list=$1
    cleaned_list=$(echo $list | tr -d '[]' | tr ';' '
    ')
    while IFS=";" read -r val; do
        python /grid_mnt/vol_home/llr/cms/ehle/NewRepos/bye_splits/tests/submission/dummy_submit.py --gen_arg "$val" --float_arg 0.11 --str_arg a_string
    done <<< "$cleaned_list"

and the `.sub` file submitted to HT Condor in `<subdmit_dir>/jobs/` that looks like:

    executable = /home/llr/cms/ehle/NewRepos/bye_splits/tests/submission/subs/dummy_submit_exec_v5.sh
    Universe              = vanilla
    Arguments = $(gen_arg) $(float_arg) $(str_arg)
    output = /home/llr/cms/ehle/NewRepos/bye_splits/tests/submission/logs/dummy_submit_C$(Cluster)P$(Process).out
    error = /home/llr/cms/ehle/NewRepos/bye_splits/tests/submission/logs/dummy_submit_C$(Cluster)P$(Process).err
    log = /home/llr/cms/ehle/NewRepos/bye_splits/tests/submission/logs/dummy_submit_C$(Cluster)P$(Process).log
    getenv                = true
    T3Queue = short
    WNTag                 = el7
    +SingularityCmd       = ""
    include: /opt/exp_soft/cms/t3/t3queue |
    queue gen_arg, float_arg, str_arg from (
    ['gen';3.14], 0.11, a_string
    ['work';'broke'], 0.11, a_string
    [9;False], 0.11, a_string
    [12.9;'hello'], 0.11, a_string
    )

All logs, outputs, and errors are written to their respective files in `<submit_dir>/logs/`. Some primary uses of `job_submit.py` include running the [skimming procedure](#skimming), iterating over each particle type, and running the [cluster studies](#cluster-size-studies) over a list of radii.

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

The optimization of the clustering radius is done via the scripts in `bye_splits/scripts/cluster_size/`. The configuration is done in the `config.yaml` file under `clusterStudies`.
The initial steps of the reconstruction chain (fill, smooth, seed) are run via

    python run_init_tasks.py --pileup <PU0/PU200>

which will produce the files required for `bye_splits/scripts/cluster_size/condor/run_cluster.py` (default value for `pileup==PU0`). One can run the script on a single radius:

    python run_cluster.py --radius <float> --particles <photons/electrons/pions> --pileup <PU0/PU200>

As the directory name suggests, `run_cluster.py` can and should be run as a `script` passed to an HTCondor job as described by [Job Submission](#job-submission) if you wish
to run over all radii. The configuration would look something like this:

```yaml
job:
user: iehle
proxy: ~/.t3/proxy.cert
queue: short
local: False
script: /grid_mnt/vol_home/llr/cms/ehle/NewRepos/bye_splits/bye_splits/scripts/cluster_size/condor/run_cluster.py
iterOver: radius
arguments:
    radius: [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
            0.01 , 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018,
            0.019, 0.02 , 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027,
            0.028, 0.029, 0.03 , 0.031, 0.032, 0.033, 0.034, 0.035, 0.036,
            0.037, 0.038, 0.039, 0.04 , 0.041, 0.042, 0.043, 0.044, 0.045,
            0.046, 0.047, 0.048, 0.049, 0.05]
    particles: pions
    pileup: PU0
photons:
    submit_dir: /data_CMS/cms/ehle/L1HGCAL/PU0/photons/
    args_per_batch: 10
electrons:
    submit_dir: /data_CMS/cms/ehle/L1HGCAL/PU0/electrons/
    args_per_batch: 10
pions:
    submit_dir: /data_CMS/cms/ehle/L1HGCAL/PU0/pions/
    args_per_batch: 10
```

This will produce the output of `cluster.cluster_default()` for each radius. These files are then combined into one larger `.hdf5` file whose keys correspond to the various radii, and combined and normalized with the gen-level data via:

    python run_combine.py

The optional `--file` argument performs the combination and normalization with the gen-level data on only `<file>`.

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

Use LLR's intranet at `llruicms01.in2p3.fr:<port>/display`

Forward it to our local machines via `ssh`. To establish a connection between the local machine and the remote `llruicms01` server, passing by the gate, use:

    ssh -L <port>:llruicms01.in2p3.fr:<port> -N <llr_username>@llrgate01.in2p3.fr
    # for instance: ssh -L 8080:lruicms01.in2p3.fr:8080 -N alves@llrgate01.in2p3.fr

The two ports do not have to be the same, but it avoids possible confusion. Leave the terminal open and running (it will not produce any output).


<a id="org3f41a7d"></a>

## Visualization in local browser

<a id="orgb4acfa6"></a>

### 1) 2D display app

In a new terminal window go to the `llruicms01` machines and launch one of the apps, for instance:

    bokeh serve bye_splits/plot/display/ --address llruicms01.in2p3.fr --port <port>  --allow-websocket-origin=localhost:<port>
    # if visualizing directly at LLR: --allow-websocket-origin=llruicms01.in2p3.fr:<port>

This uses the server-creation capabilities of `bokeh`, a `python` package for interactive visualization ([docs](https://docs.bokeh.org/en/latest/index.html)). Note the port number must match. For further customisation of `bokeh serve` see [the serve documentation](https://docs.bokeh.org/en/latest/docs/reference/command/subcommands/serve.html).
The above command should give access to the visualization under `http://localhost:8080/display`. For debugging, just run `python bye_splits/plot/display/main.py`  and see that no errors are raised.

<a id="orgbafd8a5"></a>

### 2) 3D display app

Make sure you have activated your `conda` environment. 
    conda activate <Env>

Run the following lines. With these commands, some useful packages to run the web application (e.g. `dash`, `uproot`, `awkward`, etc) will be installed in your `conda` environment:

    conda install dash
    python3 -m pip install dash-bootstrap-components
    python3 -m pip install dash-bootstrap-templates
    conda install pandas pyyaml numpy bokeh awkward uproot h5py pytables
    conda install -c conda-forge pyarrow fsspec

Then go to the `llruicms01` machine (if you are indide LLR intranet) or to your preferred machine and launch:

    python bye_splits/plot/display_plotly/main.py --port 5004 --host localhost

In a browser, go to http://localhost:5004/.
Make sure you have access to the geometry and event files, to be configured in `config.yaml`.

<a id="org4164d71"></a>

## Visualization with OpenShift OKD4

We use the [S2I](https://docs.openshift.com/container-platform/3.11/creating_images/s2i.html) (Source to Image) service via CERN's [PaaS](https://paas.docs.cern.ch/) (Platform-as-a-Service) using OpenShift to deploy and host web apps in the CERN computing environment [here](https://paas.cern.ch/). There are three ways to deploys such an app: S2I represents the easiest (but less flexible) of the three; instructions [here](https://paas.docs.cern.ch/2._Deploy_Applications/Deploy_From_Git_Repository/2-deploy-s2i-app/). It effectively abstracts away the need for Dockerfiles.

We will use S2I's simplest configuration possible under `app.sh`. The image is created alongside the packages specified in `requirements.txt`. The two latter definitions are documented [here](https://github.com/kubesphere/s2i-python-container/blob/master/2.7/README.md#source-repository-layout).

We are currently running a pod at <https://viz2-hgcal-event-display.app.cern.ch/>. The port being served by `bokeh` in `app.sh` must match the one the pod is listening to, specified at configuration time before deployment in the [OpenShift management console](https://paas.cern.ch/) at CERN. The [network visibility](https://paas.docs.cern.ch/5._Exposing_The_Application/2-network-visibility/) was also updated to allow access from outside the CERN network.


<a id="orgab38b90"></a>

### Additional information

-   [What is a pod](https://cloud.google.com/kubernetes-engine/docs/concepts/pod)?
-   [How to mount `/eos` at CERN so that it is accessible by a pod?](https://paas.docs.cern.ch/3._Storage/eos/)


<a id="orga2b91d9"></a>

# Cluster Radii Studies

A DashApp has been built to interactively explore the effect of cluster size on various cluster properties, which is currently hosted at <https://bye-splits-app-hgcal-cl-size-studies.app.cern.ch/>.
To run the app locally, you can do:

    bash run_cluster_app.sh <username>

where `<username>` is your lxplus username. The app reads the configuration file `bye_splits/plot/display_clusters/config.yaml` and assumes that you have a directory structure equivalent to the directories described in the cluster size step (depending on your choice of \`\`\`Local\`\`\`).

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

