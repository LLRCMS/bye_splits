#!/usr/bin/env bash
declare -a ITER_PARS=( $(seq 0. .1 1.) )

DRYRUN="0"
REPROCESS="0"
PLOT_TC="0"
DO_FILLING="1"
DO_SMOOTHING="1"
DO_SEEDING="1"
DO_CLUSTERING="1"
NEVENTS="-1"
CLUSTER_ALGO="min_distance"
SEED_WINDOW="1"
SMOOTH_KERNEL="default"
declare -a SELECTIONS=( "splits_only" "above_eta_2.7" )
declare -a REGIONS=( "Si" "ECAL" "MaxShower" )
declare -a CLUSTER_ALGOS=( "min_distance" "max_energy" )
declare -a SMOOTH_KERNELS=( "default" "flat_top" )
SELECTION="splits_only"
REGION="Si"

### Argument parsing
HELP_STR="Prints this help message."
DRYRUN_STR="(Boolean) Prints all the commands to be launched but does not launch them. Defaults to ${DRYRUN}."
REPROCESS_STR="(Boolean) Reprocesses the input dataset (slower). Defaults to '${REPROCESS}'."
PLOT_TC_STR="(Boolean) plot shifted trigger cells instead of originals. Defaults to '${PLOT_TC}'."
DO_FILLING_STR="(Boolean) run the filling task. Defaults to '${DO_FILLING}'."
DO_SMOOTHING_STR="(Boolean) run the smoothing task. Defaults to '${DO_SMOOTHING}'."
DO_SEEDING_STR="(Boolean) run the seeding task. Defaults to '${DO_SEEDING}'."
DO_CLUSTERING_STR="(Boolean) run the clustering task. Defaults to '${DO_CLUSTERING}'."
SELECTION_STR="(String) Which initial data selection to apply. Values supported: ${SELECTIONS[*]}.  Defaults to '${SELECTION}'."
REGION_STR="(String) Which initial region to consider. Values supported: ${REGIONS[*]}. Defaults to '${REGION}'."

function print_usage_iter_opt {
    USAGE=" $(basename "$0") [-H] [--dry-run --resubmit -t -d -n --klub_tag --stitching_on]

	-h / --help			[ ${HELP_STR} ]
	--dry-run			[ ${DRYRUN_STR} ]
	-r / --reprocess	[ ${REPROCESS_STR} ]
	-p / --plot_tc	    [ ${PLOT_TC_STR} ]
	--selection	     	[ ${SELECTION_STR} ]
	--region			[ ${REGION_STR} ]
	--no_fill			[ ${DO_FILLING_STR} ]
	--no_smooth			[ ${DO_SMOOTHING_STR} ]
	--no_seed			[ ${DO_SEEDING_STR} ]
	--no_cluster		[ ${DO_CLUSTERING_STR} ]

    Run example: bash $(basename "$0")
"
    printf "${USAGE}"
}

######################################
## Argument parsing ##################
######################################
while [[ $# -gt 0 ]]; do
    key=${1}
    case $key in
		-h|--help)
			print_usage_iter_opt
			exit 1
			;;
		--region)
			if [ -n "$2" ]; then
				if [[ ! " ${REGIONS[@]} " =~ " ${2} " ]]; then
					echo "Region ${2} is not supported."
					exit 1;
				else
					REGION="${2}";
					echo "Region to consider: ${REGION}";
				fi
			fi
			shift 2;;

		--selection)
			if [ -n "$2" ]; then
				if [[ ! " ${SELECTIONS[@]} " =~ " ${2} " ]]; then
					echo "Data selection ${2} is not supported."
					exit 1;
				else
					SELECTION="${2}";
					echo "variable to consider: ${SELECTION}";
				fi
			fi
			shift 2;;

		--cluster_algo)
			if [ -n "$2" ]; then
				if [[ ! " ${CLUSTER_ALGOS[@]} " =~ " ${2} " ]]; then
					echo "Cluster algorithm ${2} is not supported."
					exit 1;
				else
					CLUSTER_ALGO="${2}";
					echo "variable to consider: ${CLUSTER_ALGO}"
				fi
			fi
			shift 2;;

		--seed_window)
			if [ -n "$2" ]; then
				if [[ "${2}" -lt 1 ]]; then
					echo "The seed window has to be at least one."
					exit 1;
				else
					SEED_WINDOW="${2}";
					echo "variable to consider: ${SEED_WINDOW}"
				fi
			fi
			shift 2;;

		--smooth_kernel)
			if [ -n "$2" ]; then
				if [[ ! " ${SMOOTH_KERNELS[@]} " =~ " ${2} " ]]; then
					echo "Smoothing kernel '${2}' is not supported."
					exit 1;
				else
					SMOOTH_KERNEL="${2}";
					echo "variable to consider: ${SMOOTH_KERNEL}"
				fi
			fi
			shift 2;;

		--no_fill)
			DO_FILLING=0;
			shift ;;

		--no_smooth)
			DO_SMOOTHING=0;
			shift ;;

		--no_seed)
			DO_SEEDING=0;
			shift ;;

		--no_cluster)
			DO_CLUSTERING=0;
			shift ;;

		-p|--plot_tc)
			PLOT_TC=1;
			shift ;;

		-r|--reprocess)
			REPROCESS=1;
			shift ;;

		-d|--dry_run)
			DRYRUN="1";
			shift ;;

		--nevents)
			NEVENTS="${2}"
			shift 2;;

		--) shift; break;;
		*) break ;;
    esac
done

printf "===== Input Arguments =====\n"
printf "Dry-run: %s\n" ${DRYRUN}
printf "Region: %s\n" ${REGION}
printf "Selection: %s\n" ${SELECTION}
printf "Cluster algo: %s\n" ${CLUSTER_ALGO}
printf "Seed window: %s\n" ${SEED_WINDOW}
printf "Smooth kernel: %s\n" ${SMOOTH_KERNEL}
printf "Reprocess trigger cell geometry data: %s\n" ${REPROCESS}
printf "Perform filling: %s\n" ${DO_FILLING}
printf "Plot trigger cells: %s\n" ${PLOT_TC}
printf "Number of events: %s\n" ${NEVENTS}
printf "===========================\n"

if [ ${DO_FILLING} -eq 1 ]; then
	echo "Run the filling step.";
fi
if [ ${DO_SMOOTHING} -eq 1 ]; then
	echo "Run the smoothing step.";
fi
if [ ${DO_SEEDING} -eq 1 ]; then
	echo "Run the seeding step.";
fi
if [ ${DO_CLUSTERING} -eq 1 ]; then
	echo "Run the clustering step.";
fi

### Functions
function run_parallel() {
  comm="mkdir -p ../data"
  comm+="mkdir -p ../out"
	comm+="parallel -j -1 python bye_splits/iterative_optimization.py --ipar {} --sel ${SELECTION} -n ${NEVENTS} --reg ${REGION} "
	comm+="--cluster_algo ${CLUSTER_ALGO} --seed_window ${SEED_WINDOW} --smooth_kernel ${SMOOTH_KERNEL} "
	if [ ${DO_FILLING} -eq 0 ]; then
		echo "Do not run the filling step."
		comm+="--no_fill "
	fi
	if [ ${DO_SMOOTHING} -eq 0 ]; then
		echo "Do not run the smoothing step."
		comm+="--no_smooth "
	fi
	if [ ${DO_SEEDING} -eq 0 ]; then
		echo "Do not run the seeding step."
		comm+="--no_seed "
	fi
	if [ ${DO_CLUSTERING} -eq 0 ]; then
		echo "Do not run the clustering step."
		comm+="--no_cluster "
	fi

	if [ ${PLOT_TC} -eq 1 ]; then
		comm+="-p "
	fi

	comm+="$@"

	[[ ${DRYRUN} -eq 1 ]] && echo ${comm} || ${comm}
}

function run_plot() {
	comm="python plot/meta_algorithm.py -m ${@} --sel ${SELECTION} --reg ${REGION} "
	comm+="--cluster_algo ${CLUSTER_ALGO} --seed_window ${SEED_WINDOW} --smooth_kernel ${SMOOTH_KERNEL} "
	[[ ${DRYRUN} -eq 1 ]] && echo ${comm} || ${comm}
}

### Only one job can reprocess the data, and it has to be sequential
if [ ${REPROCESS} -eq 1 ]; then
	run_parallel -r ::: ${ITER_PARS[0]}
	run_parallel ::: ${ITER_PARS[@]:1}
else
	run_parallel ::: ${ITER_PARS[@]}
fi

#run_plot "${ITER_PARS[*]}"

echo "All jobs finished."
