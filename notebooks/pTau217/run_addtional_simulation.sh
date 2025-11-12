#!/bin/bash

# Define the number of trials for the simulations
num_trials=1000

# Define the number of repeated measures for each group
# You can change these values to any desired number.
repeat_measures_grp1=3
repeat_measures_grp2=4

# Define an array of efficacy values to iterate through
# Using 'declare -a' for explicit array declaration is good practice,
# though not strictly necessary for simple arrays in bash.
declare -a efficacies=(1.0 0.50 0.25)

echo "Starting LME simulations with num_trials = ${num_trials}"
echo "Repeat measures for Group 1: ${repeat_measures_grp1}"
echo "Repeat measures for Group 2: ${repeat_measures_grp2}"

# Loop through each efficacy value in the array
for efficacy in "${efficacies[@]}"; do
    echo "------------------------------------------------------------"
    echo "Running simulations for efficacy = ${efficacy}"
    echo "------------------------------------------------------------"

    # Define the output folder name using the current efficacy value
    # Note: Using printf "%.2f" to format efficacy to two decimal places for folder name consistency
    output_folder_name="outputs_e${efficacy}_m${repeat_measures_grp1}${repeat_measures_grp2}"
    echo ${output_folder_name}

    # Run the crossover LME simulation script
    echo "Executing crossover_lme_fixed_within_subj_var_mp_ptau217.py..."
    python3.11 ./crossover_lme_fixed_within_subj_var_mp_ptau217.py \
        --output-folder "${output_folder_name}" \
        --num-trials "${num_trials}" \
        --efficacy "${efficacy}" \
        --repeat-measure-grp1 "${repeat_measures_grp1}" \
        --repeat-measure-grp2 "${repeat_measures_grp2}" \
        --with-dependency \
        --log-transform

    echo "Crossover simulation for efficacy ${efficacy} completed."

    # Run the parallel LME simulation script
    echo "Executing parallel_lme_fixed_within_subj_var_mp_ptau217.py..."
    python3.11 ./parallel_lme_fixed_within_subj_var_mp_ptau217.py \
        --output-folder "${output_folder_name}" \
        --num-trials "${num_trials}" \
        --efficacy "${efficacy}" \
        --log-transform

    echo "Parallel simulation for efficacy ${efficacy} completed."
    echo "" # Add an empty line for better readability between efficacy runs
done

echo "All LME simulations completed."
# num_trials=1000
#
#
# efficacy=1.0
# python3.11 ./crossover_lme_fixed_within_subj_var_mp_ptau217.py --output-folder "outputs_e${efficacy}_m44" --num-trials $num_trials \
#     --efficacy $efficacy \
#     --repeat-measure-grp1 4 \
#     --repeat-measure-grp2 4 \
#     --with-dependency
#
# python3.11 ./parallel_lme_fixed_within_subj_var_mp_ptau217.py  --output-folder "outputs_e${efficacy}_m44" --num-trials $num_trials \
#     --efficacy $efficacy
#
# #########################################################################################################
# efficacy=0.50
# python3.11 ./crossover_lme_fixed_within_subj_var_mp_ptau217.py --output-folder "outputs_e${efficacy}_m44" --num-trials $num_trials \
#     --efficacy $efficacy \
#     --repeat-measure-grp1 4 \
#     --repeat-measure-grp2 4 \
#     --with-dependency
#
# python3.11 ./parallel_lme_fixed_within_subj_var_mp_ptau217.py  --output-folder "outputs_e${efficacy}_m44" --num-trials $num_trials \
#     --efficacy $efficacy
#
#
# #########################################################################################################
# efficacy=0.25
# python3.11 ./crossover_lme_fixed_within_subj_var_mp_ptau217.py --output-folder "outputs_e${efficacy}_m44" --num-trials $num_trials \
#     --efficacy $efficacy \
#     --repeat-measure-grp1 4 \
#     --repeat-measure-grp2 4 \
#     --with-dependency
#
# python3.11 ./parallel_lme_fixed_within_subj_var_mp_ptau217.py  --output-folder "outputs_e${efficacy}_m44" --num-trials $num_trials \
#     --efficacy $efficacy
#
# #########################################################################################################
