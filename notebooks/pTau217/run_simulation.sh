num_trials=10


efficacy=1.0
python3.11 ./crossover_lme_fixed_within_subj_var_mp_ptau217.py --output-folder "outputs_e${efficacy}_m55" --num-trials $num_trials \
    --efficacy $efficacy \
    --repeat-measure-grp1 5 \
    --repeat-measure-grp2 5 \
    --with-dependency

python3.11 ./parallel_lme_fixed_within_subj_var_mp_ptau217.py  --output-folder "outputs_e${efficacy}_m55" --num-trials $num_trials \
    --efficacy $efficacy


python3.11 ./crossover_lme_fixed_within_subj_var_mp_ptau217.py --output-folder "outputs_e${efficacy}_m33" --num-trials $num_trials \
    --efficacy $efficacy \
    --repeat-measure-grp1 3 \
    --repeat-measure-grp2 3 \
    --with-dependency

python3.11 ./parallel_lme_fixed_within_subj_var_mp_ptau217.py  --output-folder "outputs_e${efficacy}_m33" --num-trials $num_trials \
    --efficacy $efficacy

#########################################################################################################
efficacy=0.50
python3.11 ./crossover_lme_fixed_within_subj_var_mp_ptau217.py --output-folder "outputs_e${efficacy}_m55" --num-trials $num_trials \
    --efficacy $efficacy \
    --repeat-measure-grp1 5 \
    --repeat-measure-grp2 5 \
    --with-dependency

python3.11 ./parallel_lme_fixed_within_subj_var_mp_ptau217.py  --output-folder "outputs_e${efficacy}_m55" --num-trials $num_trials \
    --efficacy $efficacy


python3.11 ./crossover_lme_fixed_within_subj_var_mp_ptau217.py --output-folder "outputs_e${efficacy}_m33" --num-trials $num_trials \
    --efficacy $efficacy \
    --repeat-measure-grp1 3 \
    --repeat-measure-grp2 3 \
    --with-dependency

python3.11 ./parallel_lme_fixed_within_subj_var_mp_ptau217.py  --output-folder "outputs_e${efficacy}_m33" --num-trials $num_trials \
    --efficacy $efficacy

#########################################################################################################
efficacy=0.25
python3.11 ./crossover_lme_fixed_within_subj_var_mp_ptau217.py --output-folder "outputs_e${efficacy}_m55" --num-trials $num_trials \
    --efficacy $efficacy \
    --repeat-measure-grp1 5 \
    --repeat-measure-grp2 5 \
    --with-dependency

python3.11 ./parallel_lme_fixed_within_subj_var_mp_ptau217.py  --output-folder "outputs_e${efficacy}_m55" --num-trials $num_trials \
    --efficacy $efficacy


python3.11 ./crossover_lme_fixed_within_subj_var_mp_ptau217.py --output-folder "outputs_e${efficacy}_m33" --num-trials $num_trials \
    --efficacy $efficacy \
    --repeat-measure-grp1 3 \
    --repeat-measure-grp2 3 \
    --with-dependency

python3.11 ./parallel_lme_fixed_within_subj_var_mp_ptau217.py  --output-folder "outputs_e${efficacy}_m33" --num-trials $num_trials \
    --efficacy $efficacy

#########################################################################################################
