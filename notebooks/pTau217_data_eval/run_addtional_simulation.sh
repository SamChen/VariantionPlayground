num_trials=1000


efficacy=1.0
python3.11 ./crossover_lme_fixed_within_subj_var_mp_ptau217.py --output-folder "outputs_e${efficacy}_m44" --num-trials $num_trials \
    --efficacy $efficacy \
    --repeat-measure-grp1 10 \
    --repeat-measure-grp2 10 \
    --with-dependency
