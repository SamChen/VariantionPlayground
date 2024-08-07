using Random
using Distributions
using IterTools
using MixedModels
using DataFrames
using CSV

function generate_subject_means(between_subject_mean::Float64, between_subject_variance::Float64,
        subject_size::Int64, random_seed::Int64)
    Random.seed!(random_seed)
    dist_model = Normal(between_subject_mean, sqrt(between_subject_variance))
    subject_means = rand(dist_model, subject_size)
    return subject_means
end

function generate_subject_variances(within_subject_variance_left::Float64, within_subject_variance_right::Float64,
        subject_size::Int64, random_seed::Int64)
    Random.seed!(random_seed)
    dist_model = Uniform(within_subject_variance_left, within_subject_variance_right)
    subject_variances = rand(dist_model, subject_size)
    return subject_variances
end

function generate_samples(subject_mean::Float64, subject_variance::Float64;
        repeated_measurements::Int64=0, random_seed::Int64=0)
	Random.seed!(random_seed)
    dist_model = Normal(subject_mean, sqrt(subject_variance))
    subject_means = rand(dist_model, repeated_measurements)
    return subject_means
end

function run_trial(between_subj_mean1::Float64, between_subj_var1::Float64,
        within_subj_var_left1::Float64, within_subj_var_right1::Float64,
        between_subj_mean2::Float64, between_subj_var2::Float64,
        within_subj_var_left2::Float64, within_subj_var_right2::Float64,
        n_subj::Int64, M::Int64,
        seed::Int64)

	data = []
    # phase1
	subject_means    = generate_subject_means(between_subj_mean1, between_subj_var1, n_subj, seed)
	subject_variance = generate_subject_variances(within_subj_var_left1, within_subj_var_right1, n_subj, seed+1)
	for (subj_id, (subj_m, subj_v)) in enumerate(zip(subject_means, subject_variance))
		tmp = DataFrame(subj_id=subj_id, value=generate_samples(subj_m, subj_v; 
        repeated_measurements=M, random_seed=seed+2+subj_id))
		tmp[!, :phase] .= 0
		push!(data, tmp)
	end

    # phase2
	subject_means    = generate_subject_means(between_subj_mean2, between_subj_var2, n_subj, seed)
	subject_variance = generate_subject_variances(within_subj_var_left2, within_subj_var_right2, n_subj, seed+1)
	for (subj_id, (subj_m, subj_v)) in enumerate(zip(subject_means, subject_variance))
		tmp = DataFrame(subj_id=subj_id, value=generate_samples(subj_m, subj_v; 
        repeated_measurements=M, random_seed=seed+2+subj_id))
		tmp[!, :phase] .= 1
		push!(data, tmp)
	end

	df = vcat(data...)
    fm = @formula(value ~ phase + (phase|subj_id))
	# df = MixedModels.dataset(:df)
    fm1 = fit(MixedModel, fm, df, REML=false)
	return coeftable(fm1).cols[4][2]
end