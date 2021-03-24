include("header.jl")
include("simulator.jl")
include("visual.jl")

# load mech
gas = CreateSolution(mech);
const ns = gas.n_species;
const nr = gas.n_reactions;
const nu = ns + 2;
const np = nr;
const ind_diag = diagind(ones(nu - 1, nu - 1));
const ones_nu = ones(nu - 1);

include("sensitivity.jl")

# set condition
phi = 1.0;          # equivalence ratio
P = 10.0 * one_atm; # pressure, atm
T0 = 1200.0;        # initial temperature, K
ts, pred = get_Tcurve(phi, P, T0, zeros(nr); dT=dT, doplot=true, dTabort=dTabort);

# sampling for sensitivity
rng = Random.MersenneTwister(0x7777777);
n_sample = Int64(ceil(5 * log(np)));
p_sample = rand(rng, n_sample, np) * 0.2 .- 0.1;
τ_sample = zeros(n_sample); # IDT sample
∇f_sample = similar(p_sample); # sensitivities sample

epochs = ProgressBar(1:n_sample);
for i in epochs
    p = p_sample[i, :];

    ts, pred = get_Tcurve(phi, P, T0, p; dT=dT, dTabort=dTabort);
    idt = interpx(ts, pred[end,:], pred[end,1] + dT);
    τ_sample[i] = deepcopy(idt);

    set_description(
        epochs,
        @sprintf("sample %d with %s", i, method)
    );

    if method == "sensBVP_mthread" # under testing
        # ts, pred = downsampling(ts, pred; dT=2, verbose=false);
        ∇f_sample[i, :] = sensBVP_mthread(ts, pred, p);
        # ∇f_sample[i, :] = sensBVP(ts, pred, p);

    elseif method == "sensBF_mthread" # unsuitable for large mechanism
        ∇f_sample[i, :] = sensBF_mthread(phi, P, T0, p; dT=dT, dTabort=dTabort, pdiff=5e-3);

    elseif method == "sensBFSA" # by ForwardDiff
        ∇f_sample[i, :] = sensBFSA(phi, P, T0, p; dT=dT, dTabort=dTabort);

    else
        @show "Wrong sensitivity method.";
    end
end 

# eigen decomposition
C = ∇f_sample' * ∇f_sample;
eigs, eigvec = eigen(C);
eigs = reverse(eigs);
eigvec = reverse(eigvec, dims=2);

@save string(exp_path, "/eigs.bson") eigs ∇f_sample p_sample τ_sample;

# show resutls
h1 = plot(xlabel="Index", ylabel="Eigenvalues");
plot!(1:12, eigs[1:12], lw=3, marker=:circle, yscale=:log10);
xlims!((1, 12));
xticks!(1:1:12);

h2 = plot(xlabel="Active variable", ylabel="IDT [s]");
scatter!(p_sample * eigvec[:,1], τ_sample, yscale=:log10);

#For online blog
h = plot([h1, h2]..., legend=false);

# For paper
# h = plot([h1, h2]..., legend=false, framestyle = :box, 
#                       layout = (2, 1), size = (600, 600));
png(h, string(exp_path, "/eigs.png"));
