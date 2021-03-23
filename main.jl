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
phi = 1.0;            # equivalence ratio
P = 10.0 * one_atm; # pressure, atm
T0 = 1200.0;        # initial temperature, K

Random.seed!(0);
n_sample = Int64(ceil(5 * log(np)));
p_sample = rand(n_sample, np) .- 0.5;
τ_sample = zeros(n_sample); # IDT sample
∇f_sample = similar(p_sample); # sensitivities sample

# sampling for sensitivity   
epochs = ProgressBar(1:n_sample);
for i in epochs
    p = p_sample[i, :];

    if method == "sensBVP_mthread"
        ts, pred = get_Tcurve(phi, P, T0, p; dT=dT, dTabort=dTabort);
        ts, pred = downsampling(ts, pred; dT=2.0, verbose=false);

        τ_sample[i] = ts[end]

        set_description(
            epochs,
            string(
                @sprintf("sample %d with %d points", i, length(ts))
            ),
        )
        ∇f_sample[i, :] = sensBVP_mthread(ts, pred, p)
    else
        idt = get_idt(phi, P, T0, p; dT=dT, dTabort=dTabort);
        τ_sample[i] = idt;
        
        set_description(
            epochs,
            string(
                @sprintf("sample %d with sensBF", i)
            ),
        )
        ∇f_sample[i, :] = sensBF_mthread(phi, P, T0, p; dT=dT, dTabort=dTabort, pdiff=5e-3)
    end
end

# eigen decomposition
C = ∇f_sample' * ∇f_sample;
eigs, eigvec = eigen(C);
eigs = reverse(eigs);
eigvec = reverse(eigvec, dims=2);

@save string(exp_path, "/eigs.bason") eigs ∇f_sample;

# show resutls
h1 = plot(xlabel="Index", ylabel="Eigenvalues");
plot!(1:12, eigs[1:12], lw=3, yscale=:log10);

h2 = plot(xlabel="Active variable", ylabel="IDT [s]");
scatter!(p_sample * eigvec[:,1], τ_sample, yscale=:log10);

h = plot([h1, h2]...);

png(h, string(exp_path, "/eigs.png"));