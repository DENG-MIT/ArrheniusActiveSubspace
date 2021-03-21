include("header.jl")
include("simulator.jl")
include("visual.jl")

gas = CreateSolution(mech);
const ns = gas.n_species;
const nr = gas.n_reactions;

p = zeros(nr);
const nu = ns + 2;
const np = length(p);
const ind_diag = diagind(ones(nu - 1, nu - 1));
const ones_nu = ones(nu - 1);

include("sensBVP.jl")

T0 = 1400.0  # K
P = 40.0 * one_atm
phi = 1.0

prob = make_prob(T0, P, phi, p);

Random.seed!(0);
n_sample = Int64(floor(2 * log(np)))
p_sample = rand(n_sample, np) .- 0.5;

C = similar(p_sample);

epochs = ProgressBar(1:n_sample);
for i in epochs
    p = p_sample[i, :]
    ts, pred = get_idt(T0, P, phi, p; 
                    dT=200, dTabort=600, doplot=true);
    ts, pred = downsampling(ts, pred; dT=2.0, verbose=false)

    if pred[end, end] < T0 + 190
        println("no ignition for sample $i")
        break
    end

    set_description(
        epochs,
        string(
            @sprintf("sample %d no of points %d", i, length(ts))
        ),
    )
    C[i, :] = sensBVP_mthread(ts, pred, p)
end

eigs = eigvals(C' * C)[end - 20:end]

plt = plot(eigs);
xlabel!(plt, "Index")
ylabel!(plt, "Eigenvalues")
png(plt, "./results/nc7_ver3.1_mech/eigs.png")