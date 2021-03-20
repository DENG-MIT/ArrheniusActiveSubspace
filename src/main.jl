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

T0 = 1400.0  #K
P = 40.0 * one_atm
phi = 1.0

prob = make_prob(T0, P, phi, p)
@time ts, pred = get_idt(T0, P, phi, p; dT=400, dTabort=800)

ng = length(ts);
Fp = zeros(ng * nu, np);
Fy = BandedMatrix(zeros(ng * nu, ng * nu), (nu, nu));
