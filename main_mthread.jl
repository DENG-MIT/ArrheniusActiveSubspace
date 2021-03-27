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

phi = 1.0;          # equivalence ratio
P = 40.0 * one_atm; # pressure, atm
T0 = 1200.0;        # initial temperature, K

p = zeros(nr);
@elapsed ts, pred = get_Tcurve(phi, P, T0, p;
                                dT=dT, doplot=true, dTabort=dTabort);
# ts, pred = downsampling(ts, pred; dT=2.0, verbose=false);
idt = ts[end]
Tign = pred[end, end]
ng = length(ts)
Fy = BandedMatrix(Zeros(ng * nu, ng * nu), (nu, nu));
Fp  = zeros(ng * nu, np);
i = 1;
i_F = 1 + (i - 1) * nu:i * nu - 1;
@view(Fy[i_F, i_F])[ind_diag] .= ones_nu;
Fy[i * nu, i * nu] = -1.0;
Fy[i * nu, (i + 1) * nu] = 1.0;
du = similar(pred[:, 1])

dts = @views(ts[2:end] .- ts[1:end - 1]) ./ idt;

@threads for i = ng-10:ng
    u = @view(pred[:, i])
    du = similar(u)
    i_F = 1 + (i - 1) * nu:i * nu - 1
    @view(Fp[i_F, :]) .= jacobian((du, x) -> dudt!(du, u, x, 0.0),
                                du, p)::Array{Float64,2} .* (-idt)
end

@threads for i = 2:10
    u = @view(pred[:, i])
    du = similar(u)
    i_F = 1 + (i - 1) * nu:i * nu - 1
    @view(Fy[i_F, i_F]) .= jacobian((du, x) -> dudt!(du, x, p, 0.0),
                                du, u)::Array{Float64,2} .* (-idt)
end

@elapsed for i = 2:ng
    u = @view(pred[:, i])
    i_F = 1 + (i - 1) * nu:i * nu - 1
    @view(Fy[i_F, i * nu]) .= - dudt!(du, u, p, 0.0)
    @view(Fy[i_F, i_F])[ind_diag] .+= ones_nu ./ (dts[i - 1])
    @view(Fy[i_F, i_F .- nu])[ind_diag] .+= ones_nu ./ (-dts[i - 1])
    if i < ng
        Fy[i * nu, i * nu] = -1.0
        Fy[i * nu, (i + 1) * nu] = 1.0
    else
        Fy[i * nu, i * nu - 1] = 1.0
    end
end

@time dydp = - Fy \ sparse(Fp)
grad = @view(dydp[end, :]) ./ idt
