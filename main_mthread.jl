include("header.jl")
include("simulator.jl")
include("visual.jl")

# load mech
gas = CreateSolution(mech);
const ns = gas.n_species;
const nr = gas.n_reactions;
const nu = ns + 2;
const ind_diag = diagind(ones(nu - 1, nu - 1));
const ones_nu = ones(nu - 1);

include("sensitivity.jl")

phi = 1.0;          # equivalence ratio
P = 40.0 * one_atm; # pressure, atm
T0 = 1200.0;        # initial temperature, K

@elapsed ts, pred = get_Tcurve(phi, P, T0, zeros(nr);
                                dT=dT, doplot=true, dTabort=dTabort);
# ts, pred = downsampling(ts, pred; dT=2.0, verbose=false);

npr = 16
p = zeros(npr*3);

idt = ts[end]
Tign = pred[end, end]
ng = length(ts)
np = length(p)
Fy = BandedMatrix(Zeros(ng * nu, ng * nu), (nu, nu));
Fp  = zeros(ng * nu, np);
i = 1;
i_F = 1 + (i - 1) * nu:i * nu - 1;
@view(Fy[i_F, i_F])[ind_diag] .= ones_nu;
Fy[i * nu, i * nu] = -1.0;
Fy[i * nu, (i + 1) * nu] = 1.0;
du = similar(pred[:, 1])

dts = @views(ts[2:end] .- ts[1:end - 1]) ./ idt;

function dudtp!(du, u, p, t)
    Y = @view(u[1:ns])
    T = u[end]
    mean_MW = 1.0 / dot(Y, 1 ./ gas.MW)
    ρ_mass = P / R / T * mean_MW
    X = Y2X(gas, Y, mean_MW)
    C = Y2C(gas, Y, ρ_mass)
    cp_mole, cp_mass = get_cp(gas, T, X, mean_MW)
    h_mole = get_H(gas, T, Y, X)
    S0 = get_S(gas, T, P, X)
    _p = reshape(p, npr, 3)
    kp = @. @views(exp(_p[:, 1] + _p[:, 2] * log(T) - _p[:, 3] * 4184.0 / R / T))
    qdot::typeof(p) = wdot_func(gas.reaction, T, C, S0, h_mole; get_qdot=true)
    @. qdot[1:npr] *= kp
    wdot = gas.reaction.vk * qdot
    Ydot = wdot / ρ_mass .* gas.MW
    Tdot = -dot(h_mole, wdot) / ρ_mass / cp_mass
    du .= vcat(Ydot, Tdot)
    return du
end

@threads for i = 1:ng
    u = @view(pred[:, i])
    du = similar(u)
    i_F = 1 + (i - 1) * nu:i * nu - 1
    @view(Fp[i_F, :]) .= jacobian((du, x) -> dudtp!(du, u, x, 0.0),
                                du, p)::Array{Float64,2} .* (-idt)
end

@threads for i = 2:ng
    u = @view(pred[:, i])
    du = similar(u)
    i_F = 1 + (i - 1) * nu:i * nu - 1
    @view(Fy[i_F, i_F]) .= jacobian((du, x) -> dudt!(du, x, zeros(nr), 0.0),
                                du, u)::Array{Float64,2} .* (-idt)
end

@elapsed for i = 2:ng
    u = @view(pred[:, i])
    i_F = 1 + (i - 1) * nu:i * nu - 1
    @view(Fy[i_F, i * nu]) .= - dudt!(du, u, zeros(nr), 0.0)
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
