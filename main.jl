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

prob = make_prob(T0, P, phi, p);
@time ts, pred = get_idt(T0, P, phi, p; dT=50, dTabort=600, doplot=true);

# ng = length(ts);
# Fp = zeros(ng * nu, np);
# Fy = BandedMatrix(zeros(ng * nu, ng * nu), (nu, nu));

# grad = sensBVP!(Fy, Fp, ts, pred, p)
# @time sensBVP_mthreadfile(ts, pred, p)

idt = ts[end]
Tign = pred[end, end]
ng = length(ts)
Fy = BandedMatrix(Zeros(ng * nu, ng * nu), (nu, nu));
Fp = SharedArray{Float64}(ng * nu, np);
Fu = SharedArray{Float64}(ng * nu, nu-1);
i = 1
i_F = 1 + (i - 1) * nu:i * nu - 1
@view(Fy[i_F, i_F])[ind_diag] .= ones_nu
Fy[i * nu, i * nu] = -1.0
Fy[i * nu, (i + 1) * nu] = 1.0
du = similar(@view(pred[:, i]))

dts = @views(ts[2:end] .- ts[1:end - 1]) ./ idt
@threads for i = 2:ng
    @show "Fp_$i"
    u = @view(pred[:, i])
    i_F = 1 + (i - 1) * nu:i * nu - 1
    @view(Fp[i_F, :]) .= jacobian((du, x) ->
                                    dudt!(du, @view(pred[:, i]), x, 0.0),
                                    du, p)::Array{Float64,2} .* (-idt)
    @view(Fu[i_F, :]) .= jacobian((du, x) -> dudt!(du, x, p, 0.0),
                                    du, u)::Array{Float64,2} .* (-idt)
end
for i = 2:ng
    @show "Fy_$i"
    u = @view(pred[:, i])
    i_F = 1 + (i - 1) * nu:i * nu - 1
    @view(Fy[i_F, i_F]) .= @view(Fu[i_F, :])
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

# using SparseArrays
# Fps = sparse(Fp);
# Fys = sparse(Fy);

grad = SharedArray{Float64}(np);


@time dpi = Fy \ @view(Fp[:, 10]);
# @time dpi = Fy \ Fp;

@threads for i = 1:np
    grad[i] = (Fy \ @view(Fp[:, i]))[end]
end

# TODO: optimize linear solver for large matrix.

# dydp = - @views(Fy[1:ng * nu, 1:ng * nu] \ Fp[1:ng * nu, :])