@inbounds function residual(ts, y, p, u0, Tign, idt)
    ng = length(ts)
    pred = reshape(y, :, ng)
    F = similar(pred) * p[1]
    _ts = ts ./ idt
    i = 1
    @view(F[1:end - 1, i]) .= @views((pred[1:end - 1, i] .- u0))
    @view(F[end, i]) .= @views((pred[end, i + 1] - pred[end, i]))
    du = similar(pred[1:end - 1, 1]) * p[1]
    for i = 2:ng - 1
        @view(F[1:end - 1, i]) .=
            @views((pred[1:end - 1, i] .- pred[1:end - 1, i - 1]) ./ (_ts[i] - _ts[i - 1])) .-
            dudt!(du, @view(pred[1:end - 1, i]), p, 0.0) .* pred[end, i]
        @view(F[end, i]) .=
            @views(pred[end, i + 1] - pred[end, i])
    end
    i = ng
    @view(F[1:end - 1, i]) .=
        @views((pred[1:end - 1, i] .- pred[1:end - 1, i - 1]) ./ (_ts[i] - _ts[i - 1])) .-
        dudt!(du, @view(pred[1:end - 1, i]), p, 0.0) .* pred[end, i]
    @view(F[end, i]) .= @views((pred[end - 1, i] - Tign))
    return vcat(F...)
end

@inbounds function sensBVP(ts, pred, p)
    local ng = length(ts)
    Fp_ = zeros(ng * nu, np)
    Fy_ = BandedMatrix(zeros(ng * nu, ng * nu), (nu, nu))
    return sensBVP!(Fy_, Fp_, ts, pred, p)
end

@inbounds function sensBVP!(Fy, Fp, ts, pred, p)
    local idt = ts[end]
    local Tign = pred[end, end]
    local ng = length(ts)
    Fy .*= 0.0
    Fp .*= 0.0
    i = 1
    i_F = 1 + (i - 1) * nu:i * nu - 1
    @view(Fy[i_F, i_F])[ind_diag] .= ones_nu
    Fy[i * nu, i * nu] = -1.0
    Fy[i * nu, (i + 1) * nu] = 1.0
    du = similar(@view(pred[:, i]))

    dts = @views(ts[2:end] .- ts[1:end - 1]) ./ idt
    for i = 2:ng
        @show i
        u = @view(pred[:, i])
        i_F = 1 + (i - 1) * nu:i * nu - 1
        @view(Fy[i_F, i_F]) .= jacobian((du, x) -> dudt!(du, x, p, 0.0),
                                         du, u)::Array{Float64,2} .* (-idt)
        @view(Fy[i_F, i * nu]) .= - dudt!(du, u, p, 0.0)
        @view(Fy[i_F, i_F])[ind_diag] .+= ones_nu ./ (dts[i - 1])
        @view(Fy[i_F, i_F .- nu])[ind_diag] .+= ones_nu ./ (-dts[i - 1])
        if i < ng
            Fy[i * nu, i * nu] = -1.0
            Fy[i * nu, (i + 1) * nu] = 1.0
        else
            Fy[i * nu, i * nu - 1] = 1.0
        end
        @view(Fp[i_F, :]) .= jacobian((du, x) -> dudt!(du, u, x, 0.0),
                                       du, p)::Array{Float64,2} .* (-idt)
    end
    dydp = - @views(Fy[1:ng * nu, 1:ng * nu] \ Fp[1:ng * nu, :])
    return @view(dydp[end, :]) ./ idt
end

function sensBVP_mthread(ts, pred, p)
    idt = ts[end]
    Tign = pred[end, end]
    ng = length(ts)
    Fy = BandedMatrix(Zeros(ng * nu, ng * nu), (nu, nu));
    Fp = SharedArray{Float64}(ng * nu, np);
    Fu = SharedArray{Float64}(ng * nu, nu - 1);
    i = 1
    i_F = 1 + (i - 1) * nu:i * nu - 1
    @view(Fy[i_F, i_F])[ind_diag] .= ones_nu
    Fy[i * nu, i * nu] = -1.0
    Fy[i * nu, (i + 1) * nu] = 1.0
    du = similar(@view(pred[:, i]))

    dts = @views(ts[2:end] .- ts[1:end - 1]) ./ idt
    @threads for i = 2:ng
        # @show "Fp_$i"
        u = @view(pred[:, i])
        i_F = 1 + (i - 1) * nu:i * nu - 1
        @view(Fp[i_F, :]) .= jacobian((du, x) ->
                                        dudt!(du, @view(pred[:, i]), x, 0.0),
                                        du, p)::Array{Float64,2} .* (-idt)
        @view(Fu[i_F, :]) .= jacobian((du, x) -> dudt!(du, x, p, 0.0),
                                        du, u)::Array{Float64,2} .* (-idt)
    end
    for i = 2:ng
        # @show "Fy_$i"
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
    dydp = - Fy \ Fp
    return @view(dydp[end, :]) ./ idt
end

function downsampling(ts, pred; dT=0.1, n_max=100, verbose=false)
    ind_sample = [1]
    Ts = pred[end, :]
    _T = Ts[1]
    _t = ts[1]
    for i = 2:length(ts) - 1
        if (abs(Ts[i] - _T) > dT) & (ts[i] - _t > ts[end] / n_max)
            _T = Ts[i]
            _t = ts[i]
            push!(ind_sample, i)
        end
    end
    push!(ind_sample, length(ts))

    if verbose
        println("\n original sample size $(length(ts))
                 downsampled size $(length(ind_sample)) \n")
    end

    ts = ts[ind_sample]
    pred = pred[:, ind_sample]

    return ts, pred
end

# sensitivity calculated by Brute-force method
function sensBF_mthread(phi, P, T0, p; dT=200, dTabort=600, pdiff=5e-3)
    idt = get_idt(phi, P, T0, p; dT=dT, dTabort=dTabort)
    np = length(p);
    sens = zeros(np);
    @threads for i=1:np
        pp = deepcopy(p);
        pp[i] += pdiff;
        pidt = get_idt(phi, P, T0, pp; dT=dT, dTabort=dTabort)
        sens[i] = (pidt - idt) / pdiff
    end
    return sens ./ idt;
end