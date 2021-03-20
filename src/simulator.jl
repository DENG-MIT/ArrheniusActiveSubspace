
@inbounds function dudt!(du, u, p, t)
    Y = @view(u[1:ns])
    T = u[end]
    mean_MW = 1.0 / dot(Y, 1 ./ gas.MW)
    ρ_mass = P / R / T * mean_MW
    X = Y2X(gas, Y, mean_MW)
    C = Y2C(gas, Y, ρ_mass)
    cp_mole, cp_mass = get_cp(gas, T, X, mean_MW)
    h_mole = get_H(gas, T, Y, X)
    S0 = get_S(gas, T, P, X)
    # _p = reshape(p, nr, 3)
    # kp = @. @views(exp(_p[:, 1] + _p[:, 2] * log(T) - _p[:, 3] * 4184.0 / R / T))
    kp = exp.(p)
    qdot = wdot_func(gas.reaction, T, C, S0, h_mole; get_qdot=true) .* kp
    wdot = gas.reaction.vk * qdot
    Ydot = wdot / ρ_mass .* gas.MW
    Tdot = -dot(h_mole, wdot) / ρ_mass / cp_mass
    du .= vcat(Ydot, Tdot)
    return du
end

function make_prob(T0, P, phi, p; tfinal=1.0)
    X0 = zeros(ns);
    X0[species_index(gas, fuel)] = phi
    X0[species_index(gas, oxygen)] = fuel2air
    X0[species_index(gas, inert)] = fuel2air * 3.76
    X0 = X0 ./ sum(X0);
    Y0 = X2Y(gas, X0, dot(X0, gas.MW));
    u0 = vcat(Y0, T0);

    prob = ODEProblem(dudt!, u0, (0.0, tfinal), p);
    prob
end

function get_idt(T0, P, phi, p;
                 dT=400, dTabort=800, doplot=false, tfinal=1.0, saveat=[])

    prob = make_prob(T0, P, phi, p; tfinal=tfinal)

    condition(u, t, integrator) = u[end] > T0 + dTabort

    affect!(integrator) = terminate!(integrator)

    _cb = DiscreteCallback(condition, affect!)

    sol = solve(prob, CVODE_BDF(), saveat=saveat,
                reltol=1e-6, abstol=1e-9, callback=_cb);

    ind_ign = get_ind_ign(sol; dT=dT)
    ts = sol.t[1:ind_ign]
    pred = clamp.(Array(sol)[:, 1:ind_ign], 0, Inf);
    return ts, pred
end

function get_ind_ign(sol; dT=400)
    if maximum(sol[end, :]) - sol[end, 1] > dT
        return findfirst(sol[end, :] .> sol[end, 1] + dT)
    else
        println("Warning, no ignition")
        return length(sol.t)
    end
end
