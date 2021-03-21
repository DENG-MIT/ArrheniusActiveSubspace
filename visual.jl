function check_sol(T0, P, phi, p; i_exp=0)
    l_plt = []
    ts, pred = get_idt(T0, P, phi, p; dT=dT)
    ts .+= 1.e-6
    plt = plot(ts, pred[end, :], lw=2, label="Train")
    ylabel!(plt, "Temperature [K]")
    xlabel!(plt, "Time [s]")
    title!(plt, @sprintf("%s, IDT=%.2e [s] \n @%.1f K, %.1f atm, phi=%.1f",
                fuel, ts[end], T0, P / one_atm, phi))
    push!(l_plt, plt)
    for s in [fuel, oxygen]
        plt =
            plot(ts,
            pred[species_index(gas, "$s"), :],
            lw=2, label="Train")
        ylabel!(plt, "Y $s")
        xlabel!(plt, "Time [s]")
        push!(l_plt, plt)
    end
    pltsum = plot(l_plt..., legend=false, framestyle=:box, xscale=:log10)

    png(pltsum, string(fig_path, "/conditions/sol_$i_exp"))
end

function plot_sol(sol)
    plt = Plots.scatter(sol.t, sol[end, :]);
    xlabel!(plt, "Time [s]");
    ylabel!(plt, "T [K]");
    ind_ign = get_ind_ign(sol; dT=dT)
    idt = sol.t[ind_ign]
    xlims!(plt, (0.0, idt * 2.0))
    png(plt, string(fig_path, "/conditions/sol_"));
end