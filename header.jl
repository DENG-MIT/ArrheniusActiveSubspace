using Arrhenius
using Sundials
using LinearAlgebra
using OrdinaryDiffEq
using ForwardDiff
using ForwardDiff: jacobian, jacobian!
using YAML
using BSON: @save, @load
using Plots, Printf, Random
using Statistics
using ProgressBars
using Base.Threads
using BandedMatrices
using SharedArrays
using Dierckx # for interpolation

ENV["GKSwstype"] = "100"

# load input.yaml
runtime = YAML.load_file("./input.yaml")
expr_name = runtime["expr_name"]
is_restart = runtime["is_restart"]

# load $expr_name/config.yaml
conf = YAML.load_file("$expr_name/config.yaml")
mech = conf["mech"]
fuel = conf["fuel"]
fuel2air = Float64(conf["fuel2air"])
oxygen = conf["oxygen"]
inert = conf["inert"]

dT = Float64(conf["dTign"])
dTabort = Float64(conf["dTabort"])
method = conf["method"]

n_plot = Int64(conf["n_plot"])

if is_restart
    println("Continue to run $expr_name ...\n")
else
    println("Runing $expr_name ...\n")
end

exp_path = string(expr_name)
fig_path = string(exp_path, "/figs")
ckpt_path = string(exp_path, "/checkpoint")

if !is_restart
    if ispath(fig_path)
        rm(fig_path, recursive=true)
    end
    if ispath(ckpt_path)
        rm(ckpt_path, recursive=true)
    end
end

if ispath(fig_path) == false
    mkdir(fig_path)
    mkdir(string(fig_path, "/conditions"))
end

if ispath(ckpt_path) == false
    mkdir(ckpt_path)
end
