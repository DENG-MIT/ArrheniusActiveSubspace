using OrdinaryDiffEq, Flux, Plots
using Sundials
using ForwardDiff
using ForwardDiff: jacobian, jacobian!
using LinearAlgebra
using Random
using Statistics
using Printf
using BSON: @save, @load
using Arrhenius
using YAML
using Base.Threads
using BandedMatrices
using SharedArrays

ENV["GKSwstype"] = "100"

runtime = YAML.load_file("./input.yaml")
expr_name = runtime["expr_name"]
is_restart = runtime["is_restart"]

conf = YAML.load_file("$expr_name/config.yaml")

mech = conf["mech"]
fuel = conf["fuel"]
fuel2air = Float64(conf["fuel2air"])
oxygen = conf["oxygen"]
inert = conf["inert"]
dT = Float64(conf["dTign"])
n_plot = Int64(conf["n_plot"])

if is_restart
    println("Continue to run $expr_name ...\n")
else
    println("Runing $expr_name ...\n")
end

fig_path = string(expr_name, "/figs")
ckpt_path = string(expr_name, "/checkpoint")

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
