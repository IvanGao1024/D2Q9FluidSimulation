using Pkg

function add_package(package_name)
    if !haskey(Pkg.installed(), Symbol(package_name))
        Pkg.add(package_name)
    end
    @eval using $(Symbol(package_name))
end

add_package("Plots")
# add_package("ForwardDiff")
# add_package("LinearAlgebra")
# add_package("LaTeXStrings")
# add_package("BenchmarkTools")
# add_package("SpecialFunctions")
# add_package("Printf")
# add_package("PrettyTables")
# add_package("Logging")
# # add_package("OrdinaryDiffEq")
# add_package("StaticArrays")
# add_package("SparseArrays")
# add_package("Random")
# add_package("Test")
# add_package("NLsolve")
# add_package("Roots")

module Utils
    # export nice, DOG
    # struct Dog end      # singleton type, not exported
    # const DOG = Dog()   # named instance, exported
    # nice(x) = "nice $x" # function, exported
end;
