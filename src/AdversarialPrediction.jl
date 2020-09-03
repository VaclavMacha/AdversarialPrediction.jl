module AdversarialPrediction

# import for expression.jl
import Base.sqrt, Base.log, Base.exp

# packages used in metric.jl
using LinearAlgebra
using LBFGSB

# packages used in nn.jl
using Flux.Zygote
using Flux.Zygote: @adjoint
using Requires

include("expression.jl")
include("projection.jl")
include("metric.jl")
include("nn.jl")

# common metrics
include("common_metrics/CommonMetrics.jl")

# if Gurobi or CuArrays are loaded
function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("nncuda.jl")
end

export ConfusionMatrix, CM_Value, PerformanceMetric
export @metric, define, constraint
export special_case_positive!, special_case_negative!, cs_special_case_positive!, cs_special_case_negative!
export compute_metric, compute_constraints, objective
export ap_objective

end # module
