# negative of obj, since the original formulation is max_\theta
# whereas the optimizer only support minimization

# all the opt in the metric needs to be done in cpu

# objectve: cuda vector
function ap_objective(ps::CUDA.CuVector, y::CUDA.CuVector, pm::PerformanceMetric; args...)
    psc = CUDA.collect(ps)    # to cpu
    yc = CUDA.collect(y)      # to cpu

    obj, _ = objective(pm, psc, yc; args...)
    obj = obj + dot(psc, yc)
    return -obj
end

# custom gradient
@adjoint function ap_objective(ps::CUDA.CuVector, y::CUDA.CuVector, pm::PerformanceMetric; args...)
    psc = CUDA.collect(ps)    # to cpu
    yc = CUDA.collect(y)      # to cpu

    obj, q = objective(pm, psc, yc; args...)
    obj = obj + dot(psc, yc)

    grad = CUDA.cu(q - yc)   # to cuda
    return -obj, Δ -> (Δ * grad, nothing, nothing)
end
