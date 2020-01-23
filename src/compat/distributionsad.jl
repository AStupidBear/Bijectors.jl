using .DistributionsAD: TuringWishart, TuringInverseWishart, MultipleContinuousUnivariate,
                        MatrixContinuousUnivariate, MultipleContinuousMultivariate,
                        ProductVectorContinuousUnivariate,
                        ProductMatrixContinuousUnivariate,
                        ProductVectorContinuousMultivariate

# TuringWishart

function logpdf_with_trans(
    d::TuringWishart,
    X::AbstractMatrix{<:Real},
    transform::Bool
)
    _logpdf_with_trans_pd(d, X, transform)
end
function link(d::TuringWishart, X::AbstractMatrix{<:Real})
    _link_pd(d, X)
end
function invlink(d::TuringWishart, Y::AbstractMatrix{<:Real})
    _invlink_pd(d, Y)
end
function getlogp(d::TuringWishart, Xcf, X)
    return 0.5 * ((d.df - (dim(d) + 1)) * logdet(Xcf) - tr(d.chol \ X)) - d.c0
end

# TuringInverseWishart

function logpdf_with_trans(
    d::TuringInverseWishart,
    X::AbstractMatrix{<:Real},
    transform::Bool
)
    _logpdf_with_trans_pd(d, X, transform)
end
function link(d::TuringInverseWishart, X::AbstractMatrix{<:Real})
    _link_pd(d, X)
end
function invlink(d::TuringInverseWishart, Y::AbstractMatrix{<:Real})
    _invlink_pd(d, Y)
end
function getlogp(d::TuringInverseWishart, Xcf, X)
    Ψ = d.S
    return -0.5 * ((d.df + dim(d) + 1) * logdet(Xcf) + tr(Xcf \ Ψ)) - d.c0
end

# Multi and ArrayDist

function logpdf_with_trans(
    dist::MultipleContinuousUnivariate,
    x::AbstractVector{<:Real},
    istrans::Bool,
)
    return sum(logpdf_with_trans(dist.dist, x, istrans))
end
function link(
    dist::MultipleContinuousUnivariate,
    x::AbstractVector{<:Real},
)
    return link(dist.dist, x)
end
function invlink(
    dist::MultipleContinuousUnivariate,
    x::AbstractVector{<:Real},
)
    return invlink(dist.dist, x)
end

function logpdf_with_trans(
    dist::MatrixContinuousUnivariate,
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    return sum(logpdf_with_trans.(dist.dist, x, istrans))
end
function link(
    dist::MatrixContinuousUnivariate,
    x::AbstractMatrix{<:Real},
)
    return link(dist.dist, x)
end
function invlink(
    dist::MatrixContinuousUnivariate,
    x::AbstractMatrix{<:Real},
)
    return invlink(dist.dist, x)
end

function logpdf_with_trans(
    dist::MultipleContinuousMultivariate,
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    return sum(logpdf_with_trans(dist.dist, x, istrans))
end
function link(
    dist::MultipleContinuousMultivariate,
    x::AbstractMatrix{<:Real},
)
    return link(dist.dist, x)
end
function invlink(
    dist::MultipleContinuousMultivariate,
    x::AbstractMatrix{<:Real},
)
    return invlink(dist.dist, x)
end

function logpdf_with_trans(
    dist::ProductVectorContinuousMultivariate,
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    return sum(logpdf_with_trans.(dist.dists, [x[:,i] for i in 1:size(x, 2)], istrans))
end
function link(
    dist::ProductVectorContinuousMultivariate,
    x::AbstractMatrix{<:Real},
)
    return reduce(hcat, link.(dist.dists, [x[:,i] for i in 1:size(x, 2)]))
end
function invlink(
    dist::ProductVectorContinuousMultivariate,
    x::AbstractMatrix{<:Real},
)
    return reduce(hcat, invlink.(dist.dists, [x[:,i] for i in 1:size(x, 2)]))
end

function logpdf_with_trans(
    dist::ProductVectorContinuousUnivariate,
    x::AbstractVector{<:Real},
    istrans::Bool,
)
    return sum(logpdf_with_trans.(dist.dists, x, istrans))
end
function link(
    dist::ProductVectorContinuousUnivariate,
    x::AbstractVector{<:Real},
)
    return link.(dist.dists, x)
end
function invlink(
    dist::ProductVectorContinuousUnivariate,
    x::AbstractVector{<:Real},
)
    return invlink.(dist.dists, x)
end

function logpdf_with_trans(
    dist::ProductMatrixContinuousUnivariate,
    x::AbstractMatrix{<:Real},
    istrans::Bool,
)
    return sum(logpdf_with_trans.(dist.dists, x, istrans))
end
function link(
    dist::ProductMatrixContinuousUnivariate,
    x::AbstractMatrix{<:Real},
)
    return link.(dist.dists, x)
end
function invlink(
    dist::ProductMatrixContinuousUnivariate,
    x::AbstractMatrix{<:Real},
)
    return invlink.(dist.dists, x)
end
