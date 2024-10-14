#-------------------------------------------------------------------------------------------
# Force strided implementation on AbstractArray instances with HPTTBLAS backend
#-------------------------------------------------------------------------------------------
const SV = StridedView
function TensorOperations.tensoradd!(C::AbstractArray,
                                     A::AbstractArray, pA::Index2Tuple, conjA::Bool,
                                     α::Number, β::Number,
                                     backend::HPTTBLAS, allocator=DefaultAllocator())
    # resolve conj flags and absorb into StridedView constructor to avoid type instabilities later on
    if conjA
        stridedtensoradd!(SV(C), conj(SV(A)), pA, α, β, backend, allocator)
    else
        stridedtensoradd!(SV(C), SV(A), pA, α, β, backend, allocator)
    end
    return C
end

function TensorOperations.tensortrace!(C::AbstractArray,
                                       A::AbstractArray, p::Index2Tuple, q::Index2Tuple,
                                       conjA::Bool,
                                       α::Number, β::Number,
                                       backend::HPTTBLAS, allocator=DefaultAllocator())
    # resolve conj flags and absorb into StridedView constructor to avoid type instabilities later on
    if conjA
        stridedtensortrace!(SV(C), conj(SV(A)), p, q, α, β, backend, allocator)
    else
        stridedtensortrace!(SV(C), SV(A), p, q, α, β, backend, allocator)
    end
    return C
end

function TensorOperations.tensorcontract!(C::AbstractArray,
                                          A::AbstractArray, pA::Index2Tuple, conjA::Bool,
                                          B::AbstractArray, pB::Index2Tuple, conjB::Bool,
                                          pAB::Index2Tuple,
                                          α::Number, β::Number,
                                          backend::HPTTBLAS, allocator=DefaultAllocator())
    # resolve conj flags and absorb into StridedView constructor to avoid type instabilities later on
    if conjA && conjB
        stridedtensorcontract!(SV(C), conj(SV(A)), pA, conj(SV(B)), pB, pAB, α, β,
                               backend, allocator)
    elseif conjA
        stridedtensorcontract!(SV(C), conj(SV(A)), pA, SV(B), pB, pAB, α, β,
                               backend, allocator)
    elseif conjB
        stridedtensorcontract!(SV(C), SV(A), pA, conj(SV(B)), pB, pAB, α, β,
                               backend, allocator)
    else
        stridedtensorcontract!(SV(C), SV(A), pA, SV(B), pB, pAB, α, β,
                               backend, allocator)
    end
    return C
end

#-------------------------------------------------------------------------------------------
# StridedView implementation
#-------------------------------------------------------------------------------------------
function stridedtensoradd!(C::StridedView{T},
                           A::StridedView{T}, pA::Index2Tuple,
                           α::Number, β::Number,
                           backend::HPTTBLAS,
                           allocator=DefaultAllocator()) where {T<:BlasFloat}
    argcheck_tensoradd(C, A, pA)
    dimcheck_tensoradd(C, A, pA)
    if Base.mightalias(C, A)
        throw(ArgumentError("output tensor must not be aliased with input tensor"))
    end

    numthreads = backend.numthreads

    p = linearize(pA)

    if C.op == conj && A.op == conj
        hptt_add!(C, A, p, false, conj(α), conj(β); numthreads=numthreads)
    elseif C.op == conj
        hptt_add!(C, A, p, true, conj(α), conj(β); numthreads=numthreads)
    elseif A.op == conj
        hptt_add!(C, A, p, true, α, β; numthreads=numthreads)
    else
        hptt_add!(C, A, p, false, α, β; numthreads=numthreads)
    end
    return C
end

function stridedtensortrace!(C::StridedView,
                             A::StridedView, p::Index2Tuple, q::Index2Tuple,
                             α::Number, β::Number,
                             backend::HPTTBLAS, allocator=DefaultAllocator())
    argcheck_tensortrace(C, A, p, q)
    dimcheck_tensortrace(C, A, p, q)

    Base.mightalias(C, A) &&
        throw(ArgumentError("output tensor must not be aliased with input tensor"))

    error("Trace not yet supported with HPTTBLAS backend")
    return C
end

function stridedtensorcontract!(C::StridedView,
                                A::StridedView, pA::Index2Tuple,
                                B::StridedView, pB::Index2Tuple,
                                pAB::Index2Tuple,
                                α::Number, β::Number,
                                backend::HPTTBLAS, allocator=DefaultAllocator())
    argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    (Base.mightalias(C, A) || Base.mightalias(C, B)) &&
        throw(ArgumentError("output tensor must not be aliased with input tensor"))

    TensorOperations.blas_contract!(C, A, pA, B, pB, pAB, α, β, backend, allocator)
    return C
end
