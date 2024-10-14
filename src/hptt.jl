# HPTT wrappers
function hptt_add!(B::AbstractArray{T,0},
                   A::AbstractArray{T,0}, pA::Tuple{}, conjA::Bool,
                   α=one(T), β=zero(T);
                   numthreads=get_num_hptt_threads()) where {T<:BlasFloat}
    scalarA = conjA ? conj(A[]) : A[]
    if iszero(β)
        B[] = α * scalarA
    else
        B[] = α * scalarA + β * B[]
    end
    return B
end
function hptt_add!(B::AbstractArray{T,N},
                   A::AbstractArray{T,N}, pA::NTuple{N,Int}, conjA::Bool,
                   α=one(T), β=zero(T);
                   numthreads=get_num_hptt_threads()) where {N,T<:BlasFloat}
    sizeA, stridesA, qA = parse_hptt_array(A)
    sizeB, stridesB, qB = parse_hptt_array(B)
    iqA = TupleTools.invperm(qA)
    perm = TupleTools.getindices(TupleTools.getindices(iqA, pA), qB)

    if stridesA[1] != 1 || stridesB[1] != 1
        szA = (1, sizeA...)
        szB = (1, sizeB...)
        oszA = strides2outersize((1, stridesA...), szA)
        isnothing(oszA) &&
            throw(ArgumentError(lazy"Memory layout of A with size $sizeA and strides $stridesA is not supported by HPTT"))
        oszB = strides2outersize((1, stridesB...), szB)
        isnothing(oszB) &&
            throw(ArgumentError(lazy"Memory layout of B with size $sizeB and strides $stridesB is not supported by HPTT"))
        p = (1, (1 .+ perm)...)

        @show sizeA, stridesA, oszA
        @show sizeB, stridesB, oszB
        return _tensor_transpose!(p, N + 1,
                                  convert(T, α), conjA, A, szA, oszA,
                                  convert(T, β), B, oszB,
                                  numthreads, 0)
    else
        outersizeA = strides2outersize(stridesA, sizeA)
        isnothing(outersizeA) &&
            throw(ArgumentError(lazy"Memory layout of A with size $sizeA and strides $stridesA is not supported by HPTT"))
        outersizeB = strides2outersize(stridesB, sizeB)
        isnothing(outersizeB) &&
            throw(ArgumentError(lazy"Memory layout of B with size $sizeB and strides $stridesB is not supported by HPTT"))

        @show sizeA, stridesA, outersizeA
        @show sizeB, stridesB, outersizeB

        return _tensor_transpose!(perm, N,
                                  convert(T, α), conjA, A, sizeA, outersizeA,
                                  convert(T, β), B, outersizeB,
                                  numthreads, 0)
    end
end

# Literal HPTT wrapper
for (fname, elty) in ((:dTensorTranspose, :Float64),
                      (:sTensorTranspose, :Float32),
                      (:zTensorTranspose, :ComplexF64),
                      (:cTensorTranspose, :ComplexF32))
    sname = Expr(:quote, fname)
    @eval begin
        # void ?TensorTranspose( const int *perm, const int dim,
        # const float alpha, const float *A, const int *sizeA, const int *outerSizeA,
        # const float beta,        float *B,                   const int *outerSizeB,
        # const int numThreads, const int useRowMajor);

        function _tensor_transpose!(p, N,
                                    α::$elty, conjA, A::AbstractArray{$elty}, szA, oszA,
                                    β::$elty, B::AbstractArray{$elty}, oszB,
                                    numthreads, useRowMajor)
            sizeA = collect(Cint, szA)
            outersizeA = collect(Cint, oszA)
            outersizeB = collect(Cint, oszB)
            perm = collect(Cint, p) .- one(Cint)

            @show sizeA, outersizeA, outersizeB, perm

            if $elty <: Complex
                ccall(($sname, libhptt), Cvoid,
                      (Ptr{Cint}, Cint,
                       $elty, Cuchar, Ptr{$elty}, Ptr{Cint}, Ptr{Cint},
                       $elty, Ptr{$elty}, Ptr{Cint},
                       Cint, Cint),
                      perm, N,
                      α, conjA, A, sizeA, outersizeA,
                      β, B, outersizeB,
                      numthreads, useRowMajor)
            else
                ccall(($sname, libhptt), Cvoid,
                      (Ptr{Cint}, Cint,
                       $elty, Ptr{$elty}, Ptr{Cint}, Ptr{Cint},
                       $elty, Ptr{$elty}, Ptr{Cint},
                       Cint, Cint),
                      perm, N,
                      α, A, sizeA, outersizeA,
                      β, B, outersizeB,
                      numthreads, useRowMajor)
            end
            return B
        end
    end
end

# Auxiliary functions
function parse_hptt_array(A::AbstractArray)
    stridesA = strides(A)
    qA = TupleTools.sortperm(stridesA)
    stridesA = TupleTools.getindices(stridesA, qA)
    sizeA = TupleTools.getindices(size(A), qA)
    return sizeA, stridesA, qA
end

strides2outersize(str::Tuple{}, sz::Tuple{}) = ()
function strides2outersize(str::Tuple{Int}, sz::Tuple{Int})
    str[1] == 1 || return nothing
    return sz
end
function strides2outersize(str::NTuple{N}, sz::NTuple{N}) where {N}
    str[1] == 1 || return nothing
    osz1 = str[2]
    newstr = Base.tail(str)
    for k in 1:(N - 1)
        d, r = divrem(newstr[k], osz1)
        r == 0 || return nothing
        newstr = Base.setindex(newstr, d, k)
    end
    strtail = strides2outersize(newstr, Base.tail(sz))
    isnothing(strtail) && return nothing
    return (osz1, strtail...)
end
