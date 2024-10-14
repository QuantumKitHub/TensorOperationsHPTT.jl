module TensorOperationsHPTT

using TensorOperations
using TensorOperations: StridedView, DefaultAllocator
using TensorOperations: istrivialpermutation, BlasFloat, linearize
using TensorOperations: argcheck_tensoradd, dimcheck_tensoradd,
                        argcheck_tensortrace, dimcheck_tensortrace,
                        argcheck_tensorcontract, dimcheck_tensorcontract
using TupleTools
using HPTT_jll

export HPTTBLAS, get_num_hptt_threads, set_num_hptt_threads

const HPTT_NUM_THREADS = Ref(1)

get_num_hptt_threads() = HPTT_NUM_THREADS[]
set_num_hptt_threads(n) = HPTT_NUM_THREADS[] = n

struct HPTTBLAS <: TensorOperations.AbstractBackend
    numthreads::Int
end
HPTTBLAS() = HPTTBLAS(get_num_hptt_threads())

include("hptt.jl")
include("strided.jl")

end
