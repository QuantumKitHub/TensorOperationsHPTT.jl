using TensorOperations, TensorOperationsHPTT
using TensorOperations: IndexError
using TensorOperations: DefaultAllocator, ManualAllocator
using Test
# using Random
# Random.seed!(1234567)

@testset "TensorOperationsHPTT.jl" begin
    @test get_num_hptt_threads() == 1
    @test HPPTBLAS() == HPTTBLAS(1)

    @testset "method syntax" verbose = true begin
        include("methods.jl")
    end

    set_num_hptt_threads(2)
    @test get_num_hptt_threads() == 2
    @test HPPTBLAS() == HPTTBLAS(2)

    @testset "macro with index notation" verbose = true begin
        include("tensor.jl")
    end
end
