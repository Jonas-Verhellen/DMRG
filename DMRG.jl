using LinearAlgebra
using SparseArrays
using KrylovKit

# Algorithm Parameters
sites = 10;
max_eigenvalues = 10;
J, Jz = 1.0, 1.0;

# Local Operators
Id = Matrix{Float64}(I, 2, 2)
Sz = [0.5 0.0; 0.0 -0.5]
Sp = [0.0 0.0; 1.0 0.0]
Sm = [0.0 1.0; 0.0 0.0]

# DMRG block
mutable struct DMRG_block
	H::Array{Float64,2}
	Sz::Array{Float64,2}
	Sp::Array{Float64,2}
	Sm::Array{Float64,2}
	Id::Array{Float64,2}
end

function add_site!(J::Float64, Jz::Float64, block::DMRG_block)
	block.H = (J/2)*(kron(block.Sp, Sm) + kron(block.Sm, Sp)) + Jz*(kron(block.H, Id) + kron(block.Sz, Sz))
	block.Sz, block.Sp, block.Sm = kron(block.Id, Sz), kron(block.Id, Sp), kron(block.Id, Sm)
	block.Id = kron(block.Id, Id)
end

function superblock_hamiltonian(J::Float64, Jz::Float64, block::DMRG_block)
	return Jz*(kron(block.H, block.Id) + kron(block.Id, block.H) + kron(block.Sz , block.Sz)) +
	 	   (J/2)*(kron(block.Sp, block.Sm) + kron(block.Sm, block.Sp))
end

function entanglement(eigenvalues::Array{Float64,1})
	return sum(x -> iszero(x) ? 0.0 : -(x^2)*log(x^2), eigenvalues)
end

function truncation(eigenvalues::Array{Float64,1}, eigenvectors::Array{Array{Float64,1},1}, max_eigenvalues::Int64)
	operator = hcat(eigenvectors[max(0, end - max_eigenvalues) + 1 : end] ...)
	error = 1 - sum(eigenvalues[max(0, end - max_eigenvalues) + 1 : end])
	return operator, error
end

function DMRG(sites::Int64, max_eigenvalues::Int64, J::Float64, Jz::Float64)

	# One-site block, Hamiltonian is zero
	site = 2 # due to reflection symmetry
	block = DMRG_block(zeros(2,2), Sz, Sp, Sm, Id)
	previous_energy = -0.75

	while site < sites

		# Add a site to the block (left and right)
		site = site + 2
		add_site!(J, Jz, block)

		# Diagonlize the superblock Hamiltonian
		Hₛ = superblock_hamiltonian(J::Float64, Jz::Float64, block::DMRG_block)
		eigenvalues, eigenvectors, info = eigsolve(Hₛ, 1, :SR; ishermitian = true)

		# Groundstate energy, wavefunction and reduced density matrix
		energy, groundstate = eigenvalues[1], eigenvectors[1]
		energy_per_bond = (energy - previous_energy)/2.0;

		# Creating the density matrix
		groundstate_matrix = Matrix(reshape(groundstate, size(block.H)))
		ρ = groundstate_matrix*groundstate_matrix'

		# Calculating the entanglement entropy
		eigenvalues, eigenvectors, info = eigsolve(ρ, size(ρ)[1], :SR; krylovdim = size(ρ)[1], ishermitian = true)
		entanglement_entropy = entanglement(eigenvalues)

		# Truncate the block operators
		truncation_operator, truncation_error = truncation(eigenvalues, eigenvectors, max_eigenvalues)
		map(x -> truncation_operator'*x*truncation_operator, [block.H, block.Sz, block.Sp, block.Sm, block.Id])

		#  Print information about current step
		previous_energy = energy
 		@show energy, energy_per_bond, entanglement_entropy, truncation_error
	end

	return block

end

DMRG(sites, max_eigenvalues, J, Jz)

# Exact energy per site, for comparison
exact_energy = -log(2) + 0.25
