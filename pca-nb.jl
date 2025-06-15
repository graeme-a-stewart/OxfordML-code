### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# ╔═╡ f19618d5-458e-484d-9aeb-1623d971de6d
using LinearAlgebra

# ╔═╡ db5c3b43-5243-49b6-acb8-c816cb294beb
using Statistics

# ╔═╡ acf4c768-3a18-45fd-8bf3-431d1e06ce3a
md"""# Notebook on Principal Component Analysis

Support notebook for the Oxford ML course
"""

# ╔═╡ 7a469ecb-7e76-41a6-a025-76562dd07977
md"## Inital small example"

# ╔═╡ de002568-1f06-4471-adb3-86f0dba83c10
M = [1 1; 1 1]

# ╔═╡ 381dc416-264a-4cbe-bd6d-7e1e77d8cd7e
eigvals(M)

# ╔═╡ cc6bc969-0583-4bc6-971b-cd6ced660782
eigvecs(M)

# ╔═╡ 93fc32aa-b827-4fe9-af53-f7c9278dc0bb
md"## Animal size data"

# ╔═╡ 5e181f27-dc61-483b-97e3-9014bd635b02
A = [8.3 9.1 4.0 11.5; 3.7 5.1 1.0 6.7; 10.3 12.3 5.2 14.2]

# ╔═╡ cb39f825-0565-4b0a-96dd-b9ad651aa349
SA = A' * A

# ╔═╡ 6624749e-c65c-4235-b2b7-242eedf48ef6
eigvals(SA)

# ╔═╡ f60f4c1b-c705-4b33-bc99-b3961f3b12e9
eigvecs(SA)

# ╔═╡ 7f0a3c7f-d840-4316-ae40-d6a06ff31c35
v0 = [1,1,1,1]

# ╔═╡ 6f5d03a2-890e-4142-8a1a-2390fe957a96
v0p = SA * v0

# ╔═╡ 9912122f-69f0-4914-b176-e90fbc006a5d
v1 = v0p / sqrt(v0p'*v0p)

# ╔═╡ 06fbb10e-85e6-46b8-abbf-1672ddf0b652
(v1 * v1') * eigvals(SA)[4]

# ╔═╡ a180f2de-97aa-4e43-91f9-550874ea1e91
md"## Exercise Example 1"

# ╔═╡ 41c218e4-4e4b-482c-9512-364d0c2796f0
M1 = [0.29 0.58 0.8; 0.58 1.16 1.6; 0.8 1.6 2.21]

# ╔═╡ 98b0be12-7179-4893-bcb4-6f069bf4dfd0
md"Calculate principle eigenvector by repeated multiplication"

# ╔═╡ 747d6bcd-e1a2-4177-8c1b-aabddd87a829
V0 = [1,1,1]

# ╔═╡ 1886f7c7-88fc-4272-b140-eaa3e3e08f94
V0p = M1 * V0

# ╔═╡ 9b18fa66-4081-4164-a302-6978ffb568b2
V1 = V0p / sqrt(dot(V0p,V0p))

# ╔═╡ d2772a3a-4995-48eb-b7e7-47b67442ae8c
V1p = M1 * V1

# ╔═╡ 2d3bace1-99dd-4508-80df-00002b01686a
V2 = V1p / sqrt(dot(V1p, V1p))

# ╔═╡ ea1b3e30-f069-42f1-b9eb-70ecfb23e10d
V2p = M1 * V2

# ╔═╡ 7a678bf2-5e9f-4ded-9217-da8464b2a44d
V3 = V2p / sqrt(dot(V2p, V2p))

# ╔═╡ ff5f19fa-dcbe-48ef-a792-a17cceef4b4b
md"Find eigenvalue by seeing how much the vector is scaled by..."

# ╔═╡ cd680fef-685e-4396-b27b-853a43334bd3
M1 * V3 ./ V3

# ╔═╡ 057b8e5a-8384-4563-a85e-7ef780d24dbe
EV1 = mean(M1 * V3 ./ V3)

# ╔═╡ 077695a1-8c97-477a-a5f2-fd5d13c0950e
md"Now reconstruct the original matrix"

# ╔═╡ 0067735d-5f5b-43f1-9718-e6e3f8adfde3
RecoM1 = (V3 * V3') * EV1

# ╔═╡ 30c93e18-bcb4-4cbc-80ef-8804666ea647
md"How good is this?"

# ╔═╡ ba777844-1d38-45d4-9174-e58d1d6754a6
reldiff = (M1 .- RecoM1) ./ M1

# ╔═╡ 0244a55d-0261-4920-89c4-c127589454c6
rms = sqrt(mean(reldiff .* reldiff) / prod(size(reldiff)))

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.5"
manifest_format = "2.0"
project_hash = "8c3efc2a8bec7327ac93e0344d9faedc429a153e"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"
"""

# ╔═╡ Cell order:
# ╟─acf4c768-3a18-45fd-8bf3-431d1e06ce3a
# ╠═f19618d5-458e-484d-9aeb-1623d971de6d
# ╠═db5c3b43-5243-49b6-acb8-c816cb294beb
# ╟─7a469ecb-7e76-41a6-a025-76562dd07977
# ╠═de002568-1f06-4471-adb3-86f0dba83c10
# ╠═381dc416-264a-4cbe-bd6d-7e1e77d8cd7e
# ╠═cc6bc969-0583-4bc6-971b-cd6ced660782
# ╟─93fc32aa-b827-4fe9-af53-f7c9278dc0bb
# ╠═5e181f27-dc61-483b-97e3-9014bd635b02
# ╠═cb39f825-0565-4b0a-96dd-b9ad651aa349
# ╠═6624749e-c65c-4235-b2b7-242eedf48ef6
# ╠═f60f4c1b-c705-4b33-bc99-b3961f3b12e9
# ╠═7f0a3c7f-d840-4316-ae40-d6a06ff31c35
# ╠═6f5d03a2-890e-4142-8a1a-2390fe957a96
# ╠═9912122f-69f0-4914-b176-e90fbc006a5d
# ╠═06fbb10e-85e6-46b8-abbf-1672ddf0b652
# ╟─a180f2de-97aa-4e43-91f9-550874ea1e91
# ╠═41c218e4-4e4b-482c-9512-364d0c2796f0
# ╟─98b0be12-7179-4893-bcb4-6f069bf4dfd0
# ╠═747d6bcd-e1a2-4177-8c1b-aabddd87a829
# ╠═1886f7c7-88fc-4272-b140-eaa3e3e08f94
# ╠═9b18fa66-4081-4164-a302-6978ffb568b2
# ╠═d2772a3a-4995-48eb-b7e7-47b67442ae8c
# ╠═2d3bace1-99dd-4508-80df-00002b01686a
# ╠═ea1b3e30-f069-42f1-b9eb-70ecfb23e10d
# ╠═7a678bf2-5e9f-4ded-9217-da8464b2a44d
# ╟─ff5f19fa-dcbe-48ef-a792-a17cceef4b4b
# ╠═cd680fef-685e-4396-b27b-853a43334bd3
# ╠═057b8e5a-8384-4563-a85e-7ef780d24dbe
# ╟─077695a1-8c97-477a-a5f2-fd5d13c0950e
# ╠═0067735d-5f5b-43f1-9718-e6e3f8adfde3
# ╟─30c93e18-bcb4-4cbc-80ef-8804666ea647
# ╠═ba777844-1d38-45d4-9174-e58d1d6754a6
# ╠═0244a55d-0261-4920-89c4-c127589454c6
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
