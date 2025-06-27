add_subdirectory(discretization/Shiva)

shiva_add_code_checks( PREFIX SEMKernels
                       EXCLUDES "Shiva/*" )
add_subdirectory(discretization/finiteElement)
add_subdirectory(solver/acoustic)
