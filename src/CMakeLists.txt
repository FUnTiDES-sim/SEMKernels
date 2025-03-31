add_subdirectory(Shiva)

shiva_add_code_checks( PREFIX SEMKernels
                       EXCLUDES "Shiva/*" )
add_subdirectory(finiteElement)
