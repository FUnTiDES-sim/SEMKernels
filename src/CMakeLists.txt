
message( "Adding SHIVA" )

add_subdirectory(Shiva)

if( SHIVA_ENABLE_BLT )
  shiva_add_code_checks( PREFIX SEMKernels
                        EXCLUDES "Shiva/*" )
endif()

add_subdirectory(finiteElement)
