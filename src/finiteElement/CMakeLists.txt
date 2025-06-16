
if( USE_SEMCLASSIC) 
    set( finiteElement_headers
         SEMQkGLBasisFunctions.hpp
         SEMQkGLIntegrals.hpp
         classic/SEMQkGLIntegralSClassic.hpp
    )
endif()

if( USE_SEMOPTIM)
    set( finiteElement_headers
         SEMQkGLBasisFunctions.hpp
         SEMQkGLIntegrals.hpp
         optim/SEMQkGLIntegralsOptim.hpp
)

endif()
if( USE_SEMGEOS)
    set( finiteElement_headers
         SEMQkGLBasisFunctions.hpp
         SEMQkGLIntegrals.hpp
         geos/SEMQkGLIntegralsGeos.hpp
)
endif()

if( USE_SHIVA)
    set( finiteElement_headers
         SEMQkGLBasisFunctions.hpp
         SEMQkGLIntegrals.hpp
         shiva/SEMQkGLIntegralsShiva.hpp
    )
endif()


set( finiteElement_dependencies
     shiva
     )

add_library( finiteElement
             INTERFACE
             ${finiteElement_headers}
             ${finiteElement_dependencies} )

target_include_directories( finiteElement 
                            INTERFACE 
                            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
                            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
                            $<INSTALL_INTERFACE:include> )

install( FILES ${finiteElement_headers} 
         DESTINATION include/finiteElement )

if( SEMKERNELS_ENABLE_UNIT_TESTS )
    add_subdirectory( unitTests )
endif()
