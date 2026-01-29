set( finiteElement_headers
     makutu/LagrangeBasis1.hpp
     makutu/LagrangeBasis2.hpp
     makutu/LagrangeBasis3GL.hpp
     makutu/LagrangeBasis4GL.hpp
     makutu/LagrangeBasis5GL.hpp
     makutu/Qk_Hexahedron_Lagrange_GaussLobatto.hpp
     makutu/tensorops.hpp
     tensorial/LagrangeBasis1_Tensorial.hpp
     tensorial/LagrangeBasis2_Tensorial.hpp
     tensorial/LagrangeBasis3GL_Tensorial.hpp
     tensorial/LagrangeBasis4GL_Tensorial.hpp
     tensorial/LagrangeBasis5GL_Tensorial.hpp
     tensorial/Qk_Hexahedron_Tensorial.hpp
)

if( ENABLE_Shiva )
    list( APPEND finiteElement_headers
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
