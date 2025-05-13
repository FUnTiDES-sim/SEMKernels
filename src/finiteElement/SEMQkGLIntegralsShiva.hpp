#pragma once

#include "common/macros.hpp"
#include "common/mathUtilites.hpp"

#include "functions/bases/LagrangeBasis.hpp"
#include "functions/quadrature/Quadrature.hpp"
#include "functions/spacing/Spacing.hpp"
#include "geometry/shapes/NCube.hpp"
#include "geometry/shapes/InterpolatedShape.hpp"
#include "geometry/mapping/LinearTransform.hpp"
#include "common/ShivaMacros.hpp"
#include "common/pmpl.hpp"
#include "common/types.hpp"
#include "discretizations/finiteElementMethod/parentElements/ParentElement.hpp"

#include <stdio.h>

using namespace shiva;
using namespace shiva::functions;
using namespace shiva::geometry;
using namespace shiva::geometry::utilities;
using namespace shiva::discretizations::finiteElementMethod;

using namespace std;

/**
 * This class is the basis class for the hexahedron finite element cells with shape functions defined on Gauss-Lobatto quadrature points.
 */
template< typename REAL_TYPE,
          int ORDER,
          typename TRANSFORM,
          typename PARENT_ELEMENT >
class SEMQkGLIntegralsShiva
{
public:

  static constexpr int order = ORDER;
  static constexpr int numSupportPoints1d = ORDER + 1;

  using TransformType = TRANSFORM;

  using ParentElementType = PARENT_ELEMENT;

  using JacobianType = typename std::remove_reference_t< TransformType >::JacobianType;
  using quadrature = QuadratureGaussLobatto< double, numSupportPoints1d >;
  using basisFunction = LagrangeBasis< double, ORDER, GaussLobattoSpacing >;

  void init()
  {}


  template< int qa, int qb, int qc, typename FUNC >
  static constexpr inline
  SEMKERNELS_HOST_DEVICE
  void computeGradPhiBGradPhi( double const (&B)[6],
                               FUNC && func )
  {
    constexpr double qcoords[3] = { quadrature::template coordinate< qa >(),
                                    quadrature::template coordinate< qb >(),
                                    quadrature::template coordinate< qc >() };
    constexpr double w = quadrature::template weight< qa >() * quadrature::template weight< qb >() * quadrature::template weight< qc >();
    forSequence< numSupportPoints1d >( [&] ( auto const ici )
    {
      constexpr int i = decltype(ici)::value;      
      const int ibc = linearIndex( ORDER, i, qb, qc );
      const int aic = linearIndex( ORDER, qa, i, qc );
      const int abi = linearIndex( ORDER, qa, qb, i );
      const double gia = basisFunction::template gradient< i >( qcoords[0] );
      const double gib = basisFunction::template gradient< i >( qcoords[1] );
      const double gic = basisFunction::template gradient< i >( qcoords[2] );
//      printf("i: %d, ibc: %d, aic: %d, abi: %d, gia: %f, gib: %f, gic: %f\n", i, ibc, aic, abi, gia, gib, gic);

      forSequence< numSupportPoints1d >( [&] ( auto const icj )
      {
        constexpr int j = decltype(icj)::value;
        const int jbc = linearIndex( ORDER, j, qb, qc );
        const int ajc = linearIndex( ORDER, qa, j, qc );
        const int abj = linearIndex( ORDER, qa, qb, j );
        const double gja = basisFunction::template gradient< j >( qcoords[0] );
        const double gjb = basisFunction::template gradient< j >( qcoords[1] );
        const double gjc = basisFunction::template gradient< j >( qcoords[2] );

//        printf("j: %d, jbc: %d, ajc: %d, abj: %d, gja: %f, gjb: %f, gjc: %f\n", j, jbc, ajc, abj, gja, gjb, gjc);
        // diagonal terms
        const double w0 = w * gia * gja;
        func( ibc, jbc, w0 * B[0] );
        const double w1 = w * gib * gjb;
        func( aic, ajc, w1 * B[1] );
        const double w2 = w * gic * gjc;
        func( abi, abj, w2 * B[2] );
        // off-diagonal terms
        const double w3 = w * gib * gjc;
        func( aic, abj, w3 * B[3] );
        func( abj, aic, w3 * B[3] );
        const double w4 = w * gia * gjc;
        func( ibc, abj, w4 * B[4] );
        func( abj, ibc, w4 * B[4] );
        const double w5 = w * gia * gjb;
        func( ibc, ajc, w5 * B[5] );
        func( ajc, ibc, w5 * B[5] );
      } );
    } );
  }



  template< typename FUNC >
  static constexpr inline
  SEMKERNELS_HOST_DEVICE
  void computeStiffnessAndMassTerm( TransformType const & trilinearCell,
                             float massMatrix[],
                             FUNC && func )
  {
    JacobianType J{ 0.0 };

    // this is a compile time quadrature loop over each tensor direction
    forNestedSequence< ORDER + 1,
                       ORDER + 1,
                       ORDER + 1 >( [&] ( auto const icqa,
                                          auto const icqb,
                                          auto const icqc )
    {
      constexpr int qa = decltype(icqa)::value;
      constexpr int qb = decltype(icqb)::value;
      constexpr int qc = decltype(icqc)::value;
      // must be here, Jacobian must be put to 0 for each quadrature point
      //Jacobian matrix J
      for ( int i = 0; i < 3; ++i )
      {
        for ( int j = 0; j < 3; ++j )
        {
          J( i, j ) = 0;
        }
      }

      shiva::geometry::utilities::jacobian< quadrature, qa, qb, qc >( trilinearCell, J );

      double const detJ = determinant( J );
      
      // mass matrix
      constexpr int q = linearIndex( ORDER, qa, qb, qc );
      constexpr double w3D = quadrature::template weight< qa >() *
                             quadrature::template weight< qb >() *
                             quadrature::template weight< qc >();
      massMatrix[q] = w3D * detJ;

      double B[6] = {0};
      computeB( J, B );

//      printf( "B(%d,%d,%d) = | %18.14e %18.14e %18.14e %18.14e %18.14e %18.14e |\n", qa, qb, qc, B[0], B[1], B[2], B[3], B[4], B[5] );

      // compute detJ*J^{-1}J^{-T}
      for( int i = 0; i < 6; ++i )
      {
//        B[i] *= detJ;
      }

      // compute gradPhiI*B*gradPhiJ and stiffness vector
      computeGradPhiBGradPhi< qa, qb, qc >( B, func );
    } );
  }


  template< typename ARRAY_REAL_VIEW, typename LOCAL_ARRAY >
  static constexpr inline
  SEMKERNELS_HOST_DEVICE
  void
  gatherCoordinates( const int & elementNumber,
                     ARRAY_REAL_VIEW const & nodesCoordsX,
                     ARRAY_REAL_VIEW const & nodesCoordsY,
                     ARRAY_REAL_VIEW const & nodesCoordsZ,
                     LOCAL_ARRAY & cellData )
  {
    for ( int k = 0; k < 2; ++k )
    {
      for ( int j = 0; j < 2; ++j )
      {
        for ( int i = 0; i < 2; ++i )
        {
          int const l = linearIndex( 1, i, j, k );
          cellData( i, j, k, 0 ) = nodesCoordsX( elementNumber, l );
          cellData( i, j, k, 1 ) = nodesCoordsY( elementNumber, l );
          cellData( i, j, k, 2 ) = nodesCoordsZ( elementNumber, l );
        }
      }
    }
  }

  /**
   * @brief compute  mass Matrix stiffnessVector.
   */
  template< typename ARRAY_REAL_VIEW >
  static constexpr inline
  SEMKERNELS_HOST_DEVICE
  void computeMassMatrixAndStiffnessVector( const int & elementNumber,
                                            const int & nPointsPerElement,
                                            ARRAY_REAL_VIEW const & nodesCoordsX,
                                            ARRAY_REAL_VIEW const & nodesCoordsY,
                                            ARRAY_REAL_VIEW const & nodesCoordsZ,
                                            float massMatrixLocal[],
                                            float pnLocal[],
                                            float Y[] )
  {
    TransformType trilinearCell;
    typename TransformType::DataType & cellCoordData = trilinearCell.getData();

    gatherCoordinates( elementNumber,
                       nodesCoordsX,
                       nodesCoordsY,
                       nodesCoordsZ,
                       cellCoordData );

    for ( int q = 0; q < nPointsPerElement; q++ )
    {
      Y[q] = 0;
    }
    computeStiffnessAndMassTerm( trilinearCell, massMatrixLocal, [&] ( const int i, const int j, const double val )
    {
      Y[i] = Y[i] + val * pnLocal[j];
    } );
  }

};
