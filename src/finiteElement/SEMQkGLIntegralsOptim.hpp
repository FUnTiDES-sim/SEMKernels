#ifndef SEMQKGLINTEGRALSOPTIM_HPP_
#define SEMQKGLINTEGRALSOPTIM_HPP_

#include "SEMQkGLBasisFunctions.hpp"
#include "common/mathUtilites.hpp"
#include "common/CArray.hpp"


#include<stdio.h>


/**
 * This class is the basis class for the hexahedron finite element cells with shape functions defined on Gauss-Lobatto quadrature points.
 */
template< int ORDER,
          typename TRANSFORM_FLOAT,
          typename GRADIENT_FLOAT >
 class SEMQkGLIntegralsOptim
{
public:
  static constexpr int order = ORDER;
  constexpr static int numSupportPoints1d = ORDER + 1;


  void init()
  {}
  /////////////////////////////////////////////////////////////////////////////////////
  //  from GEOS implementation
  /////////////////////////////////////////////////////////////////////////////////////





  /**
   * @brief Compute the interpolation coefficients of the q-th quadrature point in a given direction
   * @param q the index of the quadrature point in 1D
   * @param k the index of the interval endpoint (0 or 1)
   * @return The interpolation coefficient
   */
  SEMKERNELS_HOST_DEVICE
  constexpr static TRANSFORM_FLOAT interpolationCoord( const int q, const int k )
  {
    const TRANSFORM_FLOAT alpha = (SEMQkGLBasisFunctions<ORDER, TRANSFORM_FLOAT>::parentSupportCoord( q ) + 1.0 ) / 2.0;
    return k == 0 ? ( 1.0 - alpha ) : alpha;
  }

  /**
   * @brief Compute the 1D factor of the coefficient of the jacobian on the q-th quadrature point,
   * with respect to the k-th interval endpoint (0 or 1). The computation depends on the position
   * in the basis tensor product of this term (i, equal to 0, 1 or 2) and on the direction in which
   * the gradient is being computed (dir, from 0 to 2)
   * @param q The index of the quadrature point in 1D
   * @param i The index of the position in the tensor product
   * @param k The index of the interval endpoint (0 or 1)
   * @param dir The direction in which the derivatives are being computed
   * @return The value of the jacobian factor
   */
  SEMKERNELS_HOST_DEVICE
  constexpr static TRANSFORM_FLOAT jacobianCoefficient1D( const int q, const int i, const int k, const int dir )
  {
    if ( i == dir )
    {
      return k == 0 ? -0.5 : 0.5;
    }
    else
    {
      return interpolationCoord( q, k );
    }
  }



  /**
   * @brief Calculates the isoparametric "Jacobian" transformation
   *  matrix/mapping from the parent space to the physical space.
   * @param qa The 1d quadrature point index in xi0 direction (0,1)
   * @param qb The 1d quadrature point index in xi1 direction (0,1)
   * @param qc The 1d quadrature point index in xi2 direction (0,1)
   * @param X Array containing the coordinates of the mesh support points.
   * @param J Array to store the Jacobian transformation.
   */
  template< typename COORDS_TYPE,
            typename JACOBIAN_TYPE >
  static constexpr inline
  SEMKERNELS_HOST_DEVICE
  void jacobianTransformation( int const qa,
                               int const qb,
                               int const qc,
                               COORDS_TYPE const & X,
                               JACOBIAN_TYPE ( & J )[3][3] )
  {
    for ( int k = 0; k < 8; k++ )
    {
      // const int ka = k % 2;
      // const int kb = ( k % 4 ) / 2;
      // const int kc = k / 4;
      auto const [ ka, kb, kc ] = tripleIndex<1>( k );
      for ( int j = 0; j < 3; j++ )
      {
        JACOBIAN_TYPE jacCoeff = jacobianCoefficient1D( qa, 0, ka, j ) *
                          jacobianCoefficient1D( qb, 1, kb, j ) *
                          jacobianCoefficient1D( qc, 2, kc, j );
        for ( int i = 0; i < 3; i++ )
        {
          J[i][j] +=  jacCoeff * X[k][i];
        }
      }
    }
  }

  template< typename FUNC >
  static constexpr inline
  SEMKERNELS_HOST_DEVICE
  void computeGradPhiBGradPhi( int const qa,
                               int const qb,
                               int const qc,
                               TRANSFORM_FLOAT const (&B)[6],
                               FUNC && func )
  {
    const TRANSFORM_FLOAT w = SEMQkGLBasisFunctions<ORDER,GRADIENT_FLOAT>::weight( qa ) *
                              SEMQkGLBasisFunctions<ORDER,GRADIENT_FLOAT>::weight( qb ) *
                              SEMQkGLBasisFunctions<ORDER,GRADIENT_FLOAT>::weight( qc );
    for ( int i = 0; i < numSupportPoints1d; i++ )
    {
      int const ibc = linearIndex<ORDER>( i, qb, qc );
      int const aic = linearIndex<ORDER>( qa, i, qc );
      int const abi = linearIndex<ORDER>( qa, qb, i );
      GRADIENT_FLOAT const gia = SEMQkGLBasisFunctions<ORDER,GRADIENT_FLOAT>::basisGradientAt( ORDER, i, qa );
      GRADIENT_FLOAT const gib = SEMQkGLBasisFunctions<ORDER,GRADIENT_FLOAT>::basisGradientAt( ORDER, i, qb );
      GRADIENT_FLOAT const gic = SEMQkGLBasisFunctions<ORDER,GRADIENT_FLOAT>::basisGradientAt( ORDER, i, qc );
//      printf("i: %d, ibc: %d, aic: %d, abi: %d, gia: %f, gib: %f, gic: %f\n", i, ibc, aic, abi, gia, gib, gic);
      for ( int j = 0; j < numSupportPoints1d; j++ )
      {
        int const jbc = linearIndex<ORDER>( j, qb, qc );
        int const ajc = linearIndex<ORDER>( qa, j, qc );
        int const abj = linearIndex<ORDER>( qa, qb, j );
        GRADIENT_FLOAT const gja = SEMQkGLBasisFunctions<ORDER,GRADIENT_FLOAT>::basisGradientAt( ORDER, j, qa );
        GRADIENT_FLOAT const gjb = SEMQkGLBasisFunctions<ORDER,GRADIENT_FLOAT>::basisGradientAt( ORDER, j, qb );
        GRADIENT_FLOAT const gjc = SEMQkGLBasisFunctions<ORDER,GRADIENT_FLOAT>::basisGradientAt( ORDER, j, qc );

//        printf("j: %d, jbc: %d, ajc: %d, abj: %d, gja: %f, gjb: %f, gjc: %f\n", j, jbc, ajc, abj, gja, gjb, gjc);

        // diagonal terms
        GRADIENT_FLOAT const w0 = w * gia * gja;
        func( ibc, jbc, w0 * B[0] );
        GRADIENT_FLOAT const w1 = w * gib * gjb;
        func( aic, ajc, w1 * B[1] );
        GRADIENT_FLOAT const w2 = w * gic * gjc;
        func( abi, abj, w2 * B[2] );
        // off-diagonal terms
        // GRADIENT_FLOAT const w3 = w * gib * gjc;
        // func( aic, abj, w3 * B[3] );
        // func( abj, aic, w3 * B[3] );
        // GRADIENT_FLOAT const w4 = w * gia * gjc;
        // func( ibc, abj, w4 * B[4] );
        // func( abj, ibc, w4 * B[4] );
        // GRADIENT_FLOAT const w5 = w * gia * gjb;
        // func( ibc, ajc, w5 * B[5] );
        // func( ajc, ibc, w5 * B[5] );
      }
    }
  }


  template< typename COORDS_TYPE, 
            typename FUNC >
  static constexpr inline
  SEMKERNELS_HOST_DEVICE
  void computeStiffnessAndMassTerm( int const q,
                             COORDS_TYPE const & X,
                             float mass[],
                             FUNC && func )
  {
    auto const [ qa, qb, qc ] = tripleIndex<ORDER>( q );

    TRANSFORM_FLOAT J[3][3] = { {0} };
    jacobianTransformation( qa, qb, qc, X, J );

    TRANSFORM_FLOAT detJ = determinant( J );

    TRANSFORM_FLOAT const w3D = SEMQkGLBasisFunctions<ORDER, TRANSFORM_FLOAT>::weight( qa ) * 
                                SEMQkGLBasisFunctions<ORDER, TRANSFORM_FLOAT>::weight( qb ) * 
                                SEMQkGLBasisFunctions<ORDER, TRANSFORM_FLOAT>::weight( qc );

    mass[q] = w3D * detJ;

    TRANSFORM_FLOAT B[6] = {0};
    computeB( J, B );

    for( int i = 0; i < 6; ++i )
    {
      B[i] *= detJ;
    }


//    printf( "B(%d,%d,%d): %18.14e %18.14e %18.14e %18.14e %18.14e %18.14e\n", qa, qb, qc, B[0], B[1], B[2], B[3], B[4], B[5] );

    computeGradPhiBGradPhi( qa, qb, qc, B, func );
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
    TRANSFORM_FLOAT X[8][3]{ {0} };
    int I = 0;
    for ( int k = 0; k < 2; ++k )
    {
      for ( int j = 0; j < 2; ++j )
      {
        for ( int i = 0; i < 2; ++i )
        {
          int const l = linearIndex<1>( i, j, k );
          X[I][0] = nodesCoordsX( elementNumber, l );
          X[I][1] = nodesCoordsY( elementNumber, l );
          X[I][2] = nodesCoordsZ( elementNumber, l );
          I++;
        }
      }
    }
    for ( int q = 0; q < nPointsPerElement; q++ )
    {
      Y[q] = 0;
    }
    for ( int q = 0; q < nPointsPerElement; q++ )
    {
      computeStiffnessAndMassTerm( q, X, massMatrixLocal, [&] ( const int i, const int j, const GRADIENT_FLOAT val )
      {
        GRADIENT_FLOAT localIncrement = val * pnLocal[j];
        Y[i] = Y[i] + val * pnLocal[j];
      } );
    }
  }
  /////////////////////////////////////////////////////////////////////////////////////
  //  end from GEOS implementation
  /////////////////////////////////////////////////////////////////////////////////////

};

#endif //SEMQKGLINTEGRALSOPTIM_HPP_
