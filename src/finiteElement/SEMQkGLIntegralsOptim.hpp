#ifndef SEMQKGLINTEGRALSOPTIM_HPP_
#define SEMQKGLINTEGRALSOPTIM_HPP_

//#include "dataType.hpp"
//#include "SEMmacros.hpp"
//#include "SEMdata.hpp"
#include "SEMQkGLBasisFunctions.hpp"
#include "common/mathUtilites.hpp"

#include<stdio.h>


/**
 * This class is the basis class for the hexahedron finite element cells with shape functions defined on Gauss-Lobatto quadrature points.
 */
template< int ORDER >
 class SEMQkGLIntegralsOptim
{
public:
  static constexpr int order = ORDER;
  constexpr static int numSupportPoints1d = ORDER + 1;


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
  constexpr static double interpolationCoord( const int q, const int k )
  {
    const double alpha = (SEMQkGLBasisFunctions<ORDER>::parentSupportCoord( q ) + 1.0 ) / 2.0;
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
  constexpr static double jacobianCoefficient1D( const int q, const int i, const int k, const int dir )
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
  template< typename COORDS_TYPE >
  static constexpr inline
  SEMKERNELS_HOST_DEVICE
  void jacobianTransformation( int const qa,
                               int const qb,
                               int const qc,
                               COORDS_TYPE const & X,
                               double ( & J )[3][3] )
  {
    for ( int k = 0; k < 8; k++ )
    {
      const int ka = k % 2;
      const int kb = ( k % 4 ) / 2;
      const int kc = k / 4;
      for ( int j = 0; j < 3; j++ )
      {
        double jacCoeff = jacobianCoefficient1D( qa, 0, ka, j ) *
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
                               double const (&B)[6],
                               FUNC && func )
  {
    //const double w = GLBasis.weight<SEMinfo>(qa )*GLBasis.weight<SEMinfo>(qb )*GLBasis.weight<SEMinfo>(qc );
    const double w = SEMQkGLBasisFunctions<ORDER>::weight( qa ) *
                     SEMQkGLBasisFunctions<ORDER>::weight( qb ) *
                     SEMQkGLBasisFunctions<ORDER>::weight( qc );
    for ( int i = 0; i < numSupportPoints1d; i++ )
    {
      const int ibc = linearIndex( ORDER, i, qb, qc );
      const int aic = linearIndex( ORDER, qa, i, qc );
      const int abi = linearIndex( ORDER, qa, qb, i );
      const double gia = SEMQkGLBasisFunctions<ORDER>::basisGradientAt( ORDER, i, qa );
      const double gib = SEMQkGLBasisFunctions<ORDER>::basisGradientAt( ORDER, i, qb );
      const double gic = SEMQkGLBasisFunctions<ORDER>::basisGradientAt( ORDER, i, qc );
//      printf("i: %d, ibc: %d, aic: %d, abi: %d, gia: %f, gib: %f, gic: %f\n", i, ibc, aic, abi, gia, gib, gic);
      for ( int j = 0; j < numSupportPoints1d; j++ )
      {
        const int jbc = linearIndex( ORDER, j, qb, qc );
        const int ajc = linearIndex( ORDER, qa, j, qc );
        const int abj = linearIndex( ORDER, qa, qb, j );
        const double gja = SEMQkGLBasisFunctions<ORDER>::basisGradientAt( ORDER, j, qa );
        const double gjb = SEMQkGLBasisFunctions<ORDER>::basisGradientAt( ORDER, j, qb );
        const double gjc = SEMQkGLBasisFunctions<ORDER>::basisGradientAt( ORDER, j, qc );

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
      }
    }
  }


  template< typename COORDS_TYPE, typename FUNC >
  static constexpr inline
  SEMKERNELS_HOST_DEVICE
  void computeStiffnessAndMassTerm( int const q,
                             COORDS_TYPE const & X,
                             float mass[],
                             FUNC && func )
  {
    TripleIndex ti = tripleIndex( ORDER, q );
    int const qa = ti.i0;
    int const qb = ti.i1;
    int const qc = ti.i2;

    double J[3][3] = { {0} };
    jacobianTransformation( qa, qb, qc, X, J );

    double detJ = determinant( J );

    double const w3D = SEMQkGLBasisFunctions<ORDER>::weight( qa ) * 
                       SEMQkGLBasisFunctions<ORDER>::weight( qb ) * 
                       SEMQkGLBasisFunctions<ORDER>::weight( qc );

    mass[q] = w3D * detJ;

    double B[6] = {0};
    computeB( J, B );

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
    shiva::CArrayNd<double,8,3> X{ 0.0 };
    int I = 0;
    for ( int k = 0; k < 2; ++k )
    {
      for ( int j = 0; j < 2; ++j )
      {
        for ( int i = 0; i < 2; ++i )
        {
          int l = i + j * 2 + k * (2) * (2);
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
      computeStiffnessAndMassTerm( q, X, massMatrixLocal, [&] ( const int i, const int j, const double val )
      {
        Y[i] = Y[i] + val * pnLocal[j];
      } );
    }
  }
  /////////////////////////////////////////////////////////////////////////////////////
  //  end from GEOS implementation
  /////////////////////////////////////////////////////////////////////////////////////

};

#endif //SEMQKGLINTEGRALSOPTIM_HPP_
