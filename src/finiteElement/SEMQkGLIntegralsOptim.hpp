#ifndef SEMQKGLINTEGRALSOPTIM_HPP_
#define SEMQKGLINTEGRALSOPTIM_HPP_

//#include "dataType.hpp"
//#include "SEMmacros.hpp"
//#include "SEMdata.hpp"
#include "SEMQkGLBasisFunctions.hpp"
using namespace std;

/**
 * This class is the basis class for the hexahedron finite element cells with shape functions defined on Gauss-Lobatto quadrature points.
 */
template< int ORDER >
 class SEMQkGLIntegralsOptim
{
private:
  struct SEMinfo infos;
  SEMQkGLBasisFunctions GLBasis;

  ////////////////////////////////////////////////////////////////////////////////////
  //  from GEOS implementation
  /////////////////////////////////////////////////////////////////////////////////////
  constexpr static double sqrt5 = 2.2360679774997897;
  // order of polynomial approximation
  // number of support/quadrature/nodes points in one direction
  constexpr static int numSupport1dPoints = ORDER + 1;
  constexpr static int num1dNodes = numSupport1dPoints;
  // Half the number of support points, rounded down. Precomputed for efficiency
  constexpr static int halfNodes = ( numSupport1dPoints - 1 ) / 2;
  // the number of nodes/support points per element
  constexpr static int numSupportPoints = numSupport1dPoints * numSupport1dPoints * numSupport1dPoints;

public:
  SEMKERNELS_HOST_DEVICE SEMQkGLIntegralsOptim(){};
  SEMKERNELS_HOST_DEVICE ~SEMQkGLIntegralsOptim(){};

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
  constexpr static double interpolationCoord( const int , const int q, const int k )
  {
    const double alpha = (SEMQkGLBasisFunctions::parentSupportCoord< SEMinfo >( q ) + 1.0 ) / 2.0;
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
  constexpr static double jacobianCoefficient1D( const int order, const int q, const int i, const int k, const int dir )
  {
    if ( i == dir )
    {
      return k == 0 ? -1.0 / 2.0 : 1.0 / 2.0;
    }
    else
    {
      return interpolationCoord( order, q, k );
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
  static constexpr inline
  SEMKERNELS_HOST_DEVICE
  void jacobianTransformation( int const qa,
                               int const qb,
                               int const qc,
                               double const (&X)[8][3],
                               double ( & J )[3][3] )
  {
    for ( int k = 0; k < 8; k++ )
    {
      const int ka = k % 2;
      const int kb = ( k % 4 ) / 2;
      const int kc = k / 4;
      for ( int j = 0; j < 3; j++ )
      {
        double jacCoeff = jacobianCoefficient1D( ORDER, qa, 0, ka, j ) *
                          jacobianCoefficient1D( ORDER, qb, 1, kb, j ) *
                          jacobianCoefficient1D( ORDER, qc, 2, kc, j );
        for ( int i = 0; i < 3; i++ )
        {
          J[i][j] +=  jacCoeff * X[k][i];
        }
      }
    }
  }

  /**
   * @brief computes the non-zero contributions of the d.o.f. indexd by q to the
   *   mass matrix M, i.e., the superposition matrix of the shape functions.
   * @param q The quadrature point index
   * @param X Array containing the coordinates of the mesh support points.
   * @return The diagonal mass term associated to q
   */
  static constexpr inline
  SEMKERNELS_HOST_DEVICE
  double computeMassTerm( int const q, double const (&X)[8][3] )
  {
    TripleIndex ti = tripleIndex( 1, q );
    int qa = ti.i0;
    int qb = ti.i1;
    int qc = ti.i2;
    const double w3D = SEMQkGLBasisFunctions::weight< SEMinfo >( qa ) * SEMQkGLBasisFunctions::weight< SEMinfo >( qb ) * SEMQkGLBasisFunctions::weight< SEMinfo >( qc );
    double J[3][3] = { {0} };
    jacobianTransformation( qa, qb, qc, X, J );
    return determinant( J ) * w3D;
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
    const double w = SEMQkGLBasisFunctions::weight< SEMinfo >( qa ) *
                     SEMQkGLBasisFunctions::weight< SEMinfo >( qb ) *
                     SEMQkGLBasisFunctions::weight< SEMinfo >( qc );
    for ( int i = 0; i < num1dNodes; i++ )
    {
      const int ibc = linearIndex( ORDER, i, qb, qc );
      const int aic = linearIndex( ORDER, qa, i, qc );
      const int abi = linearIndex( ORDER, qa, qb, i );
      const double gia = SEMQkGLBasisFunctions::basisGradientAt( ORDER, i, qa );
      const double gib = SEMQkGLBasisFunctions::basisGradientAt( ORDER, i, qb );
      const double gic = SEMQkGLBasisFunctions::basisGradientAt( ORDER, i, qc );
      for ( int j = 0; j < num1dNodes; j++ )
      {
        const int jbc = linearIndex( ORDER, j, qb, qc );
        const int ajc = linearIndex( ORDER, qa, j, qc );
        const int abj = linearIndex( ORDER, qa, qb, j );
        const double gja = SEMQkGLBasisFunctions::basisGradientAt( ORDER, j, qa );
        const double gjb = SEMQkGLBasisFunctions::basisGradientAt( ORDER, j, qb );
        const double gjc = SEMQkGLBasisFunctions::basisGradientAt( ORDER, j, qc );
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


  template< typename FUNC >
  static constexpr inline
  SEMKERNELS_HOST_DEVICE
  void computeStiffnessTerm( int const q,
                             double const (&X)[8][3],
                             FUNC && func )
  {
    TripleIndex ti = tripleIndex( 1, q );
    int qa = ti.i0;
    int qb = ti.i1;
    int qc = ti.i2;
    double B[6] = {0};
    double J[3][3] = { {0} };
    jacobianTransformation( qa, qb, qc, X, J );
    computeB( J, B );
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
    double X[8][3] = { {0.0 }};
    int I = 0;
    for ( int k = 0; k < ORDER + 1; k += ORDER )
    {
      for ( int j = 0; j < ORDER + 1; j += ORDER )
      {
        for ( int i = 0; i < ORDER + 1; i += ORDER )
        {
          int l = i + j * (ORDER + 1) + k * (ORDER + 1) * (ORDER + 1);
          X[I][0] = nodesCoordsX( elementNumber, l );
          X[I][1] = nodesCoordsZ( elementNumber, l );
          X[I][2] = nodesCoordsY( elementNumber, l );
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
      massMatrixLocal[q] = computeMassTerm( q, X );
      computeStiffnessTerm( q, X, [&] ( const int i, const int j, const double val )
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
