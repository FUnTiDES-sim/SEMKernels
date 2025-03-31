
#include "macros.hpp"


/**
 * @brief Calculates the linear index for support/quadrature points from ijk
 *   coordinates.
 * @param r order of polynomial approximation
 * @param i The index in the xi0 direction (0,r)
 * @param j The index in the xi1 direction (0,r)
 * @param k The index in the xi2 direction (0,r)
 * @return The linear index of the support/quadrature point (0-(r+1)^3)
 */
static constexpr inline
SEMKERNELS_HOST_DEVICE
int linearIndex( const int r,
                 const int i,
                 const int j,
                 const int k )
{
  return i + (r + 1) * j + (r + 1) * (r + 1) * k;
}

/**
 * @brief Invert the symmetric matrix @p srcSymMatrix and store the result in @p dstSymMatrix.
 * @param dstSymMatrix The 3x3 symmetric matrix to write the inverse to.
 * @param srcSymMatrix The 3x3 symmetric matrix to take the inverse of.
 * @return The determinant.
 * @note @p srcSymMatrix can contain integers but @p dstMatrix must contain floating point values.
 */
template< typename T >
static constexpr inline
SEMKERNELS_HOST_DEVICE
void symInvert( T (&dstSymMatrix)[6], T const (&srcSymMatrix)[6] )
{

  dstSymMatrix[ 0 ] = srcSymMatrix[ 1 ] * srcSymMatrix[ 2 ] - srcSymMatrix[ 3 ] * srcSymMatrix[ 3 ];
  dstSymMatrix[ 5 ] = srcSymMatrix[ 4 ] * srcSymMatrix[ 3 ] - srcSymMatrix[ 5 ] * srcSymMatrix[ 2 ];
  dstSymMatrix[ 4 ] = srcSymMatrix[ 5 ] * srcSymMatrix[ 3 ] - srcSymMatrix[ 4 ] * srcSymMatrix[ 1 ];

  T det = srcSymMatrix[ 0 ] * dstSymMatrix[ 0 ] + srcSymMatrix[ 5 ] * dstSymMatrix[ 5 ] + srcSymMatrix[ 4 ] * dstSymMatrix[ 4 ];

  T const invDet = 1.0 / det;

  dstSymMatrix[ 0 ] *= invDet;
  dstSymMatrix[ 5 ] *= invDet;
  dstSymMatrix[ 4 ] *= invDet;
  dstSymMatrix[ 1 ] = ( srcSymMatrix[ 0 ] * srcSymMatrix[ 2 ] - srcSymMatrix[ 4 ] * srcSymMatrix[ 4 ] ) * invDet;
  dstSymMatrix[ 3 ] = ( srcSymMatrix[ 5 ] * srcSymMatrix[ 4 ] - srcSymMatrix[ 0 ] * srcSymMatrix[ 3 ] ) * invDet;
  dstSymMatrix[ 2 ] = ( srcSymMatrix[ 0 ] * srcSymMatrix[ 1 ] - srcSymMatrix[ 5 ] * srcSymMatrix[ 5 ] ) * invDet;
}


/**
 * @brief Invert the symmetric matrix @p symMatrix overwritting it.
 * @param symMatrix The 3x3 symmetric matrix to take the inverse of and overwrite.
 * @return The determinant.
 * @note @p symMatrix can contain integers but @p dstMatrix must contain floating point values.
 */
template< typename T >
static inline
SEMKERNELS_HOST_DEVICE
void symInvert0( T (&symMatrix)[6] )
{
  T temp[ 6 ];
  symInvert( temp, symMatrix );

  symMatrix[0] = temp[0];
  symMatrix[1] = temp[1];
  symMatrix[2] = temp[2];
  symMatrix[3] = temp[3];
  symMatrix[4] = temp[4];
  symMatrix[5] = temp[5];
}


template< typename T >
static constexpr inline
SEMKERNELS_HOST_DEVICE
void computeB( T const (&J)[3][3],
               T (&B)[6] )
{

  // detJ
  double const detJ = +J[0][0] * ( J[1][1] * J[2][2] - J[2][1] * J[1][2] )
                      - J[0][1] * ( J[1][0] * J[2][2] - J[2][0] * J[1][2] )
                      + J[0][2] * ( J[1][0] * J[2][1] - J[2][0] * J[1][1] );

  // compute J^{T}J/detJ
  double const invDetJ = 1.0 / detJ;
  B[0] = ( J[0][0] * J[0][0] + J[1][0] * J[1][0] + J[2][0] * J[2][0] ) * invDetJ;
  B[1] = ( J[0][1] * J[0][1] + J[1][1] * J[1][1] + J[2][1] * J[2][1] ) * invDetJ;
  B[2] = ( J[0][2] * J[0][2] + J[1][2] * J[1][2] + J[2][2] * J[2][2] ) * invDetJ;
  B[3] = ( J[0][1] * J[0][2] + J[1][1] * J[1][2] + J[2][1] * J[2][2] ) * invDetJ;
  B[4] = ( J[0][0] * J[0][2] + J[1][0] * J[1][2] + J[2][0] * J[2][2] ) * invDetJ;
  B[5] = ( J[0][0] * J[0][1] + J[1][0] * J[1][1] + J[2][0] * J[2][1] ) * invDetJ;
  // compute detJ*J^{-1}J^{-T}
  symInvert0( B );
}

template< typename T, typename JTYPE >
static constexpr inline
SEMKERNELS_HOST_DEVICE
void computeB( JTYPE const & J,
               T (&B)[6] )
{

  // detJ
  double const detJ = +J( 0, 0 ) * (J( 1, 1 ) * J( 2, 2 ) - J( 2, 1 ) * J( 1, 2 ))
                      - J( 0, 1 ) * (J( 1, 0 ) * J( 2, 2 ) - J( 2, 0 ) * J( 1, 2 ))
                      + J( 0, 2 ) * (J( 1, 0 ) * J( 2, 1 ) - J( 2, 0 ) * J( 1, 1 ));

  // compute J^{T}J/detJ
  double const invDetJ = 1.0 / detJ;
  B[0] = ( J( 0, 0 ) * J( 0, 0 ) + J( 1, 0 ) * J( 1, 0 ) + J( 2, 0 ) * J( 2, 0 ) ) * invDetJ;
  B[1] = ( J( 0, 1 ) * J( 0, 1 ) + J( 1, 1 ) * J( 1, 1 ) + J( 2, 1 ) * J( 2, 1 ) ) * invDetJ;
  B[2] = ( J( 0, 2 ) * J( 0, 2 ) + J( 1, 2 ) * J( 1, 2 ) + J( 2, 2 ) * J( 2, 2 ) ) * invDetJ;
  B[3] = ( J( 0, 1 ) * J( 0, 2 ) + J( 1, 1 ) * J( 1, 2 ) + J( 2, 1 ) * J( 2, 2 ) ) * invDetJ;
  B[4] = ( J( 0, 0 ) * J( 0, 2 ) + J( 1, 0 ) * J( 1, 2 ) + J( 2, 0 ) * J( 2, 2 ) ) * invDetJ;
  B[5] = ( J( 0, 0 ) * J( 0, 1 ) + J( 1, 0 ) * J( 1, 1 ) + J( 2, 0 ) * J( 2, 1 ) ) * invDetJ;
  // compute detJ*J^{-1}J^{-T}
  symInvert0( B );
}
