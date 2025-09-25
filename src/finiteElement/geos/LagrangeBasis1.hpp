/*
 * ------------------------------------------------------------------------------------------------------------
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * Copyright (c) 2016-2024 Lawrence Livermore National Security LLC
 * Copyright (c) 2018-2024 TotalEnergies
 * Copyright (c) 2018-2024 The Board of Trustees of the Leland Stanford Junior University
 * Copyright (c) 2023-2024 Chevron
 * Copyright (c) 2019-     GEOS/GEOSX Contributors
 * All rights reserved
 *
 * See top level LICENSE, COPYRIGHT, CONTRIBUTORS, NOTICE, and ACKNOWLEDGEMENTS files for details.
 * ------------------------------------------------------------------------------------------------------------
 */

#ifndef _LAGRANGEBASIS1_HPP_
#define _LAGRANGEBASIS1_HPP_

/**
 * @file LagrangeBasis1.hpp
 */

// #include "common/DataTypes.hpp"


/**
 * This class contains the implementation for a first order (linear) Lagrange
 * polynomial basis. The parent space is defined by:
 *
 *                 o-------------o  ---> xi
 *  Index:         0             1
 *  Coordinate:   -1             1
 *
 */
class LagrangeBasis1
{
public:
  /// The number of support points for the basis
  constexpr static int numSupportPoints = 2;

  /**
   * @brief The value of the weight for the given support point
   * @param q The index of the support point
   * @return The value of the weight
   */
  constexpr static double weight( const int q )
  {
    // GEOS_UNUSED_VAR( q );
    return 1.0;
  }

  /**
   * @brief Calculate the parent coordinates for the xi0 direction, given the
   *   linear index of a support point.
   * @param supportPointIndex The linear index of support point
   * @return parent coordinate in the xi0 direction.
   */
  constexpr static double parentSupportCoord( const int supportPointIndex )
  {
    return -1.0 + 2.0 * (supportPointIndex & 1);
  }

  /**
   * @brief The value of the basis function for a support point evaluated at a
   *   point along the axes.
   * @param index The index of the support point.
   * @param xi The coordinate at which to evaluate the basis.
   * @return The value of basis function.
   */
  constexpr static double value( const int index,
                                 const double xi )
  {
    return 0.5 + 0.5 * xi * parentSupportCoord( index );
  }


  /**
   * @brief The value of the basis function for the 0 support point.
   * @param xi The coordinate at which to evaluate the basis.
   * @return The value of the basis.
   */
  constexpr static double value0( const double xi )
  {
    return 0.5 - 0.5 * xi;
  }

  /**
   * @brief The value of the basis function for the 1 support point.
   * @param xi The coordinate at which to evaluate the basis.
   * @return The value of the basis.
   */
  constexpr static double value1( const double xi )
  {
    return 0.5 + 0.5 * xi;
  }

  /**
   * @brief The value of the bubble basis function.
   * @param xi The coordinate at which to evaluate the basis.
   * @return The value of the basis.
   */
  constexpr static double valueBubble( const double xi )
  {
    return 1.0 - pow( xi, 2 );
    // return 1.0 - std::pow(2.0, xi);
  }


  /**
   * @brief The gradient of the basis function for a support point evaluated at
   *   a point along the axes.
   * @param index The index of the support point associated with the basis
   *   function.
   * @param xi The coordinate at which to evaluate the gradient.
   * @return The gradient of basis function.
   */
  constexpr static double gradient( const int index,
                                    const double xi )
  {
    // GEOS_UNUSED_VAR( xi );
    return 0.5 * parentSupportCoord( index );
  }

  /**
   * @brief The gradient of the basis function for support point 0 evaluated at
   *   a point along the axes.
   * @param xi The coordinate at which to evaluate the gradient.
   * @return The gradient of basis function (-0.5)
   */
  constexpr static double gradient0( const double xi )
  {
    // GEOS_UNUSED_VAR( xi );
    return -0.5;
  }

  /**
   * @brief The gradient of the basis function for support point 1 evaluated at
   *   a point along the axes.
   * @param xi The coordinate at which to evaluate the gradient.
   * @return The gradient of basis function (0.5)
   */
  constexpr static double gradient1( const double xi )
  {
    // GEOS_UNUSED_VAR( xi );
    return 0.5;
  }

  /**
   * @brief The gradient of the bubble basis function for support point 1 evaluated at
   *   a point along the axes.
   * @param xi The coordinate at which to evaluate the gradient.
   * @return The gradient of basis function
   */
  constexpr static double gradientBubble( const double xi )
  {
    return -2.0*xi;
  }

  /**
   * @brief The gradient of the basis function for a support point evaluated at
   *   a given support point. By symmetry, p is assumed to be in 0, ..., (N-1)/2.
   *   in the case of the first-order basis, this value is independent of p.
   * @param q The index of the basis function
   * @return The gradient of basis function.
   */
  constexpr static double gradientAt( const int q,
                                      const int )
  {
    return q == 0 ? -0.5 : 0.5;
  }

  /**
   * @struct TensorProduct2D
   *
   * A 2-dimensional basis formed from the tensor product of the 1d basis.
   *
   *               2                   3
   *                o-----------------o                           _______________
   *                |                 |                          |Node   xi0  xi1|
   *                |                 |                          |=====  ===  ===|
   *                |                 |                          | 0     -1   -1 |
   *                |                 |                          | 1      1   -1 |
   *                |                 |            xi1           | 2     -1    1 |
   *                |                 |            |             | 3      1    1 |
   *                |                 |            |             |_______________|
   *                o-----------------o            |
   *               0                   1           ------ xi0
   *
   */
  struct TensorProduct2D
  {
    /// The number of support points in the basis.
    constexpr static int numSupportPoints = 4;

    /**
     * @brief Calculates the linear index for support/quadrature points from ijk
     *   coordinates.
     * @param i The index in the xi0 direction (0,1)
     * @param j The index in the xi1 direction (0,1)
     * @return The linear index of the support/quadrature point (0-3)
     */
    constexpr static int linearIndex( const int i,
                                      const int j )
    {
      return i + 2 * j;
    }

    /**
     * @brief Calculate the Cartesian/TensorProduct index given the linear index
     *   of a support point.
     * @param linearIndex The linear index of support point
     * @param i0 The Cartesian index of the support point in the xi0 direction.
     * @param i1 The Cartesian index of the support point in the xi1 direction.
     */
    constexpr static void multiIndex( const int linearIndex,
                                      int & i0,
                                      int & i1 )
    {
      i0 = ( linearIndex & 1 );
      i1 = ( linearIndex & 2 ) >> 1;
    }

    /**
     * @brief The value of the basis function for a support point evaluated at a
     *   point along the axes.
     *
     * @param coords The coordinates (in the parent frame) at which to evaluate the basis
     * @param N Array to hold the value of the basis functions at each support point.
     */
    static void value( double const (&coords)[2],
                       double (& N)[numSupportPoints] )
    {
      for( int a=0; a<2; ++a )
      {
        for( int b=0; b<2; ++b )
        {
          const int lindex = LagrangeBasis1::TensorProduct2D::linearIndex( a, b );
          N[ lindex ] = LagrangeBasis1::value( a, coords[0] ) *
                        LagrangeBasis1::value( b, coords[1] );
        }
      }
    }

    /**
     * @brief The value of the bubble basis function evaluated at a
     *   point along the axes.
     *
     * @param coords The coordinates (in the parent frame) at which to evaluate the basis
     * @param N Array to hold the value of the basis functions.
     */
    static void valueBubble( double const (&coords)[2],
                             double (& N)[1] )
    {
      N[0] = LagrangeBasis1::valueBubble( coords[0] ) *
             LagrangeBasis1::valueBubble( coords[1] );
    }

    /**
     * @brief The parent coordinates for a support point in the xi0 direction.
     * @param linearIndex The linear index of the support point
     * @return
     */
    constexpr static double parentCoords0( int const linearIndex )
    {
      return -1.0 + 2.0 * (linearIndex & 1);
    }

    /**
     * @brief The parent coordinates for a support point in the xi1 direction.
     * @param linearIndex The linear index of the support point
     * @return
     */
    constexpr static double parentCoords1( int const linearIndex )
    {
      return -1.0 + ( linearIndex & 2 );
    }

  };

  /**
   * @struct TensorProduct3D
   *
   * A 3-dimensional basis formed from the tensor product of the 1d basis.
   *
   *                  6                   7                       ____________________
   *                   o-----------------o                       |Node   xi0  xi1  xi2|
   *                  /.                /|                       |=====  ===  ===  ===|
   *                 / .               / |                       | 0     -1   -1   -1 |
   *              4 o-----------------o 5|                       | 1      1   -1   -1 |
   *                |  .              |  |                       | 2     -1    1   -1 |
   *                |  .              |  |                       | 3      1    1   -1 |
   *                |  .              |  |                       | 4     -1   -1    1 |
   *                |  .              |  |                       | 5      1   -1    1 |
   *                |2 o..............|..o 3       xi2           | 6     -1    1    1 |
   *                | ,               | /          |             | 7      1    1    1 |
   *                |,                |/           | / xi1       |____________________|
   *                o-----------------o            |/
   *               0                   1           ------ xi0
   *
   */
  struct TensorProduct3D
  {
    /// The number of support points in the basis.
    constexpr static int numSupportPoints = 8;

    /// The number of support faces in the basis.
    constexpr static int numSupportFaces = 6;

    /**
     * @brief Calculates the linear index for support/quadrature points from ijk
     *   coordinates.
     * @param i The index in the xi0 direction (0,1)
     * @param j The index in the xi1 direction (0,1)
     * @param k The index in the xi2 direction (0,1)
     * @return The linear index of the support/quadrature point (0-7)
     */
    constexpr static int linearIndex( const int i,
                                      const int j,
                                      const int k )
    {
      return i + 2 * j + 4 * k;
    }

    /**
     * @brief Calculate the Cartesian/TensorProduct index given the linear index
     *   of a support point.
     * @param linearIndex The linear index of support point
     * @param i0 The Cartesian index of the support point in the xi0 direction.
     * @param i1 The Cartesian index of the support point in the xi1 direction.
     * @param i2 The Cartesian index of the support point in the xi2 direction.
     */
    constexpr static void multiIndex( const int linearIndex,
                                      int & i0,
                                      int & i1,
                                      int & i2 )
    {
      i0 = ( linearIndex & 1 );
      i1 = ( linearIndex & 2 ) >> 1;
      i2 = ( linearIndex & 4 ) >> 2;
    }

    /**
     * @brief The value of the basis function for a support point evaluated at a
     *   point along the axes.
     *
     * @param coords The coordinates (in the parent frame) at which to evaluate the basis
     * @param N Array to hold the value of the basis functions at each support point.
     */
    PROXY_HOST_DEVICE
    static void value( double const (&coords)[3],
                       double (& N)[numSupportPoints] )
    {
      for( int a=0; a<2; ++a )
      {
        for( int b=0; b<2; ++b )
        {
          for( int c=0; c<2; ++c )
          {
            const int lindex = LagrangeBasis1::TensorProduct3D::linearIndex( a, b, c );
            N[ lindex ] = LagrangeBasis1::value( a, coords[0] ) *
                          LagrangeBasis1::value( b, coords[1] ) *
                          LagrangeBasis1::value( c, coords[2] );
          }
        }
      }
    }

    /**
     * @brief The value of the bubble basis function for a support face evaluated at a
     *   point along the axes.
     *
     * @param coords The coordinates (in the parent frame) at which to evaluate the basis
     * @param N Array to hold the value of the basis functions at each support face.
     */
    static void valueFaceBubble( double const (&coords)[3],
                                 double (& N)[numSupportFaces] )
    {
      N[ 0 ] = LagrangeBasis1::valueBubble( coords[0] ) *
               LagrangeBasis1::value( 0, coords[1] ) *
               LagrangeBasis1::valueBubble( coords[2] );

      N[ 1 ] = LagrangeBasis1::valueBubble( coords[0] ) *
               LagrangeBasis1::valueBubble( coords[1] ) *
               LagrangeBasis1::value( 0, coords[2] );

      N[ 2 ] = LagrangeBasis1::value( 0, coords[0] ) *
               LagrangeBasis1::valueBubble( coords[1] ) *
               LagrangeBasis1::valueBubble( coords[2] );

      N[ 3 ] = LagrangeBasis1::value( 1, coords[0] ) *
               LagrangeBasis1::valueBubble( coords[1] ) *
               LagrangeBasis1::valueBubble( coords[2] );

      N[ 4 ] = LagrangeBasis1::valueBubble( coords[0] ) *
               LagrangeBasis1::value( 1, coords[1] ) *
               LagrangeBasis1::valueBubble( coords[2] );

      N[ 5 ] = LagrangeBasis1::valueBubble( coords[0] ) *
               LagrangeBasis1::valueBubble( coords[1] ) *
               LagrangeBasis1::value( 1, coords[2] );
    }

    /**
     * @brief The value of the bubble basis function derivatives for a support face evaluated at a
     *   point along the axes.
     *
     * @param coords The coordinates (in the parent frame) at which to evaluate the basis
     * @param dNdXi Array to hold the value of the basis function derivatives at each support face.
     */
    static void gradientFaceBubble( double const (&coords)[3],
                                    double (& dNdXi)[numSupportFaces][3] )
    {
      dNdXi[0][0] = LagrangeBasis1::gradientBubble( coords[0] ) *
                    LagrangeBasis1::value( 0, coords[1] ) *
                    LagrangeBasis1::valueBubble( coords[2] );
      dNdXi[0][1] = LagrangeBasis1::valueBubble( coords[0] ) *
                    LagrangeBasis1::gradient( 0, coords[1] ) *
                    LagrangeBasis1::valueBubble( coords[2] );
      dNdXi[0][2] = LagrangeBasis1::valueBubble( coords[0] ) *
                    LagrangeBasis1::value( 0, coords[1] ) *
                    LagrangeBasis1::gradientBubble( coords[2] );

      dNdXi[1][0] = LagrangeBasis1::gradientBubble( coords[0] ) *
                    LagrangeBasis1::valueBubble( coords[1] ) *
                    LagrangeBasis1::value( 0, coords[2] );
      dNdXi[1][1] = LagrangeBasis1::valueBubble( coords[0] ) *
                    LagrangeBasis1::gradientBubble( coords[1] ) *
                    LagrangeBasis1::value( 0, coords[2] );
      dNdXi[1][2] = LagrangeBasis1::valueBubble( coords[0] ) *
                    LagrangeBasis1::valueBubble( coords[1] ) *
                    LagrangeBasis1::gradient( 0, coords[2] );

      dNdXi[2][0] = LagrangeBasis1::gradient( 0, coords[0] ) *
                    LagrangeBasis1::valueBubble( coords[1] ) *
                    LagrangeBasis1::valueBubble( coords[2] );
      dNdXi[2][1] = LagrangeBasis1::value( 0, coords[0] ) *
                    LagrangeBasis1::gradientBubble( coords[1] ) *
                    LagrangeBasis1::valueBubble( coords[2] );
      dNdXi[2][2] = LagrangeBasis1::value( 0, coords[0] ) *
                    LagrangeBasis1::valueBubble( coords[1] ) *
                    LagrangeBasis1::gradientBubble( coords[2] );

      dNdXi[3][0] = LagrangeBasis1::gradient( 1, coords[0] ) *
                    LagrangeBasis1::valueBubble( coords[1] ) *
                    LagrangeBasis1::valueBubble( coords[2] );
      dNdXi[3][1] = LagrangeBasis1::value( 1, coords[0] ) *
                    LagrangeBasis1::gradientBubble( coords[1] ) *
                    LagrangeBasis1::valueBubble( coords[2] );
      dNdXi[3][2] = LagrangeBasis1::value( 1, coords[0] ) *
                    LagrangeBasis1::valueBubble( coords[1] ) *
                    LagrangeBasis1::gradientBubble( coords[2] );

      dNdXi[4][0] = LagrangeBasis1::gradientBubble( coords[0] ) *
                    LagrangeBasis1::value( 1, coords[1] ) *
                    LagrangeBasis1::valueBubble( coords[2] );
      dNdXi[4][1] = LagrangeBasis1::valueBubble( coords[0] ) *
                    LagrangeBasis1::gradient( 1, coords[1] ) *
                    LagrangeBasis1::valueBubble( coords[2] );
      dNdXi[4][2] = LagrangeBasis1::valueBubble( coords[0] ) *
                    LagrangeBasis1::value( 1, coords[1] ) *
                    LagrangeBasis1::gradientBubble( coords[2] );

      dNdXi[5][0] = LagrangeBasis1::gradientBubble( coords[0] ) *
                    LagrangeBasis1::valueBubble( coords[1] ) *
                    LagrangeBasis1::value( 1, coords[2] );
      dNdXi[5][1] = LagrangeBasis1::valueBubble( coords[0] ) *
                    LagrangeBasis1::gradientBubble( coords[1] ) *
                    LagrangeBasis1::value( 1, coords[2] );
      dNdXi[5][2] = LagrangeBasis1::valueBubble( coords[0] ) *
                    LagrangeBasis1::valueBubble( coords[1] ) *
                    LagrangeBasis1::gradient( 1, coords[2] );
    }

    /**
     * @brief The parent coordinates for a support point in the xi0 direction.
     * @param linearIndex The linear index of the support point
     * @return
     */
    constexpr static double parentCoords0( int const linearIndex )
    {
      return -1.0 + 2.0 * (linearIndex & 1);
    }

    /**
     * @brief The parent coordinates for a support point in the xi1 direction.
     * @param linearIndex The linear index of the support point
     * @return
     */
    constexpr static double parentCoords1( int const linearIndex )
    {
      return -1.0 + ( linearIndex & 2 );
    }

    /**
     * @brief The parent coordinates for a support point in the xi2 direction.
     * @param linearIndex The linear index of the support point
     * @return
     */
    constexpr static double parentCoords2( int const linearIndex )
    {
      return -1.0 + 0.5 * ( linearIndex & 4 );
    }

  };

};



#endif /* _LAGRANGEBASIS1_HPP_ */
