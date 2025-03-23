


#include <gtest/gtest.h>







void testParenthesesOperatorCT()
{
  // using Array = TestCArrayHelper::Array3d;
  // constexpr int na = Array::extent< 0 >();
  // constexpr int nb = Array::extent< 1 >();
  // constexpr int nc = Array::extent< 2 >();

  // pmpl::genericKernelWrapper( [] SHIVA_DEVICE ()
  // {
  //   forSequence< na >( [] ( auto const ica )
  //   {
  //     constexpr int a = decltype(ica)::value;
  //     forSequence< nb >( [ = ] ( auto const icb )
  //     {
  //       constexpr int b = decltype(icb)::value;
  //       forSequence< nc >( [ = ] ( auto const icc )
  //       {
  //         constexpr int c = decltype(icc)::value;
  //         static_assert( pmpl::check( TestCArrayHelper::array3d.operator()< a, b, c >(),
  //                                     3.14159 * CArrayHelper::linearIndexHelper< 2, 3, 4 >::eval( a, b, c ),
  //                                     1.0e-12 ) );
  //         static_assert( pmpl::check( TestCArrayHelper::array3d( a, b, c ),
  //                                     3.14159 * CArrayHelper::linearIndexHelper< 2, 3, 4 >::eval( a, b, c ),
  //                                     1.0e-12 ) );
  //       } );
  //     } );
  //   } );
  // } );
}

void testParenthesesOperatorRT()
{
  // double * data = nullptr;
  // constexpr int N = TestCArrayHelper::Array3d::size();
  // data = new double[N];
  // pmpl::genericKernelWrapper( N, data, [] SHIVA_DEVICE ( double * const kernelData )
  // {
  //   TestCArrayHelper::Array3d const array{ initializer< TestCArrayHelper::Array3d >( std::make_integer_sequence< int, 2 * 3 * 4 >() ) };;
  //   int const na = array.extent< 0 >();
  //   int const nb = array.extent< 1 >();
  //   int const nc = array.extent< 2 >();

  //   for ( int a = 0; a < na; ++a )
  //   {
  //     for ( int b = 0; b < nb; ++b )
  //     {
  //       for ( int c = 0; c < nc; ++c )
  //       {
  //         kernelData[ a * nb * nc + b * nc + c ] = array( a, b, c );
  //       }
  //     }
  //   }
  // } );

  // for ( int a = 0; a < TestCArrayHelper::Array3d::size(); ++a )
  // {
  //   EXPECT_EQ( data[a], 3.14159 * a );
  // }
  // delete [] data;


}


TEST( testIntegrals, testParenthesesOperator )
{
  testParenthesesOperatorCT();
  testParenthesesOperatorRT();
}


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
