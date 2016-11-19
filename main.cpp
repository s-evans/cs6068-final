#include <stdio.h>
#include "timer.h"
#include "utils.h"
#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"

int main( int argc, char** argv )
{
    // TODO: parse input command line arguments
    // input / outupt file paths
    // improve this for mutual exclusion and required options

    // Declare the supported options.
    boost::program_options::options_description desc( "Allowed options" );
    desc.add_options()
        ( "compress,c", "perform compression" )
        ( "decompress,d", "perform decompression" )
        ( "serial,s", "choose serial implementation" )
        ( "parallel,p", "choose parallel implementation" )
        ( "huffman,f", "use huffman coding" )
        ( "timing,t", "show timing information" )
        ( "help,h", "produce help message" )
    ;

    boost::program_options::variables_map vm;
    boost::program_options::store( boost::program_options::parse_command_line( argc, argv, desc ), vm );
    boost::program_options::notify( vm );

    if ( vm.count( "help" ) ) {
        std::cout << desc << "\n";
        return 1;
    }

    // TODO: implement serial compression
    // TODO: implement serial decompression
    // TODO: implement parallel compression
    // TODO: implement parallel decompression

    // TODO: implement tests

    // TODO: get completion working

    GpuTimer timer;
    timer.Start();

    timer.Stop();

    return 0;
}
