#include "boost/filesystem.hpp"
#include "boost/filesystem.hpp"
#include "boost/iostreams/device/mapped_file.hpp"
#include "boost/program_options.hpp"
#include "parallel_huffman.h"
#include "serial_huffman.h"
#include "timer.h"
#include "utils.h"
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/mman.h>

typedef enum _options_status_t {
    OPTION_1_SELECTED = 0,
    OPTION_2_SELECTED = 1,
    OPTION_ERROR = 2
} options_status_t;

int main( int argc, char** argv )
{
    // Declare the supported options.
    boost::program_options::options_description implementations( "Implementation options" );
    implementations.add_options()
    ( "serial,s", "choose serial implementation" )
    ( "parallel,p", "choose parallel implementation" );

    boost::program_options::options_description options( "Other options" );
    options.add_options()
    ( "timing,t", "produce timing information" )
    ( "help,h", "produce help message" );

    boost::filesystem::path input_file_path;
    boost::filesystem::path output_file_path;

    boost::program_options::options_description input_output( "Input / output options" );
    input_output.add_options()
    ( "input-file,i", boost::program_options::value<boost::filesystem::path>( &input_file_path ),  "path to input file" )
    ( "output-file,o", boost::program_options::value<boost::filesystem::path>( &output_file_path ), "path to output file" );

    boost::program_options::options_description desc( "Usage" );
    desc.add( implementations ).add( options ).add( input_output );

    boost::program_options::positional_options_description positional;
    positional.add( "input-file", 1 ).add( "output-file", 1 );

    boost::program_options::variables_map vm;
    boost::program_options::store( boost::program_options::parse_command_line( argc, argv, desc ), vm );
    boost::program_options::store( boost::program_options::command_line_parser( argc, argv ).options( desc ).positional( positional ).run(), vm );
    boost::program_options::notify( vm );

    if ( vm.count( "help" ) ) {
        std::cerr << desc << std::endl;
        return 1;
    }

    unsigned int err = 0;

    if ( !( vm.count( "output-file" ) && vm.count( "input-file" ) ) ) {
        std::cerr << "Please specify an input / output file" << std::endl << std::endl;
        std::cerr << input_output << std::endl;
        err = 1;
    }

    if ( vm.count( "input-file" ) ) {
        if ( !boost::filesystem::exists( input_file_path ) ) {
            std::cerr << "Input file " << input_file_path << " does not exist" << std::endl;
            err = 1;
        } else if ( !boost::filesystem::is_regular_file( input_file_path ) ) {
            std::cerr << "Input file " << input_file_path << " is not a regular file" << std::endl;
            err = 1;
        } else if ( boost::filesystem::is_empty( input_file_path ) ) {
            std::cerr << "Input file " << input_file_path << " is empty" << std::endl;
            err = 1;
        }
    }

    if ( vm.count( "output-file" ) ) {
        if ( boost::filesystem::exists( output_file_path ) ) {
            if ( !boost::filesystem::is_regular_file( output_file_path ) ) {
                std::cerr << "Output file " << output_file_path << " is not a regular file" << std::endl;
                err = 1;
            } else if ( output_file_path == input_file_path ) {
                std::cerr << "Output file path " << output_file_path << " is the same as input file path" << std::endl;
                err = 1;
            }
        }
    }

    if ( err ) {
        return err;
    }

    boost::iostreams::mapped_file input_file( input_file_path );

    if ( !input_file.is_open() ) {
        std::cerr << "Failed to open input file " << input_file_path << std::endl;
        return 1;
    }

    boost::iostreams::mapped_file_params output_file_params;
    output_file_params.path = output_file_path.string();
    output_file_params.new_file_size = input_file.size();

    boost::iostreams::mapped_file_sink output_file( output_file_params );

    if ( !output_file.is_open() ) {
        std::cerr << "Failed to open output file " << output_file_path << std::endl;
        return 1;
    }

    unsigned int output_file_size = 0;

    GpuTimer timer;

    /* std::cerr << "input_file.size(): " << input_file.size() << std::endl; */

    if ( vm.count( "serial" ) ) {
        timer.Start();
        serial_huffman_encode(
                reinterpret_cast<unsigned char*>( input_file.data() ), input_file.size(),
                reinterpret_cast<unsigned char*>( output_file.data() ), output_file_size );
        timer.Stop();
    } else {
        timer.Start();
        parallel_huffman_encode(
                reinterpret_cast<unsigned char*>( input_file.data() ), input_file.size(),
                reinterpret_cast<unsigned char*>( output_file.data() ), output_file_size );
        timer.Stop();
    }

    if ( vm.count( "timing" ) ) {
        std::cerr << "Code ran in " << timer.Elapsed() << " msecs" << std::endl;
    }

    if ( msync( output_file.data(), output_file.size(), MS_SYNC ) ) {
        std::cerr << "Failed to flush output data: " << strerror( errno ) << std::endl;
    }

    output_file.close();
    input_file.close();

    if ( truncate( output_file_path.string().c_str(), output_file_size ) ) {
        std::cerr << "Failed to truncate output file: " << strerror( errno ) << std::endl;
    }

    return err;
}
