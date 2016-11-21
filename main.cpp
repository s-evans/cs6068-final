#include <stdio.h>
#include <string>
#include <assert.h>
#include "timer.h"
#include "utils.h"
#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"
#include "boost/iostreams/device/mapped_file.hpp"
#include "serial_huffman.h"
#include "parallel_huffman.h"

typedef enum _options_status_t {
    OPTION_1_SELECTED = 0,
    OPTION_2_SELECTED = 1,
    OPTION_ERROR = 2
} options_status_t;

static options_status_t mutually_exclusive_required( boost::program_options::variables_map const& vm, std::string const& opt1, std::string const& opt2 )
{
    const unsigned int count1 = vm.count( opt1 );
    const unsigned int count2 = vm.count( opt2 );

    if ( ( count1 && count2 ) || !( count1 == 1 || count2 == 1 ) ) {
        return OPTION_ERROR;
    }

    return count1 ? OPTION_1_SELECTED : OPTION_2_SELECTED;
}

int main( int argc, char** argv )
{
    // TODO: parse input command line arguments
    // options for controlling buffer sizes

    // Declare the supported options.
    boost::program_options::options_description operations( "Operations" );
    operations.add_options()
    ( "encode,e", "perform encoding operation" )
    ( "decode,d", "perform decoding operation" );

    boost::program_options::options_description implementations( "Implementation options" );
    implementations.add_options()
    ( "serial,s", "choose serial implementation" )
    ( "parallel,p", "choose parallel implementation" );

    boost::program_options::options_description encodings( "Encoding options" );
    encodings.add_options()
    ( "huffman,f", "use huffman coding" );

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
    desc.add( operations ).add( encodings ).add( implementations ).add( options ).add( input_output );

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
    options_status_t encoding_option = mutually_exclusive_required( vm, "encode", "decode" );

    if ( encoding_option == OPTION_ERROR ) {
        std::cerr << "Please select one operation" << std::endl << std::endl;
        std::cerr << operations << std::endl;
        err = 1;
    }

    options_status_t implementation_option = mutually_exclusive_required( vm, "serial", "parallel" );

    if ( implementation_option == OPTION_ERROR ) {
        std::cerr << "Please select one implementation option" << std::endl << std::endl;
        std::cerr << implementations << std::endl;
        err = 1;
    }

    // TODO: verify that at least one encoding step was chosen

    if ( !vm.count( "huffman" ) ) {
        std::cerr << "Please specify an encoder" << std::endl << std::endl;
        std::cerr << encodings << std::endl;
        err = 1;
    }

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

    // TODO: truncate output file when finished

    unsigned int output_file_size = 0;

    if ( vm.count( "encode" ) ) {
        if ( vm.count( "serial" ) ) {
            serial_huffman_encode( input_file.data(), input_file.size(), output_file.data(), output_file_size );
        } else if ( vm.count( "parallel" ) ) {
            parallel_huffman_encode( input_file.data(), input_file.size(), output_file.data(), output_file_size );
        } else {
            assert( false );
        }
    } else if ( vm.count( "decode" ) ) {
        if ( vm.count( "serial" ) ) {
            serial_huffman_decode( input_file.data(), input_file.size(), output_file.data(), output_file_size );
        } else if ( vm.count( "parallel" ) ) {
            parallel_huffman_decode( input_file.data(), input_file.size(), output_file.data(), output_file_size );
        } else {
            assert( false );
        }
    } else {
        assert( false );
    }

    output_file.close();
    input_file.close();

    // TODO: pipelining
    // do a compression pipeline sort of thing and make it accessible on the command line
    // for example, apply various encoders in any order
    // one encoder could record timing information for us and be a no-op

    // TODO: implement tests

    GpuTimer timer;
    timer.Start();

    timer.Stop();

    return err;
}
